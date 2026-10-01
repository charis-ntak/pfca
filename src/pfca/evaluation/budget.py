"""Wall clock time and peak memory measurement (cost metric of Table 1).

Timing and memory tracing are kept separate. :func:`measure` records the wall
time of a block with tracemalloc off unless memory tracing is requested,
because tracemalloc records a traceback for every Python allocation and
inflates pure Python work (the PFCA selection loop) several times more than
compiled work (tree fitting and TreeSHAP), which would bias the runtime
comparison between methods. The memory column of the cost table is obtained
from a dedicated run with ``trace_memory=True``.
"""

from __future__ import annotations

import resource
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Budget:
    """Measurements of one block.

    wall_seconds : float
        Wall clock time of the block.
    peak_python_mb : float
        Peak size of the Python heap during the block as traced by
        tracemalloc; zero when memory tracing was not requested.
    max_rss_mb : float
        Maximum resident set size of the process at the end of the block. It
        is a high water mark over the whole process, not over the block alone.
    rss_increase_mb : float
        Increase of that high water mark during the block; zero when the
        block stayed below the earlier maximum.
    """

    wall_seconds: float = 0.0
    peak_python_mb: float = 0.0
    max_rss_mb: float = 0.0
    rss_increase_mb: float = 0.0


# One entry per active tracing block: the largest peak observed before a block
# nested in it reset the tracemalloc peak, so that nested blocks stay correct.
_carried_peaks: list[float] = []


def max_rss_mb() -> float:
    """Maximum resident set size of the process in megabytes.

    ``ru_maxrss`` is reported in kilobytes on Linux and in bytes on macOS.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024.0


@contextmanager
def measure(trace_memory: bool = False):
    """Context manager recording wall time and, on request, the Python heap peak.

    With ``trace_memory=False`` (the default) the wall time and the resident
    set size are recorded and tracemalloc is not touched, so that every method
    is timed under the same conditions. With ``trace_memory=True`` tracemalloc
    is started for the block; when it is already tracing (an enclosing block
    or the caller started it) only its peak is reset and tracing is left
    running afterwards, so that blocks may be nested. The peak of a block
    includes the peaks of the blocks nested in it.

    >>> with measure() as b:
    ...     run()
    >>> b.wall_seconds, b.max_rss_mb
    >>> with measure(trace_memory=True) as b:
    ...     run()
    >>> b.peak_python_mb
    """
    budget = Budget()
    started_here = False
    if trace_memory:
        if tracemalloc.is_tracing():
            _, peak_before = tracemalloc.get_traced_memory()
            if _carried_peaks:
                _carried_peaks[-1] = max(_carried_peaks[-1], float(peak_before))
            tracemalloc.reset_peak()
        else:
            tracemalloc.start()
            started_here = True
        _carried_peaks.append(0.0)
    rss_before = max_rss_mb()
    start = time.perf_counter()
    try:
        yield budget
    finally:
        budget.wall_seconds = time.perf_counter() - start
        if trace_memory:
            _, peak = tracemalloc.get_traced_memory()
            peak = max(float(peak), _carried_peaks.pop())
            if started_here:
                tracemalloc.stop()
            elif _carried_peaks:
                _carried_peaks[-1] = max(_carried_peaks[-1], peak)
            budget.peak_python_mb = peak / 2**20
        budget.max_rss_mb = max_rss_mb()
        budget.rss_increase_mb = max(0.0, budget.max_rss_mb - rss_before)
