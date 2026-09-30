"""Wall clock time and peak memory measurement (cost metric of Table 1)."""

from __future__ import annotations

import resource
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Budget:
    wall_seconds: float = 0.0
    peak_python_mb: float = 0.0
    max_rss_mb: float = 0.0


@contextmanager
def measure():
    """Context manager recording wall time, Python heap peak and process max RSS.

    >>> with measure() as b:
    ...     run()
    >>> b.wall_seconds, b.peak_python_mb, b.max_rss_mb
    """
    budget = Budget()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        yield budget
    finally:
        budget.wall_seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        budget.peak_python_mb = peak / 2**20
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        budget.max_rss_mb = rss / 1024.0 if rss > 2**24 else rss / 1024.0
