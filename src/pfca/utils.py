"""Shared helpers: random state handling, membership matrix checks, entropies."""

from __future__ import annotations

import numbers
from typing import Iterable, Sequence

import numpy as np


def make_rng(random_state) -> np.random.Generator:
    """Return a numpy Generator from an int, a Generator, a SeedSequence or None."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.SeedSequence):
        return np.random.default_rng(random_state)
    if random_state is None or isinstance(random_state, numbers.Integral):
        return np.random.default_rng(random_state)
    raise TypeError(f"Unsupported random_state type: {type(random_state)!r}")


def spawn_seeds(random_state, n: int) -> list[int]:
    """Derive n independent integer seeds from a random state in a reproducible way."""
    rng = make_rng(random_state)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]


def check_membership(U: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Validate a membership matrix of shape (n_features, n_concepts).

    Rows must sum to one and all entries must lie in the unit interval.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError("Membership matrix must be two dimensional (n_features, n_concepts).")
    if np.any(U < -atol) or np.any(U > 1 + atol):
        raise ValueError("Membership degrees must lie in [0, 1].")
    row_sums = U.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError("Membership degrees must sum to one across concepts for every feature.")
    return np.clip(U, 0.0, 1.0)


def normalize_rows(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale every row of a nonnegative matrix so that it sums to one."""
    A = np.asarray(A, dtype=float)
    s = A.sum(axis=1, keepdims=True)
    s = np.where(s < eps, 1.0, s)
    return A / s


def harden(U: np.ndarray) -> np.ndarray:
    """Return the crisp label of every feature (concept of maximum membership)."""
    return np.argmax(np.asarray(U), axis=1)


def crisp_membership(labels: Sequence[int], n_concepts: int | None = None) -> np.ndarray:
    """Build a crisp membership matrix from integer labels."""
    labels = np.asarray(labels, dtype=int)
    K = int(labels.max()) + 1 if n_concepts is None else int(n_concepts)
    U = np.zeros((labels.shape[0], K))
    U[np.arange(labels.shape[0]), labels] = 1.0
    return U


def row_entropy(U: np.ndarray, normalize: bool = True, eps: float = 1e-12) -> np.ndarray:
    """Entropy of every feature's membership distribution across concepts.

    With normalize=True the entropy is divided by log(K) so that it lies in
    [0, 1]; a crisp feature has entropy zero.
    """
    U = np.asarray(U, dtype=float)
    K = U.shape[1]
    P = np.clip(U, eps, 1.0)
    H = -(U * np.log(P)).sum(axis=1)
    if normalize and K > 1:
        H = H / np.log(K)
    elif normalize:
        H = np.zeros_like(H)
    return H


def concept_fuzziness(U: np.ndarray) -> np.ndarray:
    """Fuzziness of every concept: membership weighted mean of the row entropies.

    Returns an array of shape (n_concepts,) with values in [0, 1]. A crisp
    partition has fuzziness zero for every concept.
    """
    U = np.asarray(U, dtype=float)
    H = row_entropy(U, normalize=True)
    mass = U.sum(axis=0)
    mass = np.where(mass <= 0, 1.0, mass)
    return (U * H[:, None]).sum(axis=0) / mass


def fuzzy_partition_coefficient(U: np.ndarray) -> float:
    """Bezdek's fuzzy partition coefficient, in [1/K, 1]; one means crisp."""
    U = np.asarray(U, dtype=float)
    return float((U**2).sum() / U.shape[0])


def as_index_groups(groups, n_features: int, feature_names: Iterable[str] | None = None) -> list[np.ndarray]:
    """Convert group specifications (indices or names) into a list of index arrays."""
    names = list(feature_names) if feature_names is not None else None
    if isinstance(groups, dict):
        groups = list(groups.values())
    out = []
    for g in groups:
        idx = []
        for item in g:
            if isinstance(item, str):
                if names is None:
                    raise ValueError("Feature names are required to resolve string group members.")
                idx.append(names.index(item))
            else:
                idx.append(int(item))
        arr = np.asarray(sorted(set(idx)), dtype=int)
        if arr.size and (arr.min() < 0 or arr.max() >= n_features):
            raise ValueError("Group index out of range.")
        out.append(arr)
    return out
