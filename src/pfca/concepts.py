"""Fuzzy concept formation from correlated features.

Concepts are formed by fuzzy c means clustering applied to the features rather
than to the instances. Every feature is represented by its profile of absolute
correlations with all features, so that two features with similar correlation
patterns are close, or by a factor analytic loading matrix normalized row wise
when a latent structure is assumed. An a priori grouping, for example the
subscales of a questionnaire, can be included as an additional candidate
partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import FactorAnalysis

from pfca.utils import (
    as_index_groups,
    check_membership,
    concept_fuzziness,
    crisp_membership,
    fuzzy_partition_coefficient,
    harden,
    make_rng,
    normalize_rows,
)


def correlation_matrix(X: np.ndarray, method: str = "pearson") -> np.ndarray:
    """Feature by feature correlation matrix with constant features set to zero correlation."""
    X = np.asarray(X, dtype=float)
    if method == "spearman":
        R = spearmanr(X).correlation
        R = np.atleast_2d(R)
    elif method == "pearson":
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.corrcoef(X, rowvar=False)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    return R


def correlation_profile_embedding(X: np.ndarray, method: str = "pearson") -> np.ndarray:
    """Embed every feature as its vector of absolute correlations with all features."""
    return np.abs(correlation_matrix(X, method))


def fuzzy_c_means(
    Z: np.ndarray,
    n_clusters: int,
    fuzzifier: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-7,
    n_init: int = 5,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fuzzy c means clustering of the rows of Z.

    Returns the membership matrix U of shape (n_rows, n_clusters) whose rows
    sum to one, the cluster centers and the final objective value. Several
    random initializations are run and the solution with the smallest
    objective is kept. Clusters are ordered by the index of their first
    dominant row so that the output is deterministic for a given seed.
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    K = int(n_clusters)
    if K < 1:
        raise ValueError("n_clusters must be positive.")
    if fuzzifier <= 1.0:
        raise ValueError("The fuzzifier exponent must be greater than one.")
    if K >= n:
        raise ValueError("n_clusters must be smaller than the number of rows to cluster.")
    rng = make_rng(random_state)
    expo = 2.0 / (fuzzifier - 1.0)
    best = None
    for _ in range(max(1, n_init)):
        U = rng.dirichlet(np.ones(K), size=n)
        obj_prev = np.inf
        for _ in range(max_iter):
            Um = U**fuzzifier
            centers = (Um.T @ Z) / np.maximum(Um.sum(axis=0)[:, None], 1e-12)
            D2 = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            D2 = np.maximum(D2, 1e-12)
            inv = D2 ** (-expo / 2.0)
            U_new = inv / inv.sum(axis=1, keepdims=True)
            obj = float((U_new**fuzzifier * D2).sum())
            shift = np.abs(U_new - U).max()
            U = U_new
            if shift < tol or abs(obj_prev - obj) < tol * max(1.0, abs(obj)):
                break
            obj_prev = obj
        if best is None or obj < best[2]:
            best = (U, centers, obj)
    U, centers, obj = best
    order = np.argsort([int(np.argmax(U[:, k])) if np.any(harden(U) == k) else n + k for k in range(K)])
    first_rows = []
    labels = harden(U)
    for k in range(K):
        rows = np.where(labels == k)[0]
        first_rows.append(rows.min() if rows.size else n + k)
    order = np.argsort(first_rows, kind="stable")
    return U[:, order], centers[order], obj


def loading_membership(X: np.ndarray, n_factors: int, random_state=None) -> np.ndarray:
    """Membership matrix from absolute factor analytic loadings normalized row wise."""
    X = np.asarray(X, dtype=float)
    fa = FactorAnalysis(n_components=int(n_factors), random_state=random_state)
    fa.fit(X)
    L = np.abs(fa.components_.T)
    L = np.where(L.sum(axis=1, keepdims=True) <= 1e-12, 1.0 / n_factors, L)
    return normalize_rows(L)


def apriori_membership(groups, n_features: int, feature_names: Sequence[str] | None = None) -> np.ndarray:
    """Crisp membership matrix from an a priori grouping of features.

    Features that appear in no group are collected in one additional concept.
    """
    idx_groups = as_index_groups(groups, n_features, feature_names)
    labels = -np.ones(n_features, dtype=int)
    for k, g in enumerate(idx_groups):
        labels[g] = k
    if np.any(labels < 0):
        labels[labels < 0] = len(idx_groups)
    return crisp_membership(labels)


def consensus_membership(
    X: np.ndarray,
    n_clusters: int,
    fuzzifier: float = 2.0,
    n_resamples: int = 20,
    correlation: str = "pearson",
    random_state=None,
    n_init: int = 3,
) -> np.ndarray:
    """Consensus fuzzy c means over bootstrap resamples of the instances.

    The co-membership matrix (probability that two features share the dominant
    concept) is averaged over resamples, and fuzzy c means is run once more on
    the rows of the consensus matrix. This mitigates the instability of the
    partition on small samples.
    """
    X = np.asarray(X, dtype=float)
    rng = make_rng(random_state)
    n, d = X.shape
    C = np.zeros((d, d))
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        Z = correlation_profile_embedding(X[idx], correlation)
        U, _, _ = fuzzy_c_means(Z, n_clusters, fuzzifier, n_init=n_init, random_state=rng)
        C += U @ U.T
    C /= n_resamples
    U, _, _ = fuzzy_c_means(C, n_clusters, fuzzifier, n_init=n_init, random_state=rng)
    return U


@dataclass
class ConceptPartition:
    """A fuzzy partition of the features into concepts.

    Attributes
    ----------
    U : ndarray of shape (n_features, n_concepts)
        Membership matrix with rows summing to one.
    name : str
        Identifier used in the configuration tables.
    method : str
        How the partition was obtained ('fcm', 'loading', 'apriori', 'identity').
    fuzzifier : float or None
    feature_names : list of str
    """

    U: np.ndarray
    name: str
    method: str = "fcm"
    fuzzifier: float | None = None
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.U = check_membership(self.U)
        if not self.feature_names:
            self.feature_names = [f"x{j}" for j in range(self.U.shape[0])]

    @property
    def n_concepts(self) -> int:
        return int(self.U.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.U.shape[0])

    @property
    def labels(self) -> np.ndarray:
        return harden(self.U)

    @property
    def fuzziness(self) -> np.ndarray:
        return concept_fuzziness(self.U)

    @property
    def partition_coefficient(self) -> float:
        return fuzzy_partition_coefficient(self.U)

    def hardened(self) -> "ConceptPartition":
        return ConceptPartition(crisp_membership(self.labels, self.n_concepts), self.name + "_crisp", "crisp", self.fuzzifier, self.feature_names)

    def members(self, k: int, threshold: float = 0.5) -> list[str]:
        idx = np.where(self.U[:, k] >= threshold)[0]
        if idx.size == 0:
            idx = np.array([int(np.argmax(self.U[:, k]))])
        return [self.feature_names[j] for j in idx]

    def concept_names(self, threshold: float = 0.5, max_members: int = 4) -> list[str]:
        out = []
        for k in range(self.n_concepts):
            mem = self.members(k, threshold)
            label = ", ".join(mem[:max_members]) + (", ..." if len(mem) > max_members else "")
            out.append(f"C{k + 1} ({label})")
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.U, index=self.feature_names, columns=[f"C{k + 1}" for k in range(self.n_concepts)])


def identity_partition(n_features: int, feature_names: Sequence[str] | None = None) -> ConceptPartition:
    """One crisp concept per feature; PFCA reduces to SHAP on this partition."""
    return ConceptPartition(np.eye(n_features), "identity", "identity", None, list(feature_names) if feature_names else [])


class ConceptFormer:
    """Generate candidate fuzzy concept partitions of the features.

    Parameters
    ----------
    n_concepts_grid : iterable of int
        Candidate numbers of concepts K.
    fuzzifier_grid : iterable of float
        Candidate fuzzifier exponents.
    distance : {'correlation', 'loading'}
        'correlation' clusters the absolute correlation profiles with fuzzy c
        means; 'loading' derives memberships from factor analytic loadings.
    correlation : {'pearson', 'spearman'}
    apriori_groups : dict or list, optional
        A priori grouping of features, included as an extra candidate.
    consensus : bool
        Use consensus clustering over resamples of the instances.
    crisp : bool
        Harden every candidate partition (ablation).
    n_init : int
    random_state : int or None
    """

    def __init__(
        self,
        n_concepts_grid: Iterable[int] = (2, 3, 4, 5, 6),
        fuzzifier_grid: Iterable[float] = (1.5, 2.0, 2.5),
        distance: str = "correlation",
        correlation: str = "pearson",
        apriori_groups=None,
        consensus: bool = False,
        n_consensus: int = 20,
        crisp: bool = False,
        n_init: int = 5,
        random_state=None,
    ):
        self.n_concepts_grid = tuple(int(k) for k in n_concepts_grid)
        self.fuzzifier_grid = tuple(float(m) for m in fuzzifier_grid)
        self.distance = distance
        self.correlation = correlation
        self.apriori_groups = apriori_groups
        self.consensus = consensus
        self.n_consensus = n_consensus
        self.crisp = crisp
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, feature_names: Sequence[str] | None = None):
        X = np.asarray(X, dtype=float)
        self.X_ = X
        self.n_features_ = X.shape[1]
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(self.n_features_)]
        self.embedding_ = correlation_profile_embedding(X, self.correlation)
        self.correlation_ = correlation_matrix(X, self.correlation)
        self._cache: dict[tuple, ConceptPartition] = {}
        self.candidates_ = []
        for K in self.n_concepts_grid:
            if K >= self.n_features_ or K < 1:
                continue
            if self.distance == "loading":
                self.candidates_.append(self.partition(K, None))
            else:
                for m in self.fuzzifier_grid:
                    self.candidates_.append(self.partition(K, m))
        if self.apriori_groups is not None:
            self.candidates_.append(self.apriori_partition())
        if not self.candidates_:
            raise ValueError("No candidate partition could be formed; check n_concepts_grid against the number of features.")
        return self

    def apriori_partition(self) -> ConceptPartition:
        U = apriori_membership(self.apriori_groups, self.n_features_, self.feature_names_)
        return ConceptPartition(U, "apriori", "apriori", None, self.feature_names_)

    def partition(self, n_concepts: int, fuzzifier: float | None) -> ConceptPartition:
        """Return (and cache) the candidate partition with K concepts and the given fuzzifier."""
        key = (int(n_concepts), None if fuzzifier is None else round(float(fuzzifier), 4))
        if key in self._cache:
            return self._cache[key]
        K = int(n_concepts)
        seed = make_rng(self.random_state).integers(0, 2**31 - 1) if self.random_state is not None else None
        if self.distance == "loading":
            U = loading_membership(self.X_, K, random_state=seed)
            name, method = f"loading_K{K}", "loading"
        else:
            m = 2.0 if fuzzifier is None else float(fuzzifier)
            if self.consensus:
                U = consensus_membership(self.X_, K, m, self.n_consensus, self.correlation, random_state=seed, n_init=self.n_init)
                method = "consensus_fcm"
            else:
                U, _, _ = fuzzy_c_means(self.embedding_, K, m, n_init=self.n_init, random_state=seed)
                method = "fcm"
            name = f"fcm_K{K}_m{m:g}"
        part = ConceptPartition(U, name, method, fuzzifier, self.feature_names_)
        if self.crisp:
            part = ConceptPartition(crisp_membership(part.labels, K), name + "_crisp", method + "_crisp", fuzzifier, self.feature_names_)
        self._cache[key] = part
        return part
