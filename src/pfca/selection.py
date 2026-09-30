"""Multiobjective selection of the explanation configuration.

An explanation configuration consists of the number of concepts K, the
fuzzifier exponent, the sparsity level (number of concepts retained after the
alpha cut) and the alpha level of the cut. Every candidate is scored on three
objectives (Equation 4): fidelity loss of a surrogate fitted on the retained
concept attributions and evaluated out of sample, complexity as the number of
retained concepts weighted by the fuzziness of their memberships, and
instability as one minus the mean rank correlation of concept importance
across the pool. The Pareto front is obtained by exhaustive evaluation on a
grid or by NSGA II, a default solution is selected by a knee point criterion,
and a selection based membership of every feature and concept is derived from
the front.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from pfca.concepts import ConceptPartition
from pfca.fuzzification import alpha_cut_array
from pfca.utils import concept_fuzziness


@dataclass(frozen=True)
class ExplanationConfiguration:
    """Decision variables of the multiobjective problem."""

    partition: str
    n_concepts: int
    fuzzifier: float | None
    sparsity: int
    alpha: float

    def as_dict(self) -> dict:
        return {
            "partition": self.partition,
            "n_concepts": self.n_concepts,
            "fuzzifier": self.fuzzifier,
            "sparsity": self.sparsity,
            "alpha": self.alpha,
        }


@dataclass
class PartitionPool:
    """Concept level quantities of one candidate partition.

    Attributes
    ----------
    pool : ndarray of shape (P, n_explain, K)
        Concept attributions of every pool member (P = B * M).
    quantiles : ndarray of shape (n_explain, K, 5)
        Fuzzy attribution of every instance and concept as five quantiles.
    global_quantiles : ndarray of shape (K, 5)
        Fuzzy global importance of every concept.
    """

    partition: ConceptPartition
    pool: np.ndarray
    quantiles: np.ndarray
    global_quantiles: np.ndarray


def retained_concepts(quantiles: np.ndarray, global_quantiles: np.ndarray, alpha: float, sparsity: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the alpha cut and the sparsity cap.

    A concept is active for an instance when the alpha cut of its fuzzy
    attribution excludes zero. Concepts active for at least one instance pass
    the cut; they are ranked by the lower endpoint of the alpha cut of their
    fuzzy global importance and the first ``sparsity`` are retained.

    Returns the retained concept indices in rank order, the activity of every
    concept (fraction of instances for which it is active) and the passing mask.
    """
    lo, hi = alpha_cut_array(quantiles, alpha)
    active = (lo > 0.0) | (hi < 0.0)
    activity = active.mean(axis=0)
    passing = activity > 0.0
    glo, _ = alpha_cut_array(global_quantiles, alpha)
    K = quantiles.shape[1]
    order = np.lexsort((-activity, -glo))
    order = np.array([k for k in order if passing[k]], dtype=int)
    retained = order[: max(int(sparsity), 0)]
    return retained, activity, passing


def complexity(U: np.ndarray, retained: np.ndarray) -> float:
    """Number of retained concepts weighted by their fuzziness: sum_k (1 + F_k)."""
    if len(retained) == 0:
        return 0.0
    F = concept_fuzziness(U)
    return float(np.sum(1.0 + F[np.asarray(retained, dtype=int)]))


def fidelity_loss(
    concept_values: np.ndarray,
    retained: np.ndarray,
    output: np.ndarray,
    surrogate: str = "linear",
    cv: int = 5,
    random_state: int | None = 0,
) -> float:
    """Out of sample error of a surrogate reproducing the model output from retained concepts.

    Returns the cross validated mean squared error divided by the variance of
    the output, that is one minus the cross validated R squared. When no
    concept is retained the surrogate is the constant mean and the loss is one.
    """
    output = np.asarray(output, dtype=float)
    n = output.shape[0]
    var = float(np.var(output))
    if var <= 1e-18:
        return 0.0
    retained = np.asarray(retained, dtype=int)
    if retained.size == 0 or n < 4:
        return 1.0
    Z = np.asarray(concept_values, dtype=float)[:, retained]
    folds = min(cv, n)
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    sse = 0.0
    for tr, te in kf.split(Z):
        if surrogate == "linear":
            mdl = Ridge(alpha=1e-6)
        elif surrogate == "tree":
            mdl = GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=random_state)
        else:
            raise ValueError("surrogate must be 'linear' or 'tree'")
        mdl.fit(Z[tr], output[tr])
        pred = mdl.predict(Z[te])
        sse += float(np.sum((output[te] - pred) ** 2))
    return sse / (n * var)


def instability(pool: np.ndarray, retained: np.ndarray, max_pairs: int = 2000, random_state: int | None = 0) -> float:
    """One minus the mean pairwise Spearman correlation of concept importance across the pool.

    Importance of concept k for pool member p is the mean absolute attribution
    across instances. With fewer than two retained concepts the ranking is
    trivially stable and the instability is zero by convention.
    """
    retained = np.asarray(retained, dtype=int)
    if retained.size < 2:
        return 0.0
    imp = np.abs(pool[:, :, retained]).mean(axis=1)
    P = imp.shape[0]
    if P < 2:
        return 0.0
    ranks = np.apply_along_axis(rankdata, 1, imp)
    pairs = list(combinations(range(P), 2))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in sel]
    rc = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.sqrt((rc**2).sum(axis=1))
    vals = []
    for i, j in pairs:
        den = norms[i] * norms[j]
        if den <= 1e-12:
            vals.append(1.0 if np.allclose(ranks[i], ranks[j]) else 0.0)
        else:
            vals.append(float(rc[i] @ rc[j] / den))
    return float(round(float(np.clip(1.0 - np.mean(vals), 0.0, 2.0)), 12))


def pareto_mask(F: np.ndarray) -> np.ndarray:
    """Boolean mask of the non dominated rows of F (all objectives minimized)."""
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated_by_i = np.all(F >= F[i], axis=1) & np.any(F > F[i], axis=1)
        mask[dominated_by_i] = False
        dominates_i = np.all(F[mask] <= F[i], axis=1) & np.any(F[mask] < F[i], axis=1)
        if np.any(dominates_i):
            mask[i] = False
    return mask


def knee_point(F: np.ndarray, mask: np.ndarray | None = None, method: str = "utopia") -> int:
    """Index of the knee point of a Pareto front.

    'utopia' returns the front member closest, after min max normalization
    across the front, to the ideal point. 'hyperplane' returns the member with
    the largest distance below the hyperplane through the extreme points of the
    front (the members minimizing each objective).
    """
    F = np.asarray(F, dtype=float)
    if mask is None:
        mask = pareto_mask(F)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("Empty Pareto front.")
    if idx.size == 1:
        return int(idx[0])
    G = F[idx]
    lo, hi = G.min(axis=0), G.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    N = (G - lo) / span
    if method == "utopia":
        dist = np.sqrt((N**2).sum(axis=1))
        return int(idx[int(np.argmin(dist))])
    if method == "hyperplane":
        extremes = np.unique([int(np.argmin(N[:, j])) for j in range(N.shape[1])])
        if extremes.size < N.shape[1]:
            dist = np.sqrt((N**2).sum(axis=1))
            return int(idx[int(np.argmin(dist))])
        E = N[extremes]
        A = np.hstack([E, np.ones((E.shape[0], 1))])
        _, _, vh = np.linalg.svd(A)
        coef = vh[-1]
        normal, c0 = coef[:-1], coef[-1]
        if normal.sum() < 0:
            normal, c0 = -normal, -c0
        signed = -(N @ normal + c0) / max(np.linalg.norm(normal), 1e-12)
        return int(idx[int(np.argmax(signed))])
    raise ValueError("method must be 'utopia' or 'hyperplane'")


def selection_membership(configs: list[ExplanationConfiguration], retained_sets: list[np.ndarray], partitions: dict[str, ConceptPartition], mask: np.ndarray) -> np.ndarray:
    """Selection based membership of every feature.

    For every Pareto optimal configuration a feature is retained with degree
    equal to its total membership in the retained concepts; the selection
    membership is the mean of that degree over the front and lies in [0, 1].
    """
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("Empty Pareto front.")
    d = next(iter(partitions.values())).n_features
    acc = np.zeros(d)
    for i in idx:
        U = partitions[configs[i].partition].U
        r = np.asarray(retained_sets[i], dtype=int)
        if r.size:
            acc += U[:, r].sum(axis=1)
    return np.clip(acc / idx.size, 0.0, 1.0)


def concept_selection_membership(U: np.ndarray, feature_membership: np.ndarray) -> np.ndarray:
    """Membership weighted average of feature selection memberships per concept."""
    mass = U.sum(axis=0)
    mass = np.where(mass <= 0, 1.0, mass)
    return np.clip((U * feature_membership[:, None]).sum(axis=0) / mass, 0.0, 1.0)


@dataclass
class SelectionResult:
    table: pd.DataFrame
    configurations: list[ExplanationConfiguration]
    retained_sets: list[np.ndarray]
    objectives: np.ndarray
    pareto: np.ndarray
    knee: int
    feature_selection_membership: np.ndarray

    @property
    def front(self) -> pd.DataFrame:
        return self.table[self.table["pareto"]].copy()

    @property
    def knee_configuration(self) -> ExplanationConfiguration:
        return self.configurations[self.knee]


class ParetoSelector:
    """Solve the multiobjective selection problem of Equation 4.

    Parameters
    ----------
    alpha_grid : iterable of float
    sparsity_grid : iterable of int or None
        Candidate numbers of retained concepts; None uses 1 to K.
    solver : {'grid', 'nsga2'}
    knee_method : {'utopia', 'hyperplane'}
    surrogate : {'linear', 'tree'}
    cv : int
    nsga_pop_size, nsga_generations : int
    random_state : int or None
    """

    def __init__(
        self,
        alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        sparsity_grid=None,
        solver: str = "grid",
        knee_method: str = "utopia",
        surrogate: str = "linear",
        cv: int = 5,
        nsga_pop_size: int = 40,
        nsga_generations: int = 30,
        random_state: int | None = 0,
    ):
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.sparsity_grid = None if sparsity_grid is None else tuple(int(s) for s in sparsity_grid)
        self.solver = solver
        self.knee_method = knee_method
        self.surrogate = surrogate
        self.cv = cv
        self.nsga_pop_size = nsga_pop_size
        self.nsga_generations = nsga_generations
        self.random_state = random_state

    # ------------------------------------------------------------------
    def evaluate(self, pp: PartitionPool, sparsity: int, alpha: float, output: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        retained, activity, passing = retained_concepts(pp.quantiles, pp.global_quantiles, alpha, sparsity)
        medians = pp.quantiles[..., 2]
        f1 = fidelity_loss(medians, retained, output, self.surrogate, self.cv, self.random_state)
        f2 = complexity(pp.partition.U, retained)
        f3 = instability(pp.pool, retained, random_state=self.random_state)
        info = {"n_retained": int(retained.size), "n_passing": int(passing.sum()), "mean_activity": float(activity[retained].mean()) if retained.size else 0.0}
        return retained, np.array([f1, f2, f3]), info

    def _finish(self, configs, retained_sets, F, infos, partitions) -> SelectionResult:
        F = np.asarray(F, dtype=float)
        feasible = np.array([r.size > 0 for r in retained_sets])
        if not np.any(feasible):
            raise RuntimeError("No configuration retains at least one concept; lower the alpha grid or increase the number of resamples.")
        Ff = F.copy()
        Ff[~feasible] = np.inf
        mask = pareto_mask(Ff) & feasible
        knee = knee_point(Ff, mask, self.knee_method)
        rows = []
        for i, (cfg, r, f, info) in enumerate(zip(configs, retained_sets, F, infos)):
            row = cfg.as_dict()
            row.update({"fidelity_loss": f[0], "complexity": f[1], "instability": f[2]})
            row.update(info)
            row["retained"] = tuple(int(k) for k in r)
            row["feasible"] = bool(feasible[i])
            row["pareto"] = bool(mask[i])
            row["knee"] = i == knee
            rows.append(row)
        table = pd.DataFrame(rows)
        fsm = selection_membership(configs, retained_sets, partitions, mask)
        return SelectionResult(table, configs, retained_sets, F, mask, knee, fsm)

    def run_grid(self, pools: dict[str, PartitionPool], output: np.ndarray) -> SelectionResult:
        configs, retained_sets, F, infos = [], [], [], []
        for name, pp in pools.items():
            K = pp.partition.n_concepts
            sgrid = self.sparsity_grid if self.sparsity_grid is not None else tuple(range(1, K + 1))
            for alpha in self.alpha_grid:
                for s in sgrid:
                    if s < 1 or s > K:
                        continue
                    r, f, info = self.evaluate(pp, s, alpha, output)
                    configs.append(ExplanationConfiguration(name, K, pp.partition.fuzzifier, int(s), float(alpha)))
                    retained_sets.append(r)
                    F.append(f)
                    infos.append(info)
        partitions = {name: pp.partition for name, pp in pools.items()}
        return self._finish(configs, retained_sets, F, infos, partitions)

    def run_nsga2(
        self,
        pool_provider: Callable[[int, float | None, str], PartitionPool],
        n_concepts_bounds: tuple[int, int],
        fuzzifier_options: tuple[float, ...],
        output: np.ndarray,
        extra_partitions: tuple[str, ...] = (),
    ) -> SelectionResult:
        """NSGA II over (partition source, K, fuzzifier, sparsity, alpha).

        ``pool_provider(K, fuzzifier, source)`` must return the PartitionPool of
        the requested candidate; sources are 'fcm' or one of ``extra_partitions``
        (for example 'apriori'), whose K and fuzzifier are ignored.
        """
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.mixed import MixedVariableDuplicateElimination, MixedVariableMating, MixedVariableSampling
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.core.variable import Choice, Integer, Real
        from pymoo.optimize import minimize

        selector = self
        Kmin, Kmax = n_concepts_bounds
        sources = ("fcm",) + tuple(extra_partitions)
        evaluated: dict[tuple, tuple] = {}
        partitions: dict[str, ConceptPartition] = {}

        def eval_key(source, K, m, s, alpha):
            pp = pool_provider(K, m, source)
            Kp = pp.partition.n_concepts
            s_eff = int(min(max(s, 1), Kp))
            key = (pp.partition.name, s_eff, round(float(alpha), 3))
            if key not in evaluated:
                r, f, info = selector.evaluate(pp, s_eff, float(alpha), output)
                cfg = ExplanationConfiguration(pp.partition.name, Kp, pp.partition.fuzzifier, s_eff, round(float(alpha), 3))
                evaluated[key] = (cfg, r, f, info)
                partitions[pp.partition.name] = pp.partition
            return evaluated[key]

        class Problem(ElementwiseProblem):
            def __init__(self):
                variables = {
                    "source": Choice(options=list(sources)),
                    "K": Integer(bounds=(Kmin, Kmax)),
                    "m": Choice(options=list(fuzzifier_options)),
                    "s": Integer(bounds=(1, Kmax)),
                    "alpha": Real(bounds=(0.0, 1.0)),
                }
                super().__init__(vars=variables, n_obj=3)

            def _evaluate(self, x, out, *args, **kwargs):
                cfg, r, f, info = eval_key(x["source"], int(x["K"]), float(x["m"]), int(x["s"]), float(x["alpha"]))
                fvals = np.asarray(f, dtype=float)
                if r.size == 0:
                    fvals = fvals + 1e6
                out["F"] = fvals

        algorithm = NSGA2(
            pop_size=self.nsga_pop_size,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        )
        minimize(Problem(), algorithm, ("n_gen", self.nsga_generations), seed=self.random_state, verbose=False)
        configs = [v[0] for v in evaluated.values()]
        retained_sets = [v[1] for v in evaluated.values()]
        F = [v[2] for v in evaluated.values()]
        infos = [v[3] for v in evaluated.values()]
        return self._finish(configs, retained_sets, F, infos, partitions)
