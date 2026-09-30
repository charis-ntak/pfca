"""Shared experimental protocol (Section 7.3): run every method on a problem
and compute the metrics of Table 1 from a common pool of refitted models.

The attribution pool (B resamples by M model classes) is computed once per
problem by :class:`pfca.AttributionEngine`; PFCA and its ablations, feature
level SHAP and bootstrapped SHAP are all derived from that pool, so that
differences between methods are due to the explanation method only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pfca.attribution import AttributionEngine, AttributionResult, aggregate_to_concepts, output_function
from pfca.evaluation import metrics as M
from pfca.evaluation.baselines import grouped_shapley, integrated_gradients
from pfca.evaluation.budget import measure
from pfca.explainer import PFCAExplainer
from pfca.fuzzification import centroid_array, fuzzify_array, sign_confidence
from pfca.utils import crisp_membership

PFCA_VARIANTS = {
    "pfca": {},
    "pfca_crisp": {"crisp_concepts": True},
    "pfca_single_model": {"single_model": True},
    "pfca_fixed": {"fixed": True},
    "pfca_apriori": {"partition_source": "apriori"},
}
DEFAULT_METHODS = ("pfca", "pfca_crisp", "pfca_single_model", "pfca_fixed", "pfca_apriori", "shap", "bootstrapped_shap", "grouped_shap")


@dataclass
class MethodOutput:
    name: str
    level: str
    U: np.ndarray
    point: np.ndarray
    importance: np.ndarray
    order: np.ndarray
    pool: np.ndarray | None = None
    quantiles: np.ndarray | None = None
    sign_confidence: np.ndarray | None = None
    wall_seconds: float = 0.0
    peak_mb: float = 0.0
    attribution_fn: Callable[[np.ndarray], np.ndarray] | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        return int(self.point.shape[1])


@dataclass
class EvaluationContext:
    model_fn: Callable[[np.ndarray], np.ndarray]
    X_explain: np.ndarray
    background: np.ndarray
    true_attr: np.ndarray | None = None
    U_true: np.ndarray | None = None
    true_importance: np.ndarray | None = None
    replicate_feature_attr: np.ndarray | None = None
    replicate_mask: np.ndarray | None = None
    perturbation_instances: int = 20
    n_perturbations: int = 3
    perturbation_sigma: float = 0.05
    imputation: str = "conditional"
    random_state: int = 0


def _order_from_point(point: np.ndarray) -> np.ndarray:
    return np.argsort(-np.abs(point), axis=1)


def pfca_output(name: str, engine: AttributionEngine, attr: AttributionResult, feature_names, pfca_kwargs: dict, apriori_groups=None) -> MethodOutput:
    """Run PFCA or one of its ablations on a precomputed pool."""
    variant = PFCA_VARIANTS[name]
    kw = dict(pfca_kwargs)
    use_attr = attr
    if variant.get("single_model"):
        use_attr = attr.subset(model_classes=[0])
    if variant.get("crisp_concepts"):
        kw["crisp_concepts"] = True
    if variant.get("fixed"):
        grid = list(kw.get("n_concepts_grid", (2, 3, 4, 5, 6)))
        Kfix = grid[len(grid) // 2]
        kw["solver"] = "fixed"
        kw["fixed_configuration"] = {"n_concepts": Kfix, "fuzzifier": 2.0, "sparsity": Kfix, "alpha": 0.0}
    if variant.get("partition_source") == "apriori":
        if apriori_groups is None:
            raise ValueError("pfca_apriori requires apriori_groups")
        kw["apriori_groups"] = apriori_groups
        kw["partition_source"] = "apriori"
    elif apriori_groups is not None:
        kw["apriori_groups"] = apriori_groups
        kw.setdefault("partition_source", "both")
    with measure() as b:
        explainer = PFCAExplainer.from_engine(engine, feature_names, **kw)
        expl = explainer.explain_with(use_attr)
    U = expl.partition.U
    point = expl.concept_centroids
    imp = np.abs(point).mean(axis=0)
    E = engine

    def attribution_fn(Xp):
        res = E.explain(Xp)
        if variant.get("single_model"):
            res = res.subset(model_classes=[0])
        pool = aggregate_to_concepts(res.values, U).reshape(-1, Xp.shape[0], U.shape[1])
        return centroid_array(fuzzify_array(pool, axis=0, shape=explainer.fuzzy_shape, support_percentiles=explainer.support_percentiles, core_percentiles=explainer.core_percentiles))

    return MethodOutput(
        name=name,
        level="concept",
        U=U,
        point=point,
        importance=imp,
        order=_order_from_point(point),
        pool=expl.concept_pool,
        quantiles=expl.quantiles,
        sign_confidence=expl.sign_confidence,
        wall_seconds=b.wall_seconds,
        peak_mb=b.peak_python_mb,
        attribution_fn=attribution_fn,
        extra={
            "explanation": expl,
            "n_front": int(expl.selection.pareto.sum()),
            "n_evaluated": len(expl.selection.configurations),
            "n_concepts": expl.n_concepts,
            "n_retained": int(expl.retained.size),
            "alpha": expl.configuration.alpha,
            "fuzzifier": expl.configuration.fuzzifier,
            "partition": expl.configuration.partition,
            "n_type2": int(expl.type2_mask.sum()),
            "fpc": expl.partition.partition_coefficient,
        },
    )


def shap_output(engine: AttributionEngine, attr: AttributionResult, task: str, wall: float = 0.0) -> MethodOutput:
    """Feature level SHAP of the reference model (pool member b = 0, m = 0)."""
    d = attr.n_features
    point = attr.values[0, 0]
    ref = engine.reference_model_
    X_train = engine.X_train_

    def attribution_fn(Xp):
        from pfca.attribution import shapley_values

        bg = X_train[: min(engine.background_size or X_train.shape[0], X_train.shape[0])]
        return shapley_values(ref, Xp, bg, task, explainer=engine.explainer)[0]

    return MethodOutput("shap", "feature", np.eye(d), point, np.abs(point).mean(axis=0), _order_from_point(point), pool=attr.values[:, 0], wall_seconds=wall, attribution_fn=attribution_fn)


def bootstrapped_shap_output(engine: AttributionEngine, attr: AttributionResult, support=(5.0, 95.0), core=(25.0, 75.0), wall: float = 0.0) -> MethodOutput:
    """Feature level SHAP with bootstrap intervals from the reference model class."""
    d = attr.n_features
    pool = attr.values[:, 0]
    Q = fuzzify_array(pool, axis=0, support_percentiles=support, core_percentiles=core)
    point = Q[..., 2]

    def attribution_fn(Xp):
        res = engine.explain(Xp)
        return np.median(res.values[:, 0], axis=0)

    return MethodOutput("bootstrapped_shap", "feature", np.eye(d), point, np.abs(point).mean(axis=0), _order_from_point(point), pool=pool, quantiles=Q, sign_confidence=sign_confidence(pool, axis=0), wall_seconds=wall, attribution_fn=attribution_fn)


def grouped_shap_output(engine: AttributionEngine, attr: AttributionResult, X_explain: np.ndarray, groups, task: str, background_size: int = 50, random_state: int = 0) -> MethodOutput:
    """Exact Shapley values of crisp a priori groups for the reference model, with a bootstrap pool by group aggregation."""
    d = attr.n_features
    rng = np.random.default_rng(random_state)
    X_train = engine.X_train_
    bg = X_train[rng.choice(X_train.shape[0], size=min(background_size, X_train.shape[0]), replace=False)]
    g = output_function(engine.reference_model_, task)
    with measure() as b:
        point, _ = grouped_shapley(g, X_explain, bg, groups)
    labels = -np.ones(d, dtype=int)
    for k, grp in enumerate(groups):
        labels[list(grp)] = k
    K = len(groups)
    if np.any(labels < 0):
        labels[labels < 0] = K
        K += 1
        point = np.hstack([point, np.zeros((point.shape[0], 1))])
    U = crisp_membership(labels, K)
    pool = aggregate_to_concepts(attr.values[:, 0], U)

    def attribution_fn(Xp):
        return grouped_shapley(g, Xp, bg, groups)[0]

    return MethodOutput("grouped_shap", "concept", U, point, np.abs(point).mean(axis=0), _order_from_point(point), pool=pool, wall_seconds=b.wall_seconds, peak_mb=b.peak_python_mb, attribution_fn=attribution_fn)


def integrated_gradients_output(engine: AttributionEngine, X_explain: np.ndarray, task: str, model_index: int | None = None) -> MethodOutput:
    """Integrated gradients for the multilayer perceptron model class when present."""
    names = engine.model_class_names if hasattr(engine, "model_class_names") else [n for n, _ in engine.model_classes_]
    if model_index is None:
        model_index = next((i for i, n in enumerate(names) if "mlp" in n.lower()), 0)
    model = engine.models_[0][model_index]
    g = output_function(model, task)
    baseline = engine.X_train_.mean(axis=0)
    with measure() as b:
        point = integrated_gradients(g, X_explain, baseline)
    d = X_explain.shape[1]
    return MethodOutput("integrated_gradients", "feature", np.eye(d), point, np.abs(point).mean(axis=0), _order_from_point(point), wall_seconds=b.wall_seconds, peak_mb=b.peak_python_mb, attribution_fn=lambda Xp: integrated_gradients(g, Xp, baseline))


# ----------------------------------------------------------------------


def evaluate(out: MethodOutput, ctx: EvaluationContext) -> dict:
    """Compute every applicable metric of Table 1 for one method output."""
    res: dict = {"method": out.name, "level": out.level, "n_items": out.n_items, "wall_seconds": out.wall_seconds, "peak_mb": out.peak_mb}
    rng = np.random.default_rng(ctx.random_state)
    n = ctx.X_explain.shape[0]
    # Correctness -----------------------------------------------------
    if ctx.true_attr is not None and ctx.U_true is not None:
        res["recovery_error"] = M.attribution_recovery_error(out.point, ctx.true_attr, out.U, ctx.U_true)
        cr = M.concept_recovery(out.U, ctx.U_true)
        res["ari"] = cr["ari"]
        res["fpc"] = cr["fpc"]
        res["rank_recovery"] = M.rank_recovery(out.importance, ctx.true_importance, out.U, ctx.U_true)
    # Faithfulness ----------------------------------------------------
    di = M.deletion_insertion_curves(ctx.model_fn, ctx.X_explain, ctx.background, out.U, out.order, mode=ctx.imputation, random_state=ctx.random_state)
    res["deletion_auc"] = di["deletion_auc"]
    res["insertion_auc"] = di["insertion_auc"]
    output = ctx.model_fn(ctx.X_explain)
    res["surrogate_r2"] = M.surrogate_fidelity(out.point, output, random_state=ctx.random_state)
    # Stability -------------------------------------------------------
    if out.pool is not None and out.pool.shape[0] >= 2:
        imp_pool = np.abs(out.pool).mean(axis=1)
        res["rank_stability"] = M.rank_stability(imp_pool, random_state=ctx.random_state)
        if ctx.U_true is not None:
            if out.level == "concept":
                assignment = M.match_concepts_to_factors(out.U, ctx.U_true)
                A = np.zeros((out.U.shape[1], ctx.U_true.shape[1]))
                A[np.arange(assignment.size), assignment] = 1.0
                block_pool = aggregate_to_concepts(out.pool, A)
            else:
                block_pool = aggregate_to_concepts(out.pool, ctx.U_true)
            res["rank_stability_blocks"] = M.rank_stability(np.abs(block_pool).mean(axis=1), random_state=ctx.random_state)
    if out.attribution_fn is not None and ctx.perturbation_instances > 0 and ctx.n_perturbations > 0:
        idx = rng.choice(n, size=min(ctx.perturbation_instances, n), replace=False)
        try:
            res["perturbation_stability"] = M.perturbation_stability(
                out.attribution_fn, ctx.model_fn, ctx.X_explain[idx], sigma=ctx.perturbation_sigma, n_perturbations=ctx.n_perturbations, feature_scale=ctx.background.std(axis=0), random_state=ctx.random_state
            )
        except Exception as exc:  # a failing baseline must not stop the study
            res["perturbation_stability"] = float("nan")
            res["perturbation_error"] = f"{type(exc).__name__}: {exc}"
    # Calibration -----------------------------------------------------
    if out.quantiles is not None:
        if ctx.true_attr is not None and ctx.U_true is not None:
            assignment = M.match_concepts_to_factors(out.U, ctx.U_true)
            Qf = np.zeros((n, ctx.U_true.shape[1], 5))
            counts = np.zeros(ctx.U_true.shape[1])
            for k, f in enumerate(assignment):
                Qf[:, f] += out.quantiles[:, k]
                counts[f] += 1
            keep = counts == 1
            if out.level == "feature":
                keep = np.zeros_like(keep)
            if keep.any():
                cov = M.interval_coverage(Qf[:, keep], ctx.true_attr[:, keep])
                res["support_coverage_truth"] = cov["support_coverage"]
                res["core_coverage_truth"] = cov["core_coverage"]
        if ctx.replicate_feature_attr is not None:
            mask = np.ones(n, dtype=bool) if ctx.replicate_mask is None else ctx.replicate_mask
            rep = aggregate_to_concepts(ctx.replicate_feature_attr[mask], out.U)
            cov = M.interval_coverage(out.quantiles[mask], rep)
            res["support_coverage_replicate"] = cov["support_coverage"]
            res["core_coverage_replicate"] = cov["core_coverage"]
            if out.sign_confidence is not None:
                agree = M.observed_sign_agreement(np.sign(out.quantiles[mask][..., 2]), rep)
                cal = M.sign_confidence_calibration(out.sign_confidence[mask], agree)
                res["sign_confidence_ece"] = cal["ece"]
                res["sign_confidence_table"] = cal["table"]
    for k, v in out.extra.items():
        if k != "explanation":
            res[k] = v
    return res


def duplication_change(before: MethodOutput, after: MethodOutput, source: int, duplicate: int, U_true_after: np.ndarray) -> float:
    """Change of the attribution absorbing a duplicated feature (Table 1, duplication invariance).

    For a concept level method the concept containing the source feature is
    compared before and after the duplicate is added; for a feature level
    method the attribution of the source feature itself is compared, which
    exposes credit splitting.
    """
    if before.level == "feature":
        return M.duplication_invariance(before.point[:, source], after.point[:, source])
    kb = int(np.argmax(before.U[source]))
    ka = int(np.argmax(after.U[source]))
    return M.duplication_invariance(before.point[:, kb], after.point[:, ka])
