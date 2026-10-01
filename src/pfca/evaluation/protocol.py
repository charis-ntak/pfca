"""Shared experimental protocol (Section 7.3): run every method on a problem
and compute the metrics of Table 1 from a common pool of refitted models.

The attribution pool (B resamples by M model classes) is computed once per
problem by :class:`pfca.AttributionEngine`; PFCA and its ablations, feature
level SHAP and bootstrapped SHAP are all derived from that pool, so that
differences between methods are due to the explanation method only. Grouped
SHAP and the perturbation functions of the feature level baselines use the
background sample of pool member (0, 0), see :func:`pool_background`, so that
every method refers to the same expected value (Section 7.3, step 3). Wall
times are recorded with memory tracing off (see :mod:`pfca.evaluation.budget`);
``peak_mb`` is filled only when ``trace_memory=True`` is requested, which is
meant for a dedicated cost run so that every method is timed under the same
conditions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from pfca.attribution import AttributionEngine, AttributionResult, aggregate_to_concepts, output_function, shapley_values
from pfca.evaluation import metrics as M
from pfca.evaluation.baselines import grouped_shapley, integrated_gradients, lime_attribution
from pfca.evaluation.budget import measure
from pfca.explainer import PFCAExplainer
from pfca.fuzzification import centroid_array, fuzzify_array, sign_confidence
from pfca.utils import as_index_groups, crisp_membership

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
    """Output of one explanation method in the common format of the protocol.

    ``retained`` holds the indices of the items (concepts or features) on
    which the surrogate fidelity of Table 1 is computed: the retained concepts
    of the knee configuration for PFCA, the ``n_retained`` most important
    items for a baseline when the caller asks for a matched sparsity, and
    None when every item is retained.
    """

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
    retained: np.ndarray | None = None

    @property
    def n_items(self) -> int:
        return int(self.point.shape[1])

    @property
    def retained_items(self) -> np.ndarray:
        """Indices of the retained items; every item when ``retained`` is None."""
        if self.retained is None:
            return np.arange(self.n_items)
        return np.asarray(self.retained, dtype=int)


@dataclass
class EvaluationContext:
    """Data shared by every method in :func:`evaluate`.

    Array inputs are coerced to float arrays, so pandas objects are accepted.
    ``true_importance`` defaults to the mean absolute true attribution over
    the explained instances when ``true_attr`` is given, as
    :meth:`pfca.evaluation.synthetic.SyntheticProblem.true_global_importance`
    defines it. ``surrogate_metric`` selects the surrogate fidelity metric of
    Table 1: 'r2' (reported as surrogate_r2) or, for classification outputs,
    'log_loss' (reported as surrogate_log_loss).
    """

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
    surrogate_metric: str = "r2"

    def __post_init__(self):
        self.X_explain = np.asarray(self.X_explain, dtype=float)
        self.background = np.asarray(self.background, dtype=float)
        for name in ("true_attr", "U_true", "true_importance", "replicate_feature_attr"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=float))
        if self.replicate_mask is not None:
            self.replicate_mask = np.asarray(self.replicate_mask, dtype=bool)
        if self.true_attr is not None and self.true_importance is None:
            self.true_importance = np.abs(self.true_attr).mean(axis=0)
        if self.surrogate_metric not in ("r2", "log_loss"):
            raise ValueError("surrogate_metric must be 'r2' or 'log_loss'")


def _order_from_point(point: np.ndarray) -> np.ndarray:
    return np.argsort(-np.abs(point), axis=1)


def _top_items(importance: np.ndarray, n_retained: int | None) -> np.ndarray | None:
    """Indices of the ``n_retained`` most important items, or None for every item."""
    if n_retained is None:
        return None
    importance = np.asarray(importance, dtype=float)
    n = min(max(int(n_retained), 0), importance.size)
    return np.argsort(-importance, kind="stable")[:n]


def _final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def pool_background(engine: AttributionEngine, b: int = 0, m: int = 0) -> np.ndarray:
    """Background sample of pool member (b, m), reconstructed as the engine drew it.

    The engine draws the background of member (b, m) from the rows of
    resample b with a generator seeded by ``seeds_[b, m] + 1``; member (0, 0)
    is the reference model on the original training data. Sharing this sample
    with the baselines is what Section 7.3 requires (the same background
    sample and the same random seeds for every method).
    """
    rows = engine.X_train_[engine.resample_indices_[b]]
    size = engine.background_size
    if size is not None and size < rows.shape[0]:
        rng = np.random.default_rng(int(engine.seeds_[b, m]) + 1)
        return rows[rng.choice(rows.shape[0], size=int(size), replace=False)]
    return rows


def pfca_output(name: str, engine: AttributionEngine, attr: AttributionResult, feature_names, pfca_kwargs: dict, apriori_groups=None, trace_memory: bool = False) -> MethodOutput:
    """Run PFCA or one of its ablations on a precomputed pool.

    ``apriori_groups`` serves the pfca_apriori ablation, which evaluates the a
    priori partition instead of the data driven ones (Section 7.1). The data
    driven variants (pfca, pfca_crisp, pfca_single_model, pfca_fixed) keep
    the ``partition_source`` given in ``pfca_kwargs``, 'data' by default, and
    receive ``apriori_groups`` only when that source is 'both' or 'apriori'.
    In Phase A the groups are the true blocks of the simulation, which are not
    domain knowledge, so the data driven variants must not evaluate them as a
    candidate partition; in Phases B and C, where a grouping is domain
    knowledge (Section 3.2), ``partition_source='both'`` may be set
    explicitly in ``pfca_kwargs``. ``peak_mb`` is filled only with
    ``trace_memory``.
    """
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
    else:
        kw.setdefault("partition_source", "data")
        if kw["partition_source"] in ("apriori", "both") and apriori_groups is not None:
            kw["apriori_groups"] = apriori_groups
    with measure(trace_memory=trace_memory) as b:
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
        retained=np.asarray(expl.retained, dtype=int),
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
            **objective_conflict(expl.selection),
        },
    )


def objective_conflict(sel) -> dict:
    """Diagnostics of the Pareto front for the collapse risk of Section 10 of the guide.

    The guide asks to verify that the three objectives of Equation 4 conflict
    and to report the case in which the front collapses to one solution.
    Returns the number of feasible evaluated configurations, the number of
    distinct Pareto optimal explanations (one representative per partition and
    retained set, see ``SelectionResult.representative``), whether the front
    collapsed to a single distinct explanation, and the Spearman correlation
    between every pair of objectives over the feasible configurations; a
    negative correlation means that the two objectives conflict, and the
    correlation is missing when an objective is constant.
    """
    from scipy import stats

    F = np.asarray(sel.objectives, dtype=float)
    feasible = np.all(np.isfinite(F), axis=1)
    pareto = np.asarray(sel.pareto, dtype=bool)
    rep = getattr(sel, "representative", None)
    distinct = int((pareto & np.asarray(rep, dtype=bool)).sum()) if rep is not None else int(pareto.sum())
    names = ("fidelity", "complexity", "instability")
    out = {"n_feasible": int(feasible.sum()), "n_front_distinct": distinct, "front_collapsed": bool(distinct <= 1)}
    Ff = F[feasible]
    for i in range(3):
        for j in range(i + 1, 3):
            key = f"objective_corr_{names[i]}_{names[j]}"
            if Ff.shape[0] < 3 or np.ptp(Ff[:, i]) < 1e-12 or np.ptp(Ff[:, j]) < 1e-12:
                out[key] = float("nan")
            else:
                with np.errstate(invalid="ignore"):
                    out[key] = float(stats.spearmanr(Ff[:, i], Ff[:, j]).correlation)
    return out


def _pool_target_class(engine: AttributionEngine):
    target = getattr(engine, "target_class_", None)
    return 1 if target is None else target


def shap_output(engine: AttributionEngine, attr: AttributionResult, task: str, wall: float = 0.0, n_retained: int | None = None) -> MethodOutput:
    """Feature level SHAP of the reference model (pool member b = 0, m = 0).

    The perturbation function recomputes the Shapley values of the reference
    model with the background sample and the seed of pool member (0, 0), so
    that perturbed and unperturbed attributions refer to the same expected
    value. ``n_retained`` restricts the surrogate fidelity to the most
    important features (for example matched to the knee of PFCA); by default
    every feature is retained.
    """
    d = attr.n_features
    point = attr.values[0, 0]
    ref = engine.reference_model_
    bg = pool_background(engine)
    target_class = _pool_target_class(engine)
    seed = int(engine.seeds_[0, 0]) + 1

    def attribution_fn(Xp):
        return shapley_values(ref, Xp, bg, task, explainer=engine.explainer, target_class=target_class, seed=seed)[0]

    importance = np.abs(point).mean(axis=0)
    return MethodOutput("shap", "feature", np.eye(d), point, importance, _order_from_point(point), pool=attr.values[:, 0], wall_seconds=wall, attribution_fn=attribution_fn, retained=_top_items(importance, n_retained))


def bootstrapped_shap_output(engine: AttributionEngine, attr: AttributionResult, support=(5.0, 95.0), core=(25.0, 75.0), wall: float = 0.0, n_retained: int | None = None) -> MethodOutput:
    """Feature level SHAP with bootstrap intervals from the reference model class.

    ``n_retained`` restricts the surrogate fidelity to the most important
    features; by default every feature is retained.
    """
    d = attr.n_features
    pool = attr.values[:, 0]
    Q = fuzzify_array(pool, axis=0, support_percentiles=support, core_percentiles=core)
    point = Q[..., 2]

    def attribution_fn(Xp):
        res = engine.explain(Xp)
        return np.median(res.values[:, 0], axis=0)

    importance = np.abs(point).mean(axis=0)
    return MethodOutput("bootstrapped_shap", "feature", np.eye(d), point, importance, _order_from_point(point), pool=pool, quantiles=Q, sign_confidence=sign_confidence(pool, axis=0), wall_seconds=wall, attribution_fn=attribution_fn, retained=_top_items(importance, n_retained))


def grouped_shap_output(engine: AttributionEngine, attr: AttributionResult, X_explain: np.ndarray, groups, task: str, background_size: int | None = None, random_state: int = 0, n_retained: int | None = None, trace_memory: bool = False) -> MethodOutput:
    """Exact Shapley values of crisp a priori groups for the reference model, with a bootstrap pool by group aggregation.

    The background is the sample of pool member (0, 0) (see
    :func:`pool_background`), so that the grouped values and the feature
    level values of the pool refer to the same expected value, which is
    recorded in ``extra['expected']``; an integer ``background_size`` smaller
    than that sample subsamples it with ``random_state``, for cost only.
    Features outside every group form one additional group, as
    :func:`pfca.concepts.apriori_membership` does, so that the grand coalition
    sets every feature to the explained instance and the group values sum to
    the output minus the expected value. ``n_retained`` restricts the
    surrogate fidelity to the most important groups; ``peak_mb`` is filled
    only with ``trace_memory``.
    """
    d = attr.n_features
    idx_groups = [g.tolist() for g in as_index_groups(groups, d)]
    labels = -np.ones(d, dtype=int)
    for k, grp in enumerate(idx_groups):
        labels[grp] = k
    rest = [int(j) for j in np.flatnonzero(labels < 0)]
    if rest:
        labels[rest] = len(idx_groups)
        idx_groups = idx_groups + [rest]
    K = len(idx_groups)
    bg = pool_background(engine)
    if background_size is not None and background_size < bg.shape[0]:
        rng = np.random.default_rng(random_state)
        bg = bg[rng.choice(bg.shape[0], size=int(background_size), replace=False)]
    g = output_function(engine.reference_model_, task, _pool_target_class(engine))
    with measure(trace_memory=trace_memory) as b:
        point, expected = grouped_shapley(g, X_explain, bg, idx_groups)
    U = crisp_membership(labels, K)
    pool = aggregate_to_concepts(attr.values[:, 0], U)

    def attribution_fn(Xp):
        return grouped_shapley(g, Xp, bg, idx_groups)[0]

    importance = np.abs(point).mean(axis=0)
    return MethodOutput(
        "grouped_shap",
        "concept",
        U,
        point,
        importance,
        _order_from_point(point),
        pool=pool,
        wall_seconds=b.wall_seconds,
        peak_mb=b.peak_python_mb,
        attribution_fn=attribution_fn,
        extra={"expected": float(expected)},
        retained=_top_items(importance, n_retained),
    )


def integrated_gradients_output(engine: AttributionEngine, X_explain: np.ndarray, task: str, model_index: int | None = None, trace_memory: bool = False) -> MethodOutput:
    """Integrated gradients for the multilayer perceptron model class.

    Finite difference gradients are meaningful for smooth models only; a tree
    ensemble is piecewise constant, so its gradients are zero almost
    everywhere and unbounded at the split points. When ``model_index`` is
    None the first model class whose name contains 'mlp' or whose final
    estimator is a multilayer perceptron is used, and a ValueError is raised
    when there is none, so that a configuration listing integrated gradients
    without a smooth model class does not produce meaningless rows. An
    explicit ``model_index`` is used as given. The model class used is
    recorded in ``extra['model_class']``; ``peak_mb`` is filled only with
    ``trace_memory``.
    """
    names = list(getattr(engine, "model_class_names", None) or [n for n, _ in engine.model_classes_])
    if model_index is None:
        smooth = [i for i, n in enumerate(names) if "mlp" in n.lower() or "MLP" in type(_final_estimator(engine.models_[0][i])).__name__]
        if not smooth:
            raise ValueError("integrated gradients require a smooth model class (multilayer perceptron), and the pool has none; pass model_index explicitly to use another model class")
        model_index = smooth[0]
    model = engine.models_[0][model_index]
    g = output_function(model, task, _pool_target_class(engine))
    baseline = engine.X_train_.mean(axis=0)
    with measure(trace_memory=trace_memory) as b:
        point = integrated_gradients(g, X_explain, baseline)
    d = X_explain.shape[1]
    return MethodOutput(
        "integrated_gradients",
        "feature",
        np.eye(d),
        point,
        np.abs(point).mean(axis=0),
        _order_from_point(point),
        wall_seconds=b.wall_seconds,
        peak_mb=b.peak_python_mb,
        attribution_fn=lambda Xp: integrated_gradients(g, Xp, baseline),
        extra={"model_class": names[model_index]},
    )


def lime_available() -> bool:
    """True when the optional lime package is installed (Section 7.1 lists LIME where applicable)."""
    import importlib.util

    return importlib.util.find_spec("lime") is not None


def lime_output(engine: AttributionEngine, X_explain: np.ndarray, task: str, feature_names=None, num_samples: int = 2000, random_state: int = 0, n_retained: int | None = None, trace_memory: bool = False) -> MethodOutput:
    """LIME feature weights of the reference model (optional lime package, Section 7.1).

    The tabular explainer is fitted on the training data of the pool with
    ``num_samples`` perturbations per instance and the given seed; the weights
    of the local linear model are the attribution. ``n_retained`` restricts the
    surrogate fidelity to the most important features (matched sparsity);
    ``peak_mb`` is filled only with ``trace_memory``. Raises ImportError when
    lime is not installed; use :func:`lime_available` to skip the method.
    """
    model = engine.reference_model_
    X_train = engine.X_train_
    names = list(feature_names) if feature_names is not None else None
    with measure(trace_memory=trace_memory) as b:
        point = lime_attribution(model, X_train, X_explain, task, num_samples=num_samples, random_state=random_state, feature_names=names)
    d = X_explain.shape[1]
    importance = np.abs(point).mean(axis=0)
    return MethodOutput(
        "lime",
        "feature",
        np.eye(d),
        point,
        importance,
        _order_from_point(point),
        wall_seconds=b.wall_seconds,
        peak_mb=b.peak_python_mb,
        attribution_fn=lambda Xp: lime_attribution(model, X_train, Xp, task, num_samples=num_samples, random_state=random_state, feature_names=names),
        extra={"num_samples": int(num_samples)},
        retained=_top_items(importance, n_retained),
    )


# ----------------------------------------------------------------------


def evaluate(out: MethodOutput, ctx: EvaluationContext) -> dict:
    """Compute every applicable metric of Table 1 for one method output.

    Surrogate fidelity is computed on the retained items of the output
    (``MethodOutput.retained``), as Table 1 defines it, and reported as
    surrogate_r2 (or surrogate_log_loss); the value on every item is reported
    beside it with the suffix _all, for reference. For an efficient
    attribution (feature level SHAP, exact grouped Shapley values) the columns
    sum to the output minus the expected value, so the surrogate on every
    item has R squared one by construction, whatever the ranking quality.
    The entry ``per_instance`` of the result is a data frame with the raw
    values per explained instance (recovery error, deletion and insertion
    areas, model output, coverage indicators and sign agreement where they
    apply), which the experiment scripts store as required by Section 7.3.
    """
    res: dict = {"method": out.name, "level": out.level, "n_items": out.n_items, "wall_seconds": out.wall_seconds, "peak_mb": out.peak_mb}
    rng = np.random.default_rng(ctx.random_state)
    n = ctx.X_explain.shape[0]
    per_instance: dict[str, np.ndarray] = {}
    # Correctness -----------------------------------------------------
    if ctx.true_attr is not None and ctx.U_true is not None:
        res["recovery_error"] = M.attribution_recovery_error(out.point, ctx.true_attr, out.U, ctx.U_true)
        est_f = M.aggregate_by_assignment(out.point, M.match_concepts_to_factors(out.U, ctx.U_true), ctx.U_true.shape[1])
        truth_scale = float(np.mean(np.abs(ctx.true_attr)))
        per_instance["recovery_error"] = np.mean(np.abs(est_f - ctx.true_attr), axis=1) / (truth_scale if truth_scale > 1e-12 else 1.0)
        cr = M.concept_recovery(out.U, ctx.U_true)
        res["ari"] = cr["ari"]
        res["fpc"] = cr["fpc"]
        res["rank_recovery"] = M.rank_recovery(out.point, ctx.true_importance, out.U, ctx.U_true)
    # Faithfulness ----------------------------------------------------
    di = M.deletion_insertion_curves(ctx.model_fn, ctx.X_explain, ctx.background, out.U, out.order, mode=ctx.imputation, random_state=ctx.random_state)
    res["deletion_auc"] = di["deletion_auc"]
    res["insertion_auc"] = di["insertion_auc"]
    per_instance["deletion_auc"] = di["deletion_auc_per_instance"]
    per_instance["insertion_auc"] = di["insertion_auc_per_instance"]
    output = ctx.model_fn(ctx.X_explain)
    per_instance["model_output"] = np.asarray(output, dtype=float)
    retained = out.retained_items
    key = "surrogate_r2" if ctx.surrogate_metric == "r2" else "surrogate_log_loss"
    res[key] = M.surrogate_fidelity(out.point[:, retained], output, metric=ctx.surrogate_metric, random_state=ctx.random_state)
    res[key + "_all"] = res[key] if retained.size == out.n_items else M.surrogate_fidelity(out.point, output, metric=ctx.surrogate_metric, random_state=ctx.random_state)
    res["n_retained"] = int(retained.size)
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
                T = ctx.true_attr[:, keep]
                per_instance["support_covered_truth"] = ((T >= Qf[:, keep][..., 0]) & (T <= Qf[:, keep][..., 4])).mean(axis=1)
                per_instance["core_covered_truth"] = ((T >= Qf[:, keep][..., 1]) & (T <= Qf[:, keep][..., 3])).mean(axis=1)
        if ctx.replicate_feature_attr is not None:
            mask = np.ones(n, dtype=bool) if ctx.replicate_mask is None else ctx.replicate_mask
            rep = aggregate_to_concepts(ctx.replicate_feature_attr[mask], out.U)
            cov = M.interval_coverage(out.quantiles[mask], rep)
            res["support_coverage_replicate"] = cov["support_coverage"]
            res["core_coverage_replicate"] = cov["core_coverage"]
            Qm = out.quantiles[mask]
            sup_cov = np.full(n, np.nan)
            core_cov = np.full(n, np.nan)
            sup_cov[mask] = ((rep >= Qm[..., 0]) & (rep <= Qm[..., 4])).mean(axis=1)
            core_cov[mask] = ((rep >= Qm[..., 1]) & (rep <= Qm[..., 3])).mean(axis=1)
            per_instance["support_covered_replicate"] = sup_cov
            per_instance["core_covered_replicate"] = core_cov
            if out.sign_confidence is not None:
                agree = M.observed_sign_agreement(np.sign(out.quantiles[mask][..., 2]), rep)
                cal = M.sign_confidence_calibration(out.sign_confidence[mask], agree)
                res["sign_confidence_ece"] = cal["ece"]
                res["sign_confidence_table"] = cal["table"]
                sa = np.full(n, np.nan)
                sa[mask] = agree.mean(axis=1)
                per_instance["sign_agreement_replicate"] = sa
    for k, v in out.extra.items():
        if k != "explanation":
            res[k] = v
    res["per_instance"] = pd.DataFrame(per_instance, index=pd.RangeIndex(n, name="instance"))
    return res


def matched_sparsity(out: MethodOutput) -> dict[str, int]:
    """Sparsity of a PFCA explanation used to match the baselines' surrogate fidelity.

    Concept level baselines retain as many items as the explanation retains
    concepts; feature level baselines retain the effective number of features
    covered by the retained concepts (the total membership mass of those
    concepts, rounded), so that the surrogate fidelity of Table 1 is compared
    at a comparable sparsity. The value on every item is reported beside it
    by :func:`evaluate` (suffix _all).
    """
    retained = out.retained_items
    n_concepts = int(retained.size)
    n_features = int(round(float(np.asarray(out.U)[:, retained].sum()))) if retained.size else 0
    return {"concept": max(n_concepts, 1), "feature": max(n_features, 1)}


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
