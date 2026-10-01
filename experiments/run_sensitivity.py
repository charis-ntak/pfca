"""Sensitivity analyses of Section 7.5 of the study design guide.

Every analysis varies one design choice of PFCA at a time around a default
setting and records, for every cell, seed, factor and setting, the metrics of
Table 1 that apply to Phase A together with descriptors of the chosen
explanation. The default setting uses B = 100 resamples (or the largest
number of resamples listed in the configuration), M = 3 model classes (or all
configured classes), trapezoidal fuzzy numbers with the 5th and 95th
percentiles as support and the 25th and 75th percentiles as core, the
correlation distance for fuzzy c means, the exhaustive grid solver and the
utopia knee point. The default setting is evaluated once per cell and seed and
its row is repeated under every factor, so that each factor reads as a
complete one way comparison.

For every synthetic cell and seed the training sample, the explained
instances and the replicate sample are drawn as in run_phase_a.py, the
reference gradient boosting model is tuned on a tuning sample, and one
attribution pool is fitted at the largest number of resamples with every
model class. Smaller settings of B and M are obtained by subsetting that pool
(resamples 0 to B - 1, which keeps the reference fit first, and model classes
0 to M - 1), so that no pool is refitted per setting. When the configuration
asks for more model classes than are listed, a ridge model is appended to the
list before the pool is fitted.

Factors
-------
resamples       number of resamples B. The mean absolute difference between
                the quantile arrays at B and at the largest B, computed with
                the partition and configuration chosen at the largest B held
                fixed through the fixed solver, measures the stabilization of
                the fuzzy summaries and identifies the smallest adequate B.
model_classes   number of model classes M.
shape           trapezoidal against triangular fuzzy numbers.
percentiles     percentile levels defining support and core.
distance        correlation based against loading based concept formation.
solver          exhaustive grid against NSGA II. The fraction of the grid
                front recovered by NSGA II and the hypervolume of both fronts
                are stored on the NSGA II row. Two configurations are the same
                front member when they share the partition and the retained
                concept set, which implies the same objective vector; an exact
                match on the alpha level is also counted but is rare because
                NSGA II searches alpha continuously. The hypervolume is exact
                and is computed by a dimension sweep along the third objective
                (see hypervolume_3d) after min max normalization of the union
                of the two fronts, with reference point (1.1, 1.1, 1.1).
knee            utopia against hyperplane knee point and against two
                alternative representative selections taken from the same
                front: the sparsest front member whose fidelity loss is within
                10 percent of the best fidelity loss on the front, and the
                front member whose complexity is closest to the median
                complexity of the front.

Perturbation stability is not part of the sensitivity metrics and is switched
off by default (perturbation_instances 0) because it refits nothing but
recomputes the whole pool of Shapley values per perturbation.

Usage
-----
python experiments/run_sensitivity.py --config experiments/configs/sensitivity.yaml --results results
python experiments/run_sensitivity.py --config experiments/configs/sensitivity_smoke.yaml --results results --name sensitivity_smoke
python experiments/run_sensitivity.py --config experiments/configs/sensitivity.yaml --factors resamples,solver --seeds 3 --n-jobs 2

Outputs, in <results>/<name>/
-----------------------------
config.yaml, environment.json   frozen configuration, package versions and command line
metrics.csv                     one row per cell, seed, factor and setting (resumable)
runs.csv                        one row per cell and seed: seeds, tuned reference parameters and pool cost
summary.csv                     mean and standard deviation over seeds per cell, factor and setting
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ResultWriter, flatten_row, load_config, model_classes_from_names, prepare_run, tune_gradient_boosting  # noqa: E402

from pfca.attribution import AttributionEngine, AttributionResult, aggregate_to_concepts, output_function, shapley_values  # noqa: E402
from pfca.evaluation import protocol  # noqa: E402
from pfca.evaluation.budget import measure  # noqa: E402
from pfca.evaluation.synthetic import make_synthetic  # noqa: E402
from pfca.explainer import PFCAExplainer, PFCAExplanation  # noqa: E402
from pfca.fuzzification import centroid_array, fuzzify_array  # noqa: E402
from pfca.selection import SelectionResult  # noqa: E402

FACTORS = ("resamples", "model_classes", "shape", "percentiles", "distance", "solver", "knee")
KNEE_METHODS = ("utopia", "hyperplane")
ALTERNATIVE_SELECTIONS = ("sparsest_within_10pct", "median_complexity")
KEY_COLUMNS = ["cell", "seed", "factor", "setting"]
HV_REFERENCE = (1.1, 1.1, 1.1)
TASK = "regression"


class AlignedResultWriter(ResultWriter):
    """ResultWriter whose appended rows are aligned to the columns already in the file.

    Rows of different factors carry different columns (for example the front
    comparison exists only on the NSGA II row and the stabilization measure
    only on the rows of the resamples factor). Rows are reindexed to the
    existing columns, and when new columns appear the file is rewritten with
    the union of columns, which is cheap at the size of a metrics table.
    """

    @staticmethod
    def _serialize(row: dict) -> dict:
        out = {}
        for k, v in row.items():
            if isinstance(v, pd.DataFrame):
                v = v.to_dict(orient="list")
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, (list, tuple, dict)):
                v = json.dumps(v)
            out[k] = v
        return out

    def append(self, rows: list[dict]) -> None:
        if not rows:
            return
        new = pd.DataFrame([self._serialize(r) for r in rows])
        if self.path.exists():
            prev = pd.read_csv(self.path)
            if set(new.columns) <= set(prev.columns):
                new.reindex(columns=prev.columns).to_csv(self.path, mode="a", header=False, index=False)
            else:
                cols = list(prev.columns) + [c for c in new.columns if c not in prev.columns]
                pd.concat([prev.reindex(columns=cols), new.reindex(columns=cols)], ignore_index=True).to_csv(self.path, index=False)
        else:
            new.to_csv(self.path, index=False)
        for r in rows:
            if all(c in r for c in self.key_columns):
                self._done.add(tuple(str(r[c]) for c in self.key_columns))


# ----------------------------------------------------------------------
# Settings and plan
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """One combination of the design choices varied in Section 7.5."""

    n_resamples: int
    n_model_classes: int
    fuzzy_shape: str
    support: tuple[float, float]
    core: tuple[float, float]
    concept_distance: str
    solver: str
    knee_method: str

    def replace(self, **changes) -> "Setting":
        return dataclasses.replace(self, **changes)

    def descriptors(self) -> dict:
        return {
            "n_resamples": int(self.n_resamples),
            "n_model_classes": int(self.n_model_classes),
            "fuzzy_shape": self.fuzzy_shape,
            "support_low": float(self.support[0]),
            "support_high": float(self.support[1]),
            "core_low": float(self.core[0]),
            "core_high": float(self.core[1]),
            "concept_distance": self.concept_distance,
            "solver": self.solver,
            "knee_method": self.knee_method,
        }


def cell_label(cell: dict) -> str:
    return f"{cell['family']}_n{int(cell['n_samples'])}_d{int(cell['n_features'])}_rho{float(cell['rho']):g}"


def percentile_label(support, core, shape: str = "trapezoidal") -> str:
    s = f"s{support[0]:g}-{support[1]:g}"
    return s if shape == "triangular" else f"{s}_c{core[0]:g}-{core[1]:g}"


def resolve_model_class_names(cfg: dict) -> list[str]:
    """Configured model classes, with a ridge model appended when more classes are asked for than listed."""
    names = [str(n) for n in cfg["model_classes"]]
    counts = [int(m) for m in cfg.get("model_class_counts", [len(names)])]
    if counts and max(counts) > len(names) and "ridge" not in names:
        names.append("ridge")
    return names


def default_setting(cfg: dict, n_classes_available: int) -> Setting:
    """Default setting: B 100 or the largest available, M 3 or all, trapezoidal, 5/95 and 25/75, correlation, grid, utopia."""
    d = dict(cfg.get("defaults") or {})
    resamples = sorted(int(b) for b in cfg["resamples"])
    B = int(d.get("n_resamples", 100))
    B = B if B in resamples else max(resamples)
    M = min(int(d.get("n_model_classes", 3)), int(n_classes_available))
    support = tuple(float(v) for v in d.get("support_percentiles", (5.0, 95.0)))
    core = tuple(float(v) for v in d.get("core_percentiles", (25.0, 75.0)))
    knee = str(d.get("knee_method", "utopia"))
    if knee not in KNEE_METHODS:
        raise ValueError(f"defaults.knee_method must be one of {KNEE_METHODS}, got '{knee}'")
    return Setting(B, M, str(d.get("fuzzy_shape", "trapezoidal")), support, core, str(d.get("concept_distance", "correlation")), str(d.get("solver", "grid")), knee)


def build_plan(cfg: dict, default: Setting, n_classes_available: int, factors: list[str]) -> list[tuple[str, str, Setting]]:
    """List of (factor, setting name, setting), one entry per row of the metrics table per cell and seed."""
    plan: list[tuple[str, str, Setting]] = []
    if "resamples" in factors:
        for B in sorted({int(b) for b in cfg["resamples"]}):
            plan.append(("resamples", f"B{B}", default.replace(n_resamples=B)))
    if "model_classes" in factors:
        for M in sorted({int(m) for m in cfg["model_class_counts"]}):
            if M < 1 or M > n_classes_available:
                print(f"[sensitivity] skipping M={M}: only {n_classes_available} model classes are available")
                continue
            plan.append(("model_classes", f"M{M}", default.replace(n_model_classes=M)))
    if "shape" in factors:
        for shape in cfg["shapes"]:
            plan.append(("shape", str(shape), default.replace(fuzzy_shape=str(shape))))
    if "percentiles" in factors:
        for spec in cfg["percentiles"]:
            support = tuple(float(v) for v in spec["support"])
            core = tuple(float(v) for v in spec.get("core", default.core))
            plan.append(("percentiles", percentile_label(support, core, default.fuzzy_shape), default.replace(support=support, core=core)))
    if "distance" in factors:
        for dist in cfg["distances"]:
            plan.append(("distance", str(dist), default.replace(concept_distance=str(dist))))
    if "solver" in factors:
        for solver in cfg["solvers"]:
            plan.append(("solver", str(solver), default.replace(solver=str(solver))))
    if "knee" in factors:
        for knee in cfg["knee_methods"]:
            if knee not in KNEE_METHODS + ALTERNATIVE_SELECTIONS:
                raise ValueError(f"Unknown knee method or representative selection '{knee}'")
            plan.append(("knee", str(knee), default.replace(knee_method=str(knee))))
    return plan


# ----------------------------------------------------------------------
# Front comparison (solver factor)
# ----------------------------------------------------------------------


def _dominated_area_2d(S: np.ndarray, reference: np.ndarray) -> float:
    """Area dominated by points S (minimization) below the reference point in two objectives."""
    S = S[np.argsort(S[:, 0], kind="stable")]
    area, y_prev = 0.0, float(reference[1])
    for x, y in S:
        if y < y_prev:
            area += (float(reference[0]) - float(x)) * (y_prev - float(y))
            y_prev = float(y)
    return area


def hypervolume_3d(points: np.ndarray, reference=HV_REFERENCE) -> float:
    """Exact hypervolume dominated by a set of points in three minimized objectives.

    The objective space is swept along the third objective: the points are
    sorted by that objective and, for every consecutive pair of levels, the
    area dominated in the first two objectives by the points at or below the
    lower level is multiplied by the thickness of the slice. The routine is
    exact for three objectives and costs O(n^2 log n). Points that do not
    strictly dominate the reference point contribute nothing.
    """
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    ref = np.asarray(reference, dtype=float)
    P = P[np.all(P < ref, axis=1)]
    if P.shape[0] == 0:
        return 0.0
    levels = np.unique(P[:, 2])
    upper = np.append(levels[1:], ref[2])
    hv = 0.0
    for z, z_next in zip(levels, upper):
        S = P[P[:, 2] <= z, :2]
        hv += _dominated_area_2d(S, ref[:2]) * (float(z_next) - float(z))
    return float(hv)


def _front_members(sel: SelectionResult) -> tuple[set, set, list]:
    """Front members keyed by (partition, retained set) and by (partition, sparsity, alpha), plus objective rows."""
    by_set, exact, rows = set(), set(), []
    for cfg, r, f, p in zip(sel.configurations, sel.retained_sets, sel.objectives, sel.pareto):
        if not p:
            continue
        by_set.add((cfg.partition, frozenset(int(k) for k in r)))
        exact.add((cfg.partition, int(cfg.sparsity), round(float(cfg.alpha), 3)))
        rows.append(np.asarray(f, dtype=float))
    return by_set, exact, rows


def compare_fronts(grid: SelectionResult, nsga: SelectionResult) -> dict:
    """Recovery of the grid front by NSGA II and hypervolume of both fronts under a common normalization."""
    g_set, g_exact, g_rows = _front_members(grid)
    n_set, n_exact, n_rows = _front_members(nsga)
    Fg, Fn = np.vstack(g_rows), np.vstack(n_rows)
    both = np.vstack([Fg, Fn])
    lo, hi = both.min(axis=0), both.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    hv_grid = hypervolume_3d((Fg - lo) / span)
    hv_nsga = hypervolume_3d((Fn - lo) / span)
    knee_g = grid.configurations[grid.knee]
    knee_n = nsga.configurations[nsga.knee]
    same_knee = (knee_g.partition, frozenset(int(k) for k in grid.retained_sets[grid.knee])) == (knee_n.partition, frozenset(int(k) for k in nsga.retained_sets[nsga.knee]))
    return {
        "front_recovery": len(g_set & n_set) / len(g_set) if g_set else float("nan"),
        "front_recovery_exact": len(g_exact & n_exact) / len(g_exact) if g_exact else float("nan"),
        "n_front_grid": len(g_set),
        "n_front_nsga2": len(n_set),
        "n_evaluated_grid": len(grid.configurations),
        "n_evaluated_nsga2": len(nsga.configurations),
        "hypervolume_grid": hv_grid,
        "hypervolume_nsga2": hv_nsga,
        "hypervolume_ratio": hv_nsga / hv_grid if hv_grid > 1e-12 else float("nan"),
        "same_knee_as_grid": bool(same_knee),
    }


# ----------------------------------------------------------------------
# Alternative representative selections (knee factor)
# ----------------------------------------------------------------------


def alternative_index(sel: SelectionResult, method: str, tolerance: float = 0.10) -> int:
    """Index in the evaluated table of an alternative representative front member.

    'sparsest_within_10pct' returns the front member with the fewest retained
    concepts among those whose fidelity loss is within ``tolerance`` (relative)
    of the best fidelity loss on the front; ties are broken by complexity and
    then by fidelity loss. 'median_complexity' returns the front member whose
    complexity is closest to the median complexity of the front; ties are
    broken by fidelity loss and then by instability.
    """
    front = sel.front
    if front.empty:
        raise ValueError("Empty Pareto front.")
    if method == "sparsest_within_10pct":
        best = float(front["fidelity_loss"].min())
        cand = front[front["fidelity_loss"] <= best * (1.0 + tolerance) + 1e-12]
        cand = cand.sort_values(["n_retained", "complexity", "fidelity_loss"], kind="stable")
        return int(cand.index[0])
    if method == "median_complexity":
        med = float(front["complexity"].median())
        cand = front.assign(_gap=(front["complexity"] - med).abs()).sort_values(["_gap", "fidelity_loss", "instability"], kind="stable")
        return int(cand.index[0])
    raise ValueError(f"Unknown representative selection '{method}'")


def output_from_explanation(name: str, expl: PFCAExplanation, engine: AttributionEngine, wall_seconds: float, peak_mb: float) -> protocol.MethodOutput:
    """MethodOutput of an explanation rebuilt at another front member, mirroring protocol.pfca_output."""
    U = expl.partition.U
    point = expl.concept_centroids
    shape, support, core = expl.fuzzy_shape, expl.support_percentiles, expl.core_percentiles
    B, M = expl.attribution.n_resamples, expl.attribution.n_model_classes

    def attribution_fn(Xp):
        res = engine.explain(Xp).subset(resamples=np.arange(B), model_classes=np.arange(M))
        pool = aggregate_to_concepts(res.values, U).reshape(-1, Xp.shape[0], U.shape[1])
        return centroid_array(fuzzify_array(pool, axis=0, shape=shape, support_percentiles=support, core_percentiles=core))

    return protocol.MethodOutput(
        name=name,
        level="concept",
        U=U,
        point=point,
        importance=np.abs(point).mean(axis=0),
        order=np.argsort(-np.abs(point), axis=1),
        pool=expl.concept_pool,
        quantiles=expl.quantiles,
        sign_confidence=expl.sign_confidence,
        wall_seconds=wall_seconds,
        peak_mb=peak_mb,
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


# ----------------------------------------------------------------------
# One cell and seed
# ----------------------------------------------------------------------


class CellRun:
    """Problem, tuned reference model, attribution pool and evaluation context of one cell and seed.

    Settings are evaluated lazily and cached, so that the default setting is
    computed once although it appears under every factor.
    """

    def __init__(self, cell: dict, seed: int, cfg: dict, class_names: list[str], default: Setting):
        self.cell, self.seed, self.cfg, self.default = cell, int(seed), cfg, default
        fam, n, d, rho = cell["family"], int(cell["n_samples"]), int(cell["n_features"]), float(cell["rho"])
        self.base_seed = 1000 * self.seed + 7
        self.problem = make_synthetic(fam, n_samples=n, n_features=d, rho=rho, random_state=self.base_seed)
        X_tune, y_tune = self.problem.sample(max(50, int(cfg.get("tune_fraction", 0.34) * n)), random_state=self.base_seed + 1)
        self.X_explain, _ = self.problem.sample(int(cfg["n_explain"]), random_state=self.base_seed + 2)
        X_rep, y_rep = self.problem.sample(n, random_state=self.base_seed + 3)
        ref, self.ref_params, self.ref_score = tune_gradient_boosting(self.problem.X, self.problem.y, X_tune, y_tune, TASK, self.seed, cfg.get("tuning_grid"))
        classes = model_classes_from_names(class_names, TASK, self.seed)
        classes[0] = ("gbm", ref.__class__(**ref.get_params()))
        self.class_names = [name for name, _ in classes]
        self.B_max = max(int(b) for b in cfg["resamples"])
        self.engine = AttributionEngine(model_classes=classes, n_resamples=self.B_max, background_size=cfg["background_size"], explainer=cfg.get("explainer", "auto"), n_jobs=cfg.get("n_jobs", 1), random_state=self.seed)
        with measure(trace_memory=bool(self.cfg.get("trace_memory", False))) as b_pool:
            self.engine.fit(self.problem.X, self.problem.y)
            self.attr: AttributionResult = self.engine.explain(self.X_explain)
        self.pool_budget = b_pool
        rep_model = ref.__class__(**ref.get_params()).fit(X_rep, y_rep)
        rep_attr, _ = shapley_values(rep_model, self.X_explain, X_rep[: cfg["background_size"]], TASK, explainer="tree")
        self.groups = [list(map(int, b)) for b in self.problem.blocks] if cfg.get("include_apriori", False) else None
        self.ctx = protocol.EvaluationContext(
            model_fn=output_function(self.engine.reference_model_, TASK),
            X_explain=self.X_explain,
            background=self.problem.X[: min(500, self.problem.X.shape[0])],
            true_attr=self.problem.true_concept_attributions(self.X_explain),
            U_true=self.problem.true_membership,
            true_importance=self.problem.true_global_importance(self.X_explain),
            replicate_feature_attr=rep_attr,
            perturbation_instances=int(cfg.get("perturbation_instances", 0)),
            n_perturbations=int(cfg.get("n_perturbations", 0)),
            imputation=cfg.get("imputation", "conditional"),
            random_state=self.seed,
        )
        self._cache: dict[Setting, tuple[protocol.MethodOutput, dict]] = {}

    # ------------------------------------------------------------------
    def pfca_kwargs(self, setting: Setting) -> dict:
        kw = dict(self.cfg["pfca"])
        kw["n_concepts_grid"] = tuple(int(k) for k in kw.get("n_concepts_grid", (2, 3, 4, 5, 6)) if int(k) < self.problem.n_features)
        kw["fuzzy_shape"] = setting.fuzzy_shape
        kw["support_percentiles"] = tuple(setting.support)
        kw["core_percentiles"] = tuple(setting.core)
        kw["concept_distance"] = setting.concept_distance
        kw["solver"] = setting.solver
        kw["knee_method"] = setting.knee_method if setting.knee_method in KNEE_METHODS else self.default.knee_method
        kw["nsga_pop_size"] = int(self.cfg.get("nsga_pop_size", 40))
        kw["nsga_generations"] = int(self.cfg.get("nsga_generations", 30))
        return kw

    def subset(self, setting: Setting) -> AttributionResult:
        return self.attr.subset(resamples=np.arange(int(setting.n_resamples)), model_classes=np.arange(int(setting.n_model_classes)))

    def run(self, setting: Setting) -> tuple[protocol.MethodOutput, dict]:
        """Explain and evaluate one setting (cached)."""
        if setting in self._cache:
            return self._cache[setting]
        if setting.knee_method in ALTERNATIVE_SELECTIONS:
            result = self._run_alternative(setting)
        else:
            out = protocol.pfca_output("pfca", self.engine, self.subset(setting), self.problem.feature_names, self.pfca_kwargs(setting), self.groups, trace_memory=bool(self.cfg.get("trace_memory", False)))
            result = (out, protocol.evaluate(out, self.ctx))
        self._cache[setting] = result
        return result

    def _run_alternative(self, setting: Setting) -> tuple[protocol.MethodOutput, dict]:
        base_out, _ = self.run(setting.replace(knee_method=self.default.knee_method))
        base_expl: PFCAExplanation = base_out.extra["explanation"]
        kw = self.pfca_kwargs(setting)
        if self.groups is not None:
            kw["apriori_groups"] = self.groups
            kw.setdefault("partition_source", "both")
        with measure(trace_memory=bool(self.cfg.get("trace_memory", False))) as b:
            index = alternative_index(base_expl.selection, setting.knee_method)
            expl = PFCAExplainer(**kw).explanation_at(base_expl, index)
        out = output_from_explanation("pfca", expl, self.engine, base_out.wall_seconds + b.wall_seconds, max(base_out.peak_mb, b.peak_python_mb))
        out.extra["selected_index"] = index
        return out, protocol.evaluate(out, self.ctx)

    # ------------------------------------------------------------------
    def fixed_quantiles(self, setting: Setting, configuration) -> PFCAExplanation:
        """Explanation of a setting with the partition and configuration forced through the fixed solver."""
        kw = self.pfca_kwargs(setting)
        kw["solver"] = "fixed"
        fc = {"sparsity": int(configuration.sparsity), "alpha": float(configuration.alpha)}
        if configuration.partition == "apriori":
            fc["partition"] = "apriori"
        else:
            fc["n_concepts"] = int(configuration.n_concepts)
            if configuration.fuzzifier is not None:
                fc["fuzzifier"] = float(configuration.fuzzifier)
        kw["fixed_configuration"] = fc
        if self.groups is not None:
            kw["apriori_groups"] = self.groups
            kw.setdefault("partition_source", "both")
        explainer = PFCAExplainer.from_engine(self.engine, self.problem.feature_names, **kw)
        return explainer.explain_with(self.subset(setting))

    def stabilization(self, plan: list[tuple[str, str, Setting]]) -> dict[str, dict]:
        """Mean absolute difference of the quantile arrays between every B and the largest B (resamples factor)."""
        entries = [(name, s) for f, name, s in plan if f == "resamples"]
        if not entries:
            return {}
        largest = max(entries, key=lambda e: e[1].n_resamples)[1]
        out_max, _ = self.run(largest)
        chosen = out_max.extra["explanation"].configuration
        ref = self.fixed_quantiles(largest, chosen)
        scale = float(np.mean(np.abs(ref.quantiles[..., 2])))
        result = {}
        for name, s in entries:
            try:
                e = ref if s == largest else self.fixed_quantiles(s, chosen)
            except Exception as exc:  # the fixed configuration may retain no concept at a small B
                result[name] = {"stabilization_error": f"{type(exc).__name__}: {exc}", "fixed_partition": chosen.partition, "fixed_sparsity": int(chosen.sparsity), "fixed_alpha": float(chosen.alpha)}
                continue
            mad = float(np.mean(np.abs(e.quantiles - ref.quantiles)))
            result[name] = {
                "quantile_mad_to_largest_b": mad,
                "quantile_mad_to_largest_b_rel": mad / scale if scale > 1e-12 else float("nan"),
                "global_quantile_mad_to_largest_b": float(np.mean(np.abs(e.global_quantiles - ref.global_quantiles))),
                "support_width_fixed_configuration": float(np.mean(e.quantiles[..., 4] - e.quantiles[..., 0])),
                "sign_confidence_mad_to_largest_b": float(np.mean(np.abs(e.sign_confidence - ref.sign_confidence))),
                "fixed_partition": chosen.partition,
                "fixed_sparsity": int(chosen.sparsity),
                "fixed_alpha": float(chosen.alpha),
            }
        return result

    # ------------------------------------------------------------------
    def common_fields(self) -> dict:
        return {
            "cell": cell_label(self.cell),
            **{k: self.cell[k] for k in ("family", "n_samples", "n_features", "rho")},
            "seed": self.seed,
            "base_seed": self.base_seed,
            "pool_n_resamples": self.B_max,
            "pool_n_model_classes": len(self.class_names),
            "pool_model_classes": ",".join(self.class_names),
            "pool_wall_seconds": self.pool_budget.wall_seconds,
            "pool_peak_mb": self.pool_budget.peak_python_mb,
            "reference_params": str(self.ref_params),
            "reference_tuning_score": self.ref_score,
            "noise_sd": self.problem.noise_sd,
            "n_factors": self.problem.n_factors,
            "n_features_total": self.problem.n_features,
        }

    def row(self, factor: str, name: str, setting: Setting, out: protocol.MethodOutput, metrics: dict, extras: dict) -> dict:
        row = self.common_fields()
        row.update({"factor": factor, "setting": name, "model_classes_used": ",".join(self.class_names[: setting.n_model_classes])})
        row.update(setting.descriptors())
        row.update(flatten_row(metrics))
        expl: PFCAExplanation = out.extra["explanation"]
        sel = expl.selection
        index = int(out.extra.get("selected_index", sel.knee))
        width = expl.quantiles[..., 4] - expl.quantiles[..., 0]
        scale = float(np.mean(np.abs(expl.quantiles[..., 2])))
        row.update(
            {
                "sparsity": int(expl.configuration.sparsity),
                "selected_index": index,
                "selected_fidelity_loss": float(sel.objectives[index, 0]),
                "selected_complexity": float(sel.objectives[index, 1]),
                "selected_instability": float(sel.objectives[index, 2]),
                "mean_support_width": float(np.mean(width)),
                "mean_support_width_rel": float(np.mean(width)) / scale if scale > 1e-12 else float("nan"),
                "mean_sign_confidence": float(np.mean(expl.sign_confidence)),
                "explain_wall_seconds": out.wall_seconds,
            }
        )
        row.update(extras)
        return row

    def run_plan(self, plan: list[tuple[str, str, Setting]]) -> list[dict]:
        rows = []
        extras_by_entry: dict[tuple[str, str], dict] = {}
        try:
            for name, stab in self.stabilization(plan).items():
                extras_by_entry[("resamples", name)] = stab
        except Exception as exc:
            traceback.print_exc()
            for f, name, _ in plan:
                if f == "resamples":
                    extras_by_entry[(f, name)] = {"stabilization_error": f"{type(exc).__name__}: {exc}"}
        solver_settings = {name: s for f, name, s in plan if f == "solver"}
        if "grid" in solver_settings and "nsga2" in solver_settings:
            try:
                g_out, _ = self.run(solver_settings["grid"])
                n_out, _ = self.run(solver_settings["nsga2"])
                comp = compare_fronts(g_out.extra["explanation"].selection, n_out.extra["explanation"].selection)
                extras_by_entry[("solver", "nsga2")] = dict(comp, hypervolume=comp["hypervolume_nsga2"])
                extras_by_entry[("solver", "grid")] = {"hypervolume": comp["hypervolume_grid"], "n_front_grid": comp["n_front_grid"], "n_evaluated_grid": comp["n_evaluated_grid"]}
            except Exception as exc:
                traceback.print_exc()
                extras_by_entry[("solver", "nsga2")] = {"front_comparison_error": f"{type(exc).__name__}: {exc}"}
        for factor, name, setting in plan:
            extras = extras_by_entry.get((factor, name), {})
            try:
                out, metrics = self.run(setting)
                rows.append(self.row(factor, name, setting, out, metrics, extras))
            except Exception as exc:  # record the failure and continue with the other settings
                traceback.print_exc()
                row = self.common_fields()
                row.update({"factor": factor, "setting": name, "method": "pfca"})
                row.update(setting.descriptors())
                row.update(extras)
                row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
        return rows

    def run_record(self, default: Setting) -> dict:
        rec = self.common_fields()
        rec.update(
            {
                "default_n_resamples": default.n_resamples,
                "default_n_model_classes": default.n_model_classes,
                "n_explain": int(self.X_explain.shape[0]),
                "pool_seed_first": int(self.engine.seeds_[0, 0]),
                "pool_seeds": self.engine.seeds_.tolist(),
            }
        )
        return rec


# ----------------------------------------------------------------------


def summarize(metrics_path: Path, out_path: Path) -> pd.DataFrame | None:
    """Mean and standard deviation over seeds per cell, factor and setting."""
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if df.empty:
        return None
    by = ["cell", "factor", "setting"]
    skip = set(by) | {"seed", "base_seed", "n_samples", "n_features", "rho", "pool_n_resamples", "pool_n_model_classes", "selected_index", "n_factors", "n_features_total"}
    num = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(by, sort=False)[num]
    mean, sd = g.mean(), g.std()
    sd.columns = [f"{c}_sd" for c in sd.columns]
    summary = pd.concat([g.size().rename("n_seeds"), mean, sd], axis=1).copy().reset_index()
    summary.to_csv(out_path, index=False)
    show = [c for c in ("n_seeds", "recovery_error", "rank_recovery", "rank_stability", "support_coverage_truth", "core_coverage_truth", "sign_confidence_ece", "n_front", "n_type2", "mean_support_width", "quantile_mad_to_largest_b", "front_recovery", "hypervolume", "explain_wall_seconds") if c in summary.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4g}".format):
        print(summary[by + show].to_string(index=False))
    return summary


def parse_factors(text: str | None) -> list[str]:
    if not text:
        return list(FACTORS)
    factors = [t.strip() for t in text.split(",") if t.strip()]
    unknown = [f for f in factors if f not in FACTORS]
    if unknown:
        raise SystemExit(f"Unknown factors {unknown}; choose from {list(FACTORS)}")
    return [f for f in FACTORS if f in factors]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/configs/sensitivity.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--name", default="sensitivity")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="override the number of seeds")
    ap.add_argument("--factors", default=None, help="comma separated subset of " + ",".join(FACTORS))
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"n_jobs": args.n_jobs, "n_seeds": args.seeds})
    factors = parse_factors(args.factors)
    cfg["factors_run"] = factors
    class_names = resolve_model_class_names(cfg)
    default = default_setting(cfg, len(class_names))
    cfg["resolved_default"] = default.descriptors()
    cfg["resolved_model_classes"] = class_names
    plan = build_plan(cfg, default, len(class_names), factors)
    out_dir = prepare_run(args.results, args.name, cfg)
    writer = AlignedResultWriter(out_dir / "metrics.csv", KEY_COLUMNS)
    runs = AlignedResultWriter(out_dir / "runs.csv", ["cell", "seed"])
    cells = [dict(c) for c in cfg["cells"]]
    n_seeds = int(cfg["n_seeds"])
    print(f"[sensitivity] {len(cells)} cells x {n_seeds} seeds x {len(plan)} settings, factors {factors}, results in {out_dir}")
    print(f"[sensitivity] model classes {class_names}; default setting {default.descriptors()}")
    t0 = time.time()
    for i, cell in enumerate(cells):
        label = cell_label(cell)
        for seed in range(n_seeds):
            if all(writer.is_done({"cell": label, "seed": seed, "factor": f, "setting": s}) for f, s, _ in plan):
                continue
            t = time.time()
            run = CellRun(cell, seed, cfg, class_names, default)
            print(f"[sensitivity] {label} seed {seed}: base seed {run.base_seed}, pool seeds start at {int(run.engine.seeds_[0, 0])}, pool {run.B_max} x {len(run.class_names)} fitted in {run.pool_budget.wall_seconds:.1f}s", flush=True)
            rows = run.run_plan(plan)
            writer.append(rows)
            runs.append([run.run_record(default)])
            print(f"[sensitivity] cell {i + 1}/{len(cells)} {label} seed {seed} done in {time.time() - t:.1f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    summarize(out_dir / "metrics.csv", out_dir / "summary.csv")
    print("[sensitivity] finished")


if __name__ == "__main__":
    main()
