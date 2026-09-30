"""Phase A: synthetic ground truth experiments (Sections 6.1 and 7.3 of the guide).

For every cell of the design (family x sample size x dimension x within block
correlation x seed) a training sample, a tuning sample and 200 fresh
instances to explain are generated, the reference gradient boosting model is
tuned, a pool of B resamples by M model classes is fitted, every method is
run from that pool and all metrics of Table 1 are stored per method and cell
in a CSV file that supports resuming.

Usage
-----
python experiments/run_phase_a.py --config experiments/configs/phase_a.yaml --results results
python experiments/run_phase_a.py --config experiments/configs/phase_a_smoke.yaml --results results --name phase_a_smoke
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ResultWriter, flatten_row, load_config, model_classes_from_names, prepare_run, tune_gradient_boosting  # noqa: E402

from pfca.attribution import AttributionEngine, output_function, shapley_values  # noqa: E402
from pfca.evaluation import protocol  # noqa: E402
from pfca.evaluation.budget import measure  # noqa: E402
from pfca.evaluation.synthetic import make_synthetic, phase_a_grid  # noqa: E402


def run_cell(cell: dict, cfg: dict) -> tuple[list[dict], dict]:
    fam, n, d, rho, seed = cell["family"], cell["n_samples"], cell["n_features"], cell["rho"], cell["seed"]
    base_seed = 1000 * seed + 7
    problem = make_synthetic(fam, n_samples=n, n_features=d, rho=rho, random_state=base_seed)
    X_tune, y_tune = problem.sample(max(50, int(cfg["tune_fraction"] * n)), random_state=base_seed + 1)
    X_explain, _ = problem.sample(cfg["n_explain"], random_state=base_seed + 2)
    X_rep, y_rep = problem.sample(n, random_state=base_seed + 3)
    task = "regression"
    ref, ref_params, ref_score = tune_gradient_boosting(problem.X, problem.y, X_tune, y_tune, task, seed)
    classes = model_classes_from_names(cfg["model_classes"], task, seed)
    tuned = ("gbm", ref.__class__(**ref.get_params()))
    if classes and classes[0][0] == "gbm":
        classes[0] = tuned
    else:
        classes = [tuned] + [c for c in classes if c[0] != "gbm"]
    engine = AttributionEngine(model_classes=classes, n_resamples=cfg["n_resamples"], background_size=cfg["background_size"], explainer=cfg.get("explainer", "auto"), n_jobs=cfg.get("n_jobs", 1), random_state=seed)
    with measure() as b_pool:
        engine.fit(problem.X, problem.y)
        attr = engine.explain(X_explain)
    # replicated attribution from a fresh training sample (calibration metrics)
    rep_model = ref.__class__(**ref.get_params()).fit(X_rep, y_rep)
    rep_attr, _ = shapley_values(rep_model, X_explain, X_rep[: cfg["background_size"]], task, explainer="tree")
    groups = [list(map(int, b)) for b in problem.blocks]
    ctx = protocol.EvaluationContext(
        model_fn=output_function(engine.reference_model_, task),
        X_explain=X_explain,
        background=problem.X[: min(500, problem.X.shape[0])],
        true_attr=problem.true_concept_attributions(X_explain),
        U_true=problem.true_membership,
        true_importance=problem.true_global_importance(X_explain),
        replicate_feature_attr=rep_attr,
        perturbation_instances=cfg.get("perturbation_instances", 20),
        n_perturbations=cfg.get("n_perturbations", 3),
        imputation=cfg.get("imputation", "conditional"),
        random_state=seed,
    )
    pfca_kwargs = dict(cfg["pfca"])
    pfca_kwargs["n_concepts_grid"] = tuple(k for k in pfca_kwargs.get("n_concepts_grid", (2, 3, 4, 5, 6)) if k < problem.n_features)
    outputs = {}
    rows = []
    for name in cfg["methods"]:
        try:
            if name in protocol.PFCA_VARIANTS:
                out = protocol.pfca_output(name, engine, attr, problem.feature_names, pfca_kwargs, groups)
            elif name == "shap":
                out = protocol.shap_output(engine, attr, task, wall=b_pool.wall_seconds / (engine.n_resamples * len(classes)))
            elif name == "bootstrapped_shap":
                out = protocol.bootstrapped_shap_output(engine, attr, wall=b_pool.wall_seconds / len(classes))
            elif name == "grouped_shap":
                out = protocol.grouped_shap_output(engine, attr, X_explain, groups, task, background_size=cfg.get("grouped_background", 50), random_state=seed)
            elif name == "integrated_gradients":
                out = protocol.integrated_gradients_output(engine, X_explain, task)
            else:
                raise ValueError(f"unknown method {name}")
            outputs[name] = out
            r = protocol.evaluate(out, ctx)
        except Exception as exc:  # record the failure and continue with the other methods
            traceback.print_exc()
            r = {"method": name, "error": f"{type(exc).__name__}: {exc}"}
        r.update(cell)
        r.update({"pool_wall_seconds": b_pool.wall_seconds, "pool_peak_mb": b_pool.peak_python_mb, "reference_params": str(ref_params), "reference_tuning_score": ref_score, "noise_sd": problem.noise_sd, "n_factors": problem.n_factors, "n_features_total": problem.n_features})
        if name in protocol.PFCA_VARIANTS and "explanation" in getattr(out, "extra", {}):
            r["pfca_wall_seconds"] = out.wall_seconds + b_pool.wall_seconds
        rows.append(flatten_row(r))
    # duplication experiment: rerun on the problem without the duplicate columns
    if fam == "redundancy" and problem.duplicates:
        keep = np.arange(problem.n_features - len(problem.duplicates))
        base_engine = AttributionEngine(model_classes=classes, n_resamples=cfg["n_resamples"], background_size=cfg["background_size"], explainer=cfg.get("explainer", "auto"), n_jobs=cfg.get("n_jobs", 1), random_state=seed)
        base_engine.fit(problem.X[:, keep], problem.y)
        base_attr = base_engine.explain(X_explain[:, keep])
        base_groups = [[j for j in g if j in set(keep.tolist())] for g in groups]
        base_kwargs = dict(pfca_kwargs)
        base_kwargs["n_concepts_grid"] = tuple(k for k in base_kwargs["n_concepts_grid"] if k < keep.size)
        src, pos, sd = problem.duplicates[0]
        for name, out in outputs.items():
            try:
                if name in protocol.PFCA_VARIANTS:
                    before = protocol.pfca_output(name, base_engine, base_attr, [problem.feature_names[j] for j in keep], base_kwargs, base_groups)
                elif name == "shap":
                    before = protocol.shap_output(base_engine, base_attr, task)
                elif name == "bootstrapped_shap":
                    before = protocol.bootstrapped_shap_output(base_engine, base_attr)
                elif name == "grouped_shap":
                    before = protocol.grouped_shap_output(base_engine, base_attr, X_explain[:, keep], base_groups, task, background_size=cfg.get("grouped_background", 50), random_state=seed)
                else:
                    continue
                change = protocol.duplication_change(before, out, src, pos, problem.true_membership)
                for r in rows:
                    if r["method"] == name:
                        r["duplication_change"] = change
            except Exception as exc:
                traceback.print_exc()
                for r in rows:
                    if r["method"] == name:
                        r["duplication_error"] = f"{type(exc).__name__}: {exc}"
    return rows, {"cell": cell}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/configs/phase_a.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--name", default="phase_a")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="override the number of seeds")
    ap.add_argument("--only-family", default=None)
    ap.add_argument("--max-cells", type=int, default=None, help="stop after this many cells (for testing)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"n_jobs": args.n_jobs, "n_seeds": args.seeds})
    out_dir = prepare_run(args.results, args.name, cfg)
    writer = ResultWriter(out_dir / "metrics.csv", ["family", "n_samples", "n_features", "rho", "seed", "method"])
    cells = phase_a_grid(cfg["families"], cfg["n_samples"], cfg["n_features"], cfg["rhos"], cfg["n_seeds"])
    if args.only_family:
        cells = [c for c in cells if c["family"] == args.only_family]
    if args.max_cells:
        cells = cells[: args.max_cells]
    print(f"[phase A] {len(cells)} cells, results in {out_dir}")
    t0 = time.time()
    for i, cell in enumerate(cells):
        if all(writer.is_done({**cell, "method": m}) for m in cfg["methods"]):
            continue
        t = time.time()
        rows, _ = run_cell(cell, cfg)
        writer.append(rows)
        print(f"[phase A] cell {i + 1}/{len(cells)} {cell} done in {time.time() - t:.1f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    print("[phase A] finished")


if __name__ == "__main__":
    main()
