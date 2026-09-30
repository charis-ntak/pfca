"""Phase B: public benchmark experiments (Sections 6.2 and 7.1 to 7.3 of the guide).

For every benchmark dataset and every repetition of the 60/20/20 split, the
reference gradient boosting model is tuned on the tuning split and fixed, a
pool of B resamples by M model classes is fitted on the training split, a
fixed set of test instances is explained by every method from that pool, and
the metrics of Table 1 that apply to Phase B (faithfulness, stability,
calibration against a replicated attribution, cost) are stored per dataset,
repetition and method in a CSV file that supports resuming.

Phase B has no ground truth. The replicated attribution used by the
calibration metrics comes from the reference model class refitted on the
training split of repetition (r + 1) mod n_repeats and attributed on the
explained instances of repetition r that are not part of that training split.
For classification the explained output is the probability of the positive
class (label 1). Methods that need a priori groups (pfca_apriori and
grouped_shap) run only on datasets that provide groups.

Usage
-----
python experiments/run_phase_b.py --config experiments/configs/phase_b.yaml --results results
python experiments/run_phase_b.py --config experiments/configs/phase_b_smoke.yaml --results results --name phase_b_smoke
python experiments/run_phase_b.py --config experiments/configs/phase_b.yaml --datasets breast_cancer,spambase --max-repeats 3 --n-jobs 2

Outputs, in <results>/<name>/
-----------------------------
config.yaml, environment.json   frozen configuration, package versions and command line
metrics.csv                     one row per dataset, repetition and method with the metrics of Table 1
runs.csv                        one row per dataset and repetition: seeds, split sizes, tuned reference
                                parameters, test performance and the indices of the explained instances
summary.csv                     mean of the numeric metrics over repetitions per dataset and method
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ResultWriter, flatten_row, load_config, model_classes_from_names, prepare_run, tune_gradient_boosting  # noqa: E402

from pfca.attribution import AttributionEngine, output_function, shapley_values  # noqa: E402
from pfca.evaluation import datasets, protocol  # noqa: E402
from pfca.evaluation.budget import measure  # noqa: E402
from pfca.utils import as_index_groups  # noqa: E402

GROUP_METHODS = ("pfca_apriori", "grouped_shap")
POSITIVE_CLASS = 1


# Column alignment of ragged rows is handled by common.ResultWriter; the name is kept for readability.
AlignedResultWriter = ResultWriter

def repetition_seed(base_seed: int, r: int) -> int:
    """Seed of repetition r; every random component of the repetition derives from it."""
    return int(base_seed) + 1000 * int(r) + 7


def load_datasets(cfg: dict) -> list[datasets.Dataset]:
    """Load the datasets named in the configuration, or the curated list when none is named."""
    names = cfg.get("datasets")
    lo, hi = int(cfg.get("min_features", 15)), int(cfg.get("max_features", 100))
    kwargs = {"max_rows": cfg.get("max_rows"), "random_state": int(cfg.get("seed", 0)), "data_home": cfg.get("data_home")}
    if names is None:
        return datasets.load_benchmarks(lo, hi, None, **kwargs)
    listed = {b["name"] for b in datasets.BENCHMARKS}
    out = []
    for name in names:
        if name in listed:
            out.extend(datasets.load_benchmarks(lo, hi, [name], **kwargs))
            continue
        try:
            ds = datasets.load_benchmark(name, **kwargs)
        except Exception as exc:
            print(f"[phase B] skipping {name}: {type(exc).__name__}: {exc}")
            continue
        if not (lo <= ds.n_features <= hi):
            print(f"[phase B] skipping {name}: {ds.n_features} features outside [{lo}, {hi}]")
            continue
        out.append(ds)
    return out


def applicable_methods(methods: list[str], ds: datasets.Dataset, cfg: dict) -> list[str]:
    """Methods of the configuration that apply to the dataset."""
    has_mlp = any("mlp" in str(n) for n in cfg.get("model_classes", []))
    out = []
    for m in methods:
        if m in GROUP_METHODS and ds.groups is None:
            print(f"[phase B] {ds.name}: skipping {m}, the dataset has no a priori groups")
        elif m == "integrated_gradients" and not has_mlp:
            print(f"[phase B] {ds.name}: skipping {m}, no multilayer perceptron among the model classes")
        else:
            out.append(m)
    return out


def reference_test_performance(model, X_test: np.ndarray, y_test: np.ndarray, task: str) -> dict:
    """Test set performance of the fixed reference model, for reporting."""
    from sklearn.metrics import log_loss, r2_score, roc_auc_score

    if task == "classification":
        p = np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)
        auc = float(roc_auc_score(y_test, p)) if np.unique(y_test).size == 2 else float("nan")
        return {"reference_test_log_loss": float(log_loss(y_test, p, labels=[0, 1])), "reference_test_auc": auc}
    pred = model.predict(X_test)
    return {"reference_test_r2": float(r2_score(y_test, pred)), "reference_test_rmse": float(np.sqrt(np.mean((y_test - pred) ** 2)))}


def replicate_attribution(ds: datasets.Dataset, splits: list, r: int, ref, X_explain: np.ndarray, explain_idx: np.ndarray, cfg: dict) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Replicated feature level attribution for the calibration metrics.

    The reference model class is refitted, with the seed of repetition
    r_rep = (r + 1) mod n_repeats, on the training split of r_rep, and
    attributed on the explained instances of repetition r. The mask marks the
    explained instances that are not in that training split; only those are
    used by the calibration metrics.
    """
    n_repeats = len(splits)
    r_rep = (r + 1) % n_repeats
    info = {"replicate_repeat": r_rep, "n_replicate_instances": 0}
    if r_rep == r:
        print("[phase B] a single repetition gives no independent replicate; calibration metrics are skipped")
        info["replicate_repeat"] = None
        return None, None, info
    idx_train_rep = splits[r_rep][1]
    mask = ~np.isin(explain_idx, idx_train_rep)
    info["n_replicate_instances"] = int(mask.sum())
    if not mask.any():
        print(f"[phase B] repetition {r}: every explained instance lies in the replicate training split; calibration metrics are skipped")
        return None, None, info
    seed_rep = repetition_seed(int(cfg.get("seed", 0)), r_rep)
    rng = np.random.default_rng(seed_rep)
    X_rep, y_rep = ds.X[idx_train_rep], ds.y[idx_train_rep]
    rep_model = ref.__class__(**ref.get_params()).set_params(random_state=seed_rep).fit(X_rep, y_rep)
    n_bg = min(int(cfg["background_size"]), X_rep.shape[0])
    bg = X_rep[rng.choice(X_rep.shape[0], size=n_bg, replace=False)]
    rep_attr, _ = shapley_values(rep_model, X_explain, bg, ds.task, explainer="tree", target_class=POSITIVE_CLASS, seed=seed_rep)
    info["replicate_seed"] = seed_rep
    return rep_attr, mask, info


def run_repetition(ds: datasets.Dataset, splits: list, r: int, cfg: dict, out_dir: Path) -> tuple[list[dict], dict]:
    """Run every applicable method on one repetition of the split of one dataset."""
    _, idx_train, idx_tune, idx_test = splits[r]
    seed = repetition_seed(int(cfg.get("seed", 0)), r)
    task = ds.task
    rng = np.random.default_rng(seed)
    n_explain = min(int(cfg["n_explain"]), idx_test.size)
    explain_idx = np.sort(rng.choice(idx_test, size=n_explain, replace=False))
    X_tr, y_tr = ds.X[idx_train], ds.y[idx_train]
    X_tune, y_tune = ds.X[idx_tune], ds.y[idx_tune]
    X_test, y_test = ds.X[idx_test], ds.y[idx_test]
    X_explain = ds.X[explain_idx]
    print(f"[phase B] {ds.name} repetition {r}: seed {seed}, train {idx_train.size}, tune {idx_tune.size}, test {idx_test.size}, explained {n_explain}", flush=True)
    # reference model tuned on the tuning split and fixed for every method
    ref, ref_params, ref_score = tune_gradient_boosting(X_tr, y_tr, X_tune, y_tune, task, seed, cfg.get("tuning_grid"))
    test_perf = reference_test_performance(ref, X_test, y_test, task)
    classes = model_classes_from_names(cfg["model_classes"], task, seed)
    classes[0] = ("gbm", ref.__class__(**ref.get_params()))
    engine = AttributionEngine(
        model_classes=classes,
        n_resamples=cfg["n_resamples"],
        background_size=cfg["background_size"],
        explainer=cfg.get("explainer", "auto"),
        task=task,
        target_class=POSITIVE_CLASS,
        n_jobs=cfg.get("n_jobs", 1),
        random_state=seed,
    )
    with measure() as b_pool:
        engine.fit(X_tr, y_tr)
        attr = engine.explain(X_explain)
    rep_attr, rep_mask, rep_info = replicate_attribution(ds, splits, r, ref, X_explain, explain_idx, cfg)
    groups = None
    if ds.groups is not None:
        groups = [list(map(int, g)) for g in as_index_groups(ds.groups, ds.n_features, ds.feature_names)]
    ctx = protocol.EvaluationContext(
        model_fn=output_function(engine.reference_model_, task, POSITIVE_CLASS),
        X_explain=X_explain,
        background=X_tr[: min(int(cfg.get("background_rows", 500)), X_tr.shape[0])],
        replicate_feature_attr=rep_attr,
        replicate_mask=rep_mask,
        perturbation_instances=cfg.get("perturbation_instances", 20),
        n_perturbations=cfg.get("n_perturbations", 3),
        perturbation_sigma=cfg.get("perturbation_sigma", 0.05),
        imputation=cfg.get("imputation", "conditional"),
        random_state=seed,
    )
    pfca_kwargs = dict(cfg["pfca"])
    pfca_kwargs["n_concepts_grid"] = tuple(k for k in pfca_kwargs.get("n_concepts_grid", (2, 3, 4, 5, 6)) if k < ds.n_features)
    common = {
        "dataset": ds.name,
        "task": task,
        "repeat": r,
        "seed": seed,
        "n_rows": int(ds.X.shape[0]),
        "n_features": ds.n_features,
        "n_train": int(idx_train.size),
        "n_tune": int(idx_tune.size),
        "n_test": int(idx_test.size),
        "n_explain": n_explain,
        "n_resamples": int(cfg["n_resamples"]),
        "n_model_classes": len(classes),
        "pool_wall_seconds": b_pool.wall_seconds,
        "pool_peak_mb": b_pool.peak_python_mb,
        "reference_params": json.dumps(ref_params),
        "reference_tuning_score": ref_score,
        **test_perf,
        **rep_info,
    }
    rows = []
    for name in applicable_methods(cfg["methods"], ds, cfg):
        out = None
        try:
            if name in protocol.PFCA_VARIANTS:
                out = protocol.pfca_output(name, engine, attr, ds.feature_names, pfca_kwargs, groups)
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
            row = protocol.evaluate(out, ctx)
        except Exception as exc:  # record the failure and continue with the other methods
            traceback.print_exc()
            row = {"method": name, "error": f"{type(exc).__name__}: {exc}"}
        row.update(common)
        if out is not None and name in protocol.PFCA_VARIANTS and "explanation" in out.extra:
            row["pfca_wall_seconds"] = out.wall_seconds + b_pool.wall_seconds
            if cfg.get("save_explanations", False):
                folder = out_dir / "explanations"
                folder.mkdir(exist_ok=True)
                out.extra["explanation"].save(str(folder / f"{ds.name}_r{r}_{name}.json"))
        rows.append(flatten_row(row))
    run_row = {
        **common,
        "split_seed": int(cfg.get("seed", 0)) + r,
        "model_classes": json.dumps([n for n, _ in classes]),
        "engine_seeds": json.dumps(np.asarray(attr.seeds).tolist()),
        "explained_indices": json.dumps([int(i) for i in explain_idx]),
        "methods": json.dumps([row["method"] for row in rows]),
        "n_errors": int(sum("error" in row for row in rows)),
    }
    return rows, run_row


def summarize(metrics_path: Path, out_path: Path) -> pd.DataFrame | None:
    """Mean of the numeric metrics over repetitions per dataset and method."""
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if df.empty:
        return None
    skip = {"repeat", "seed", "n_rows", "n_train", "n_tune", "n_test", "replicate_repeat", "replicate_seed"}
    num = [c for c in df.select_dtypes("number").columns if c not in skip]
    g = df.groupby(["dataset", "method"], sort=False)
    summary = g[num].mean()
    summary.insert(0, "n_repeats", g.size())
    summary = summary.reset_index()
    summary.to_csv(out_path, index=False)
    show = [c for c in ("n_repeats", "deletion_auc", "insertion_auc", "surrogate_r2", "rank_stability", "perturbation_stability", "support_coverage_replicate", "core_coverage_replicate", "sign_confidence_ece", "wall_seconds") if c in summary.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(summary[["dataset", "method"] + show].to_string(index=False))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/configs/phase_b.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--name", default="phase_b")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--datasets", default=None, help="comma separated dataset names overriding the configuration")
    ap.add_argument("--max-repeats", type=int, default=None, help="cap the number of split repetitions (for testing)")
    args = ap.parse_args(argv)
    names = [s.strip() for s in args.datasets.split(",") if s.strip()] if args.datasets else None
    cfg = load_config(args.config, {"n_jobs": args.n_jobs, "datasets": names})
    cfg.setdefault("seed", 0)
    if args.max_repeats is not None:
        cfg["n_repeats"] = int(min(int(cfg["n_repeats"]), args.max_repeats))
    out_dir = prepare_run(args.results, args.name, cfg)
    writer = AlignedResultWriter(out_dir / "metrics.csv", ["dataset", "repeat", "method"])
    runs = AlignedResultWriter(out_dir / "runs.csv", ["dataset", "repeat"])
    dsets = load_datasets(cfg)
    if not dsets:
        print("[phase B] no dataset could be loaded; nothing to do")
        return 1
    print(f"[phase B] {len(dsets)} datasets ({', '.join(f'{d.name}: {d.X.shape[0]} rows, {d.n_features} features, {d.task}' for d in dsets)}), {cfg['n_repeats']} repetitions, seed {cfg['seed']}, results in {out_dir}")
    t0 = time.time()
    for ds in dsets:
        splits = list(datasets.stratified_splits(ds.y, ds.task, n_repeats=int(cfg["n_repeats"]), random_state=int(cfg["seed"])))
        methods = applicable_methods(cfg["methods"], ds, cfg)
        for r in range(len(splits)):
            if all(writer.is_done({"dataset": ds.name, "repeat": r, "method": m}) for m in methods):
                continue
            t = time.time()
            rows, run_row = run_repetition(ds, splits, r, cfg, out_dir)
            writer.append(rows)
            run_row["elapsed_seconds"] = time.time() - t
            runs.append([run_row])
            print(f"[phase B] {ds.name} repetition {r + 1}/{len(splits)} done in {time.time() - t:.1f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    summarize(out_dir / "metrics.csv", out_dir / "summary.csv")
    print("[phase B] finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
