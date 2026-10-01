"""Phase C: psychological questionnaire with known subscales (Sections 6.3 and 7.3 of the guide).

The questionnaire is loaded from a CSV file and a JSON specification of its
subscales with pfca.evaluation.datasets.load_questionnaire (the format is
described in data/README.md). For every repetition of the 60/20/20 split the
reference gradient boosting model is tuned on the tuning split, a pool of B
resamples by M model classes is fitted on the training split, and every
method is run from that pool on a fixed set of test instances. The a priori
subscale grouping enters twice: as a candidate partition (PFCA with
partition_source 'both', the pfca_apriori ablation, and the crisp groups of
grouped SHAP, where the covariates form their own group) and as the external
criterion against which every estimated partition is scored by the adjusted
Rand index and the fuzzy partition coefficient.

Calibration metrics use a replicate attribution as in Phase B: the reference
model class is refitted on the training split of the next repetition and its
Shapley values are computed for the explained instances that do not belong to
that split (replicate_mask), so that the replicate is an independent estimate.

Outputs in the run directory
----------------------------
metrics.csv                        one row per repetition and method with every metric of Table 1 that applies without ground truth
summary_by_method.csv              mean, standard deviation, count and confidence interval of every metric per method
reliability_repeatRR.csv           per construct reliability table of the knee explanation of PFCA and of bootstrapped SHAP aggregated by subscale
reliability.csv                    the same tables stacked over repetitions
reliability_summary.csv            mean widths per method, level and matched subscale across repetitions
pfca_explanation_repeatRR.json     the knee explanation of PFCA (PFCAExplanation.save)
membership_repeatRR.png            membership heat map of the selected partition
overlap_repeatRR.png               overlap of the selected concepts with the subscales
example_repeatRR_instanceII.txt    linguistic description of an example instance (PFCAExplanation.describe)
example_repeatRR_instanceII_*.png  linguistic bars, fuzzy attribution and feature level SHAP bars of the example instance

Usage
-----
python experiments/make_demo_questionnaire.py --n 300 --seed 0
python experiments/run_phase_c.py --config experiments/configs/phase_c_smoke.yaml --results results --name phase_c_smoke
python experiments/run_phase_c.py --config experiments/configs/phase_c.yaml --results results
python experiments/run_phase_c.py --config experiments/configs/phase_c.yaml --csv data/raw/questionnaire.csv --spec data/raw/questionnaire_spec.json --repeats 3
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PerInstanceWriter, ROOT, ResultWriter, flatten_row, load_config, model_classes_from_names, prepare_run, tune_gradient_boosting  # noqa: E402

from pfca.attribution import AttributionEngine, aggregate_to_concepts, output_function, shapley_values  # noqa: E402
from pfca.concepts import apriori_membership  # noqa: E402
from pfca.evaluation import metrics as M  # noqa: E402
from pfca.evaluation import protocol  # noqa: E402
from pfca.evaluation.budget import measure  # noqa: E402
from pfca.evaluation.datasets import Dataset, load_questionnaire, stratified_splits  # noqa: E402
from pfca.evaluation.statistics import summarize_by  # noqa: E402
from pfca.fuzzification import fuzzify_array, sign_confidence, tfn_from_quantiles  # noqa: E402
from pfca.plotting import GRID, MUTED, SEQUENTIAL, TEXT, plot_fuzzy_attribution, plot_linguistic_bars, plot_membership, plot_shap_bars  # noqa: E402

RELIABILITY_KEYS = ["repeat", "method", "concept"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def resolve_path(path: str) -> Path:
    """Resolve a data path relative to the working directory, then to the repository root."""
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p
    alt = ROOT / p
    return alt if alt.exists() else p


def index_groups(ds: Dataset) -> tuple[list[list[int]], list[str]]:
    """Subscales (and the covariate group) as lists of feature indices, in the order of the specification."""
    if not ds.groups:
        raise ValueError("The questionnaire specification defines no subscales.")
    pos = {name: j for j, name in enumerate(ds.feature_names)}
    names = list(ds.groups.keys())
    groups = [[pos[c] for c in ds.groups[g]] for g in names]
    return groups, names


def subscale_overlap(U: np.ndarray, U_sub: np.ndarray) -> np.ndarray:
    """Share of the membership mass of every concept that lies in every subscale, shape (K, S); rows sum to one."""
    O = np.asarray(U, dtype=float).T @ np.asarray(U_sub, dtype=float)
    mass = np.asarray(U, dtype=float).sum(axis=0)
    return O / np.where(mass <= 0, 1.0, mass)[:, None]


def reference_test_scores(model, X_test: np.ndarray, y_test: np.ndarray, task: str) -> dict:
    """Held out performance of the fixed reference model, for reporting."""
    from sklearn.metrics import accuracy_score, log_loss, r2_score

    if task == "regression":
        return {"reference_test_r2": float(r2_score(y_test, model.predict(X_test)))}
    p = np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)
    return {"reference_test_accuracy": float(accuracy_score(y_test, (p >= 0.5).astype(int))), "reference_test_log_loss": float(log_loss(y_test, p, labels=[0, 1]))}


def _width_rows(Q: np.ndarray, sc: np.ndarray, names: list[str], method: str, level: str, share: np.ndarray, subscale_names: list[str], extra=None) -> list[dict]:
    """Rows of the reliability table for a set of fuzzy attributions stored as five quantile vectors (n, K, 5)."""
    support_w = Q[..., 4] - Q[..., 0]
    core_w = Q[..., 3] - Q[..., 1]
    med = Q[..., 2]
    rows = []
    for k, name in enumerate(names):
        mean_abs = float(np.mean(np.abs(med[:, k])))
        row = {
            "method": method,
            "level": level,
            "concept": name,
            "index": k,
            "mean_support_width": float(support_w[:, k].mean()),
            "mean_core_width": float(core_w[:, k].mean()),
            "mean_abs_median": mean_abs,
            "relative_support_width": float(support_w[:, k].mean() / mean_abs) if mean_abs > 1e-12 else float("nan"),
            "relative_core_width": float(core_w[:, k].mean() / mean_abs) if mean_abs > 1e-12 else float("nan"),
            "mean_sign_confidence": float(sc[:, k].mean()),
            "mean_disagreement": float("nan"),
            "fraction_type2": float("nan"),
        }
        if extra is not None:
            row.update(extra(k))
        best = int(np.argmax(share[k]))
        row["matched_subscale"] = subscale_names[best]
        row["matched_overlap"] = float(share[k, best])
        for s, sname in enumerate(subscale_names):
            row[f"overlap_{sname}"] = float(share[k, s])
        rows.append(row)
    return rows


def reliability_table(expl, attr, U_sub: np.ndarray, subscale_names: list[str], feature_names: list[str]) -> pd.DataFrame:
    """Per construct reliability of the knee explanation of PFCA against bootstrapped feature level SHAP.

    For every PFCA concept the table holds the mean support width, mean core
    width, mean sign confidence, mean disagreement index, fraction of type 2
    attributions and the membership overlap with every subscale. The same
    widths are computed for bootstrapped feature level SHAP of the reference
    model class aggregated by subscale (sum of the feature values within each
    subscale for every pool member, then percentiles) and for every single
    item, so that reliability differences between constructs concealed by
    feature level SHAP can be reported.
    """
    shape, sup, core = expl.fuzzy_shape, expl.support_percentiles, expl.core_percentiles
    retained = set(int(k) for k in expl.retained)
    G = [tfn_from_quantiles(q) for q in expl.global_quantiles]
    share_pfca = subscale_overlap(expl.partition.U, U_sub)

    def pfca_extra(k):
        return {
            "retained": k in retained,
            "activity": float(expl.activity[k]),
            "selection_membership": float(expl.concept_selection_membership[k]),
            "concept_fuzziness": float(expl.partition.fuzziness[k]),
            "global_importance_centroid": float(G[k].centroid()),
            "mean_disagreement": float(expl.disagreement[:, k].mean()),
            "fraction_type2": float(expl.type2_mask[:, k].mean()),
        }

    rows = _width_rows(expl.quantiles, expl.sign_confidence, expl.concept_names, "pfca", "concept", share_pfca, subscale_names, pfca_extra)
    # bootstrapped feature level SHAP of the reference model class, aggregated by subscale
    pool = attr.values[:, 0]
    agg = aggregate_to_concepts(pool, U_sub)
    Q_sub = fuzzify_array(agg, axis=0, shape=shape, support_percentiles=sup, core_percentiles=core)
    rows += _width_rows(Q_sub, sign_confidence(agg, axis=0), subscale_names, "bootstrapped_shap_by_subscale", "subscale", np.eye(len(subscale_names)), subscale_names)
    # bootstrapped feature level SHAP per item
    Q_feat = fuzzify_array(pool, axis=0, shape=shape, support_percentiles=sup, core_percentiles=core)
    rows += _width_rows(Q_feat, sign_confidence(pool, axis=0), list(feature_names), "bootstrapped_shap", "feature", U_sub, subscale_names)
    return pd.DataFrame(rows)


def plot_overlap(share: np.ndarray, concept_names: list[str], subscale_names: list[str], path: Path) -> None:
    """Heat map of the share of every concept's membership mass in every subscale (single hue, direct labels)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    K, S = share.shape
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * S, 0.8 + 0.45 * K))
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for k in range(K):
        for s in range(S):
            ax.text(s, k, f"{share[k, s]:.2f}", ha="center", va="center", fontsize=8, color=TEXT if share[k, s] < 0.6 else "white")
    ax.set_xticks(range(S))
    ax.set_xticklabels(subscale_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"C{k + 1}" for k in range(K)], fontsize=8)
    ax.set_xlabel("subscale")
    ax.set_ylabel("concept")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_examples(expl, attr, feature_names: list[str], share: np.ndarray, subscale_names: list[str], instances, out_dir: Path, r: int) -> list[str]:
    """Write the linguistic description, the explanation JSON and the figures of the example instances."""
    import matplotlib.pyplot as plt

    written = []
    tag = f"repeat{r:02d}"
    json_path = out_dir / f"pfca_explanation_{tag}.json"
    expl.save(str(json_path))
    written.append(json_path.name)
    ax = plot_membership(expl.partition)
    ax.figure.savefig(out_dir / f"membership_{tag}.png", bbox_inches="tight")
    plt.close(ax.figure)
    written.append(f"membership_{tag}.png")
    plot_overlap(share, expl.concept_names, subscale_names, out_dir / f"overlap_{tag}.png")
    written.append(f"overlap_{tag}.png")
    for i in instances:
        i = int(i)
        if i < 0 or i >= expl.n_explain:
            print(f"[phase C] example instance {i} is outside the explained set of size {expl.n_explain}; skipped")
            continue
        stem = f"example_{tag}_instance{i:02d}"
        text = expl.describe(i, retained_only=True)
        full = expl.describe(i, retained_only=False)
        with open(out_dir / f"{stem}.txt", "w", encoding="utf8") as fh:
            fh.write("Retained concepts of the knee explanation\n")
            fh.write(text + "\n\n")
            fh.write("All concepts\n")
            fh.write(full + "\n\n")
            fh.write("Concept membership overlap with the subscales (share of membership mass)\n")
            for k, name in enumerate(expl.concept_names):
                fh.write("  " + name + ": " + ", ".join(f"{s} {share[k, j]:.2f}" for j, s in enumerate(subscale_names)) + "\n")
            fh.write(f"\nExplanation JSON: {json_path.name}\n")
        written.append(f"{stem}.txt")
        ax = plot_linguistic_bars(expl, i)
        ax.set_title(f"PFCA linguistic explanation, instance {i}", loc="left")
        ax.figure.savefig(out_dir / f"{stem}_linguistic_bars.png", bbox_inches="tight")
        plt.close(ax.figure)
        ax = plot_fuzzy_attribution(expl, i)
        ax.set_title(f"Fuzzy concept attributions, instance {i}", loc="left")
        ax.figure.savefig(out_dir / f"{stem}_fuzzy_attribution.png", bbox_inches="tight")
        plt.close(ax.figure)
        ax = plot_shap_bars(attr.values[0, 0, i], feature_names, title=f"Feature level SHAP, instance {i}")
        ax.figure.savefig(out_dir / f"{stem}_shap_bars.png", bbox_inches="tight")
        plt.close(ax.figure)
        written += [f"{stem}_linguistic_bars.png", f"{stem}_fuzzy_attribution.png", f"{stem}_shap_bars.png"]
    return written


# ----------------------------------------------------------------------
# one repetition
# ----------------------------------------------------------------------


def run_repeat(ds: Dataset, split: tuple, next_train: np.ndarray | None, cfg: dict, out_dir: Path) -> tuple[list[dict], pd.DataFrame | None]:
    r, idx_train, idx_tune, idx_test = split
    seed = int(cfg.get("seed", 0)) + int(r)
    rng = np.random.default_rng(seed)
    task = ds.task
    X, y = ds.X, ds.y
    X_tr, y_tr = X[idx_train], y[idx_train]
    X_tune, y_tune = X[idx_tune], y[idx_tune]
    X_test, y_test = X[idx_test], y[idx_test]
    n_explain = min(int(cfg["n_explain"]), idx_test.size)
    idx_explain = idx_test[:n_explain]
    X_explain = X[idx_explain]
    groups, group_names = index_groups(ds)
    U_sub = apriori_membership(groups, ds.n_features)
    print(f"[phase C] repeat {r}: seed {seed}, train {idx_train.size}, tune {idx_tune.size}, test {idx_test.size}, explained {n_explain}", flush=True)

    # reference model (fixed for every method) and pool
    ref, ref_params, ref_score = tune_gradient_boosting(X_tr, y_tr, X_tune, y_tune, task, seed, grid=cfg.get("tuning_grid"))
    classes = model_classes_from_names(cfg["model_classes"], task, seed)
    classes[0] = ("gbm", ref.__class__(**ref.get_params()))
    engine = AttributionEngine(model_classes=classes, n_resamples=cfg["n_resamples"], background_size=cfg["background_size"], explainer=cfg.get("explainer", "auto"), task=task, n_jobs=cfg.get("n_jobs", 1), random_state=seed)
    trace = bool(cfg.get("trace_memory", False))
    with measure(trace_memory=trace) as b_pool:
        engine.fit(X_tr, y_tr)
        attr = engine.explain(X_explain)
    test_scores = reference_test_scores(engine.reference_model_, X_test, y_test, task)

    # replicate attribution from the training split of the next repetition
    rep_mask = ~np.isin(idx_explain, next_train) if next_train is not None else np.zeros(len(idx_explain), dtype=bool)
    rep_attr = None
    if rep_mask.any():
        X_rep, y_rep = X[next_train], y[next_train]
        rep_model = ref.__class__(**ref.get_params()).fit(X_rep, y_rep)
        bg_size = min(int(cfg["background_size"]), X_rep.shape[0])
        bg = X_rep[rng.choice(X_rep.shape[0], size=bg_size, replace=False)]
        rep_attr, _ = shapley_values(rep_model, X_explain, bg, task, explainer="tree")
    else:
        print("[phase C] every explained instance belongs to the next training split; calibration metrics are skipped", flush=True)

    ctx = protocol.EvaluationContext(
        model_fn=output_function(engine.reference_model_, task),
        X_explain=X_explain,
        background=X_tr[: min(500, X_tr.shape[0])],
        replicate_feature_attr=rep_attr,
        replicate_mask=rep_mask if rep_attr is not None else None,
        perturbation_instances=cfg.get("perturbation_instances", 20),
        n_perturbations=cfg.get("n_perturbations", 3),
        imputation=cfg.get("imputation", "conditional"),
        random_state=seed,
    )
    pfca_kwargs = dict(cfg["pfca"])
    pfca_kwargs["n_concepts_grid"] = tuple(k for k in pfca_kwargs.get("n_concepts_grid", (2, 3, 4, 5, 6)) if k < ds.n_features)
    common = {
        "repeat": int(r),
        "seed": seed,
        "dataset": ds.name,
        "task": task,
        "n_features": ds.n_features,
        "n_subscales": len(group_names) - (1 if "covariates" in group_names else 0),
        "n_train": int(idx_train.size),
        "n_tune": int(idx_tune.size),
        "n_test": int(idx_test.size),
        "n_explain": n_explain,
        "n_replicate": int(rep_mask.sum()),
        "pool_wall_seconds": b_pool.wall_seconds,
        "pool_peak_mb": b_pool.peak_python_mb,
        "reference_params": str(ref_params),
        "reference_tuning_score": ref_score,
        **test_scores,
    }
    outputs = {}
    rows = []
    matched: dict[str, int] = {}
    for name in cfg["methods"]:
        out = None
        try:
            if name in protocol.PFCA_VARIANTS:
                out = protocol.pfca_output(name, engine, attr, ds.feature_names, pfca_kwargs, groups, trace_memory=trace)
                if name == "pfca":
                    matched = protocol.matched_sparsity(out)
            elif name == "shap":
                out = protocol.shap_output(engine, attr, task, wall=b_pool.wall_seconds / (engine.n_resamples * len(classes)), n_retained=matched.get("feature"))
            elif name == "bootstrapped_shap":
                out = protocol.bootstrapped_shap_output(engine, attr, wall=b_pool.wall_seconds / len(classes), n_retained=matched.get("feature"))
            elif name == "grouped_shap":
                out = protocol.grouped_shap_output(engine, attr, X_explain, groups, task, background_size=cfg.get("grouped_background"), random_state=seed, n_retained=matched.get("concept"), trace_memory=trace)
            elif name == "integrated_gradients":
                out = protocol.integrated_gradients_output(engine, X_explain, task)
            else:
                raise ValueError(f"unknown method {name}")
            outputs[name] = out
            res = protocol.evaluate(out, ctx)
            # concept recovery of the estimated partition against the subscale partition (Table 1, phases A and C)
            cr = M.concept_recovery(out.U, U_sub)
            res["ari_subscales"] = cr["ari"]
            res["fpc"] = cr["fpc"]
            res["n_concepts_out"] = int(out.U.shape[1])
            share = subscale_overlap(out.U, U_sub)
            res["mean_matched_overlap"] = float(share.max(axis=1).mean())
            if name in protocol.PFCA_VARIANTS:
                expl = out.extra["explanation"]
                res["knee_partition_is_apriori"] = bool(expl.configuration.partition == "apriori")
                res["front_partitions"] = sorted(set(expl.selection.front["partition"].tolist()))
        except Exception as exc:  # record the failure and continue with the other methods
            traceback.print_exc()
            res = {"method": name, "error": f"{type(exc).__name__}: {exc}"}
        res.update(common)
        if name in protocol.PFCA_VARIANTS and "explanation" in getattr(out, "extra", {}):
            res["pfca_wall_seconds"] = out.wall_seconds + b_pool.wall_seconds
        rows.append({**flatten_row(res), "per_instance": res.get("per_instance")})

    # per construct reliability table and example outputs of the full method
    table = None
    if "pfca" in outputs:
        try:
            expl = outputs["pfca"].extra["explanation"]
            table = reliability_table(expl, attr, U_sub, group_names, ds.feature_names)
            table.insert(0, "repeat", int(r))
            table.insert(1, "seed", seed)
            table.to_csv(out_dir / f"reliability_repeat{r:02d}.csv", index=False)
            share = subscale_overlap(expl.partition.U, U_sub)
            written = save_examples(expl, attr, ds.feature_names, share, group_names, cfg.get("example_instances", []), out_dir, int(r))
            print(f"[phase C] repeat {r}: wrote reliability_repeat{r:02d}.csv and {len(written)} example files", flush=True)
        except Exception:
            traceback.print_exc()
    return rows, table


# ----------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------


def write_summaries(out_dir: Path) -> None:
    """Aggregate the metric and reliability files over repetitions."""
    metrics_path = out_dir / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        skip = {"repeat", "seed", "n_features", "n_subscales", "n_train", "n_tune", "n_test", "n_explain", "n_replicate"}
        cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]
        if cols and "method" in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                summarize_by(df, ["method"], cols).to_csv(out_dir / "summary_by_method.csv")
    rel_path = out_dir / "reliability.csv"
    if rel_path.exists():
        rel = pd.read_csv(rel_path)
        cols = [c for c in ("mean_support_width", "mean_core_width", "relative_support_width", "relative_core_width", "mean_sign_confidence", "mean_disagreement", "fraction_type2", "matched_overlap") if c in rel.columns]
        if cols:
            summary = rel.groupby(["method", "level", "matched_subscale"])[cols].agg(["mean", "std", "count"])
            summary.to_csv(out_dir / "reliability_summary.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/configs/phase_c.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--name", default="phase_c")
    ap.add_argument("--csv", default=None, help="override csv_path of the configuration")
    ap.add_argument("--spec", default=None, help="override spec_path of the configuration")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None, help="override the number of repetitions of the split")
    ap.add_argument("--max-repeats", type=int, default=None, help="stop after this many repetitions (for testing)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"n_jobs": args.n_jobs, "n_repeats": args.repeats, "csv_path": args.csv, "spec_path": args.spec})
    csv_path, spec_path = resolve_path(cfg["csv_path"]), resolve_path(cfg["spec_path"])
    if not csv_path.exists() or not spec_path.exists():
        raise SystemExit(f"Questionnaire files not found: {csv_path} / {spec_path}. The real dataset is not distributed; see data/README.md, or run experiments/make_demo_questionnaire.py for a simulated file.")
    ds = load_questionnaire(str(csv_path), str(spec_path))
    out_dir = prepare_run(args.results, args.name, cfg)
    writer = ResultWriter(out_dir / "metrics.csv", ["repeat", "method"])
    per_instance_writer = PerInstanceWriter(out_dir / "per_instance.csv")
    rel_writer = ResultWriter(out_dir / "reliability.csv", RELIABILITY_KEYS)
    n_repeats = int(cfg["n_repeats"])
    splits = list(stratified_splits(ds.y, ds.task, n_repeats=n_repeats, random_state=int(cfg.get("split_seed", 0))))
    if args.max_repeats:
        splits = splits[: args.max_repeats]
    print(f"[phase C] dataset {ds.name}: {ds.X.shape[0]} rows, {ds.n_features} features, task {ds.task}, subscales {list(ds.groups.keys())}")
    print(f"[phase C] {len(splits)} repetitions, base seed {cfg.get('seed', 0)}, split seed {cfg.get('split_seed', 0)}, results in {out_dir}")
    t0 = time.time()
    for i, split in enumerate(splits):
        r = split[0]
        if all(writer.is_done({"repeat": r, "method": m}) for m in cfg["methods"]):
            continue
        t = time.time()
        next_train = splits[(i + 1) % len(splits)][1] if len(splits) > 1 else None  # a single repetition has no independent replicate
        rows, table = run_repeat(ds, split, next_train, cfg, out_dir)
        per_instance_writer.append_from_rows(rows, ["repeat"])
        writer.append(rows)
        if table is not None:
            rel_writer.append([row for row in table.to_dict(orient="records") if not rel_writer.is_done(row)])
        print(f"[phase C] repeat {i + 1}/{len(splits)} done in {time.time() - t:.1f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    write_summaries(out_dir)
    print("[phase C] finished")


if __name__ == "__main__":
    main()
