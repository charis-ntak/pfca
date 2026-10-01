"""Tables and figures of Table 2 of the study design guide (Section 9), generated from the saved metric files.

Every item is built from the files written by the experiment runners, never
by hand (Section 8.2 of the guide). An item whose inputs are missing is
skipped with a printed note, so the script can be run at any stage of the
study. Figures use the palette and style of pfca.plotting with method colors
assigned in a fixed order, one figure per item, saved as PNG and PDF under
<out>/figures; tables are written as CSV and Markdown under <out>/tables.

Items of Table 2
----------------
 1  properties satisfied by PFCA, grouped SHAP, bootstrapped SHAP and feature
    level SHAP (Section 5)                          tables/properties.csv, tables/properties.md
 2  schematic of the pipeline (Section 3)           figures/pipeline_schematic
 3  attribution recovery error and rank stability against within block
    correlation by method, faceted by family, mean and 95 percent confidence
    band over seeds (Phase A)                       figures/phase_a_recovery_stability, tables/phase_a_recovery_stability.csv
 4  duplication experiment, redundancy family      figures/phase_a_duplication, tables/phase_a_duplication.csv
 5  faithfulness and stability on the Phase B datasets with Friedman average
    ranks, Nemenyi critical difference and Wilcoxon tests with Holm correction
    against feature level SHAP                      tables/phase_b_metrics.csv, phase_b_friedman.csv,
                                                    phase_b_friedman_pairwise.csv, phase_b_wilcoxon_holm.csv, phase_b_table.md
 6  reliability diagrams of the sign confidence (bins pooled over rows
    weighted by count) and interval coverage against the nominal levels with
    the tolerance band of Section 7.6               figures/reliability_sign_confidence, figures/interval_coverage,
                                                    tables/reliability_bins.csv, tables/interval_coverage.csv
 7  example Pareto front with three explanations, produced by
    experiments/example_front.py                    figures/example_front (checked, not redrawn)
 8  runtime and memory against d (Phase A) and against B and M
    (sensitivity)                                   tables/runtime_memory.csv, tables/runtime_memory.md
 9  sensitivity analysis summary                    tables/sensitivity_summary.csv, tables/sensitivity_summary.md
10  Phase C figure, produced by run_phase_c.py      figures/phase_c_example_linguistic_bars.png, figures/phase_c_subscale_overlap.png (copied)

Items beyond Table 2 that the guide requires elsewhere
11  linear mixed models of the Phase A metrics with method, sample size,
    dimension and correlation as fixed effects and the seed as random
    intercept, per family (Section 7.4)             tables/phase_a_mixed_models.csv, tables/phase_a_mixed_models.md
12  the pre registered success criteria of Section 7.6 evaluated against
    feature level SHAP, every failure reported     tables/success_criteria.csv, tables/success_criteria.md
13  Pareto front diagnostics: size of the front, collapse to one solution
    and conflict between the objectives (Section 10) tables/pareto_front_diagnostics.csv, tables/pareto_front_diagnostics.md

The list of produced and skipped items is written to <results>/<run name>/items.csv
next to the frozen configuration and environment.

Usage
-----
python experiments/make_figures.py --results results --out .
python experiments/make_figures.py --results results --phase-a phase_a_smoke --phase-b phase_b_smoke --sensitivity sensitivity_smoke --out .
python experiments/make_figures.py --results results --phase-c phase_c --stability-column rank_stability_blocks --out paper
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ResultWriter, prepare_run  # noqa: E402

from pfca import plotting  # noqa: E402
from pfca.evaluation.statistics import friedman_nemenyi, mixed_model, paired_effect_sizes, summarize_by, wilcoxon_holm  # noqa: E402
from pfca.plotting import GRID, MUTED, PALETTE, TEXT, plot_reliability, style_axes  # noqa: E402

plt = plotting._plt()

# Fixed method order: the palette color of a method never changes with the set of methods shown.
METHOD_ORDER = ["pfca", "shap", "bootstrapped_shap", "grouped_shap", "pfca_crisp", "pfca_single_model", "pfca_fixed", "pfca_apriori", "integrated_gradients", "lime"]
METHOD_LABELS = {
    "pfca": "PFCA",
    "shap": "Feature level SHAP",
    "bootstrapped_shap": "Bootstrapped SHAP",
    "grouped_shap": "Grouped SHAP",
    "pfca_crisp": "PFCA, crisp concepts",
    "pfca_single_model": "PFCA, single model class",
    "pfca_fixed": "PFCA, fixed configuration",
    "pfca_apriori": "PFCA, a priori partition",
    "integrated_gradients": "Integrated gradients",
    "lime": "LIME",
}
ABLATIONS = {"pfca_crisp", "pfca_single_model", "pfca_fixed", "pfca_apriori"}
FAMILY_ORDER = ["additive", "interaction", "redundancy"]
FACTOR_ORDER = ["resamples", "model_classes", "shape", "percentiles", "distance", "solver", "knee", "labels"]
PHASE_A_KEYS = ["family", "n_samples", "n_features", "rho", "seed"]
MEDIUM_EFFECT = 0.5  # Cohen's d of the paired differences at which an effect counts as medium (Section 7.6)
MIN_DATASETS = 6  # datasets required by Section 6.2 for paired tests across datasets to be meaningful
REFERENCE_METHOD = "shap"
# column, label, higher is better
PHASE_B_METRICS = [
    ("deletion_auc", "deletion AUC", False),
    ("insertion_auc", "insertion AUC", True),
    ("surrogate_r2", "surrogate R2", True),
    ("rank_stability", "rank stability", True),
    ("perturbation_stability", "perturbation stability", False),
]
NOMINAL = {"support": 0.90, "core": 0.50}
TOLERANCE = 0.05
EFFECT_SIZE_RANDOM_STATE = 0  # bootstrap confidence intervals of the effect sizes (pfca.evaluation.statistics.paired_effect_sizes)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def note(text: str) -> None:
    print(f"[make figures] {text}", flush=True)


def method_color(method: str) -> str:
    if method in METHOD_ORDER and METHOD_ORDER.index(method) < len(PALETTE):
        return PALETTE[METHOD_ORDER.index(method)]
    return MUTED


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, str(method))


def method_linestyle(method: str):
    return (0, (4, 2)) if method in ABLATIONS else "-"


def ordered_methods(present) -> list[str]:
    present = set(present)
    return [m for m in METHOD_ORDER if m in present] + sorted(m for m in present if m not in METHOD_ORDER)


def ordered_values(values, order: list) -> list:
    values = list(dict.fromkeys(values))
    return [v for v in order if v in values] + [v for v in values if v not in order]


def _read_ragged(path: Path, label: str) -> pd.DataFrame:
    """Read a metrics file whose later rows carry more fields than the header.

    The runners append the rows of every cell in one block; a block that
    carries an additional metric (for example duplication_change in the
    redundancy family) has more fields than the header written by the first
    block. The additional fields are read from the end of the row: numeric
    values are taken as duplication_change and text values as
    duplication_error when the header lacks them, the rest are left unnamed.
    """
    with open(path, newline="", encoding="utf8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    width = max(len(r) for r in rows)
    n_extra = width - len(header)
    body = [r + [""] * (width - len(r)) for r in rows[1:]]
    df = pd.DataFrame(body, columns=header + [f"unnamed_{i}" for i in range(n_extra)])
    df = df.replace({"": np.nan})
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            pass
    renamed = {}
    for i in range(n_extra):
        c = f"unnamed_{i}"
        numeric = pd.api.types.is_numeric_dtype(df[c])
        if numeric and "duplication_change" not in df.columns:
            renamed[c] = "duplication_change"
        elif not numeric and "duplication_error" not in df.columns:
            renamed[c] = "duplication_error"
    df = df.rename(columns=renamed)
    note(f"{label}: {sum(len(r) > len(header) for r in rows[1:])} rows have {n_extra} more fields than the header; the extra fields were read as {list(renamed.values()) or 'unnamed columns'}")
    return df


def read_metrics(path: Path, label: str, keys: list[str] | None = None) -> pd.DataFrame | None:
    """Read a metrics file, tolerating trailing extra fields, drop rows that recorded an error and keep the last row per key.

    ``keys`` are the columns that identify a unit of the design (for example
    family, cell, seed and method); a resumed run may have appended a unit
    twice and only its last row is kept.
    """
    if not path.exists():
        note(f"{label}: {path} not found")
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = _read_ragged(path, label)
    if "error" in df.columns:
        bad = df["error"].notna()
        if bad.any():
            note(f"{label}: {int(bad.sum())} rows with a recorded error are ignored")
            df = df[~bad].copy()
    if keys and all(k in df.columns for k in keys):
        dup = df.duplicated(subset=keys, keep="last")
        if dup.any():
            note(f"{label}: {int(dup.sum())} rows repeat an earlier unit of the design ({', '.join(keys)}); the last row of every unit is kept")
            df = df[~dup].copy()
    if df.empty:
        note(f"{label}: {path} holds no usable rows")
        return None
    n_methods = int(df["method"].nunique()) if "method" in df.columns else 0
    note(f"{label}: {len(df)} rows, {n_methods} method{'s' if n_methods != 1 else ''} from {path}")
    return df


def replication_summary(df: pd.DataFrame, by: list[str], metrics: list[str], replication: str) -> pd.DataFrame:
    """Mean per replication (seed, dataset) within every group, then mean, sd, count and 95 percent confidence interval over replications."""
    metrics = [m for m in metrics if m in df.columns]
    per = df.groupby(by + [replication], sort=False, dropna=False)[metrics].mean().reset_index()
    out = summarize_by(per, by, metrics)
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def save_figure(fig, fig_dir: Path, stem: str) -> list[Path]:
    paths = []
    for ext in ("png", "pdf"):
        p = fig_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.3f}", int_cols=(), col_formats: dict | None = None) -> str:
    """Markdown table; floats use floatfmt, columns in int_cols print as integers, col_formats overrides per column."""
    col_formats = col_formats or {}
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c, v in zip(df.columns, row):
            if v is None or (isinstance(v, float) and not np.isfinite(v)) or (isinstance(v, np.floating) and not np.isfinite(v)):
                cells.append("")
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            elif c in int_cols or isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(col_formats.get(c, floatfmt).format(float(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_text(path: Path, text: str) -> Path:
    with open(path, "w", encoding="utf8") as fh:
        fh.write(text.rstrip() + "\n")
    return path


def shared_legend(fig, axes, ncol: int = 4, bottom: float = 0.0) -> None:
    handles: dict[str, object] = {}
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    if handles:
        fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=min(ncol, len(handles)), bbox_to_anchor=(0.5, bottom), fontsize=8)


# ----------------------------------------------------------------------
# item 1: properties table
# ----------------------------------------------------------------------

# Traits of the compared methods from which the property entries follow.
METHOD_TRAITS = {
    "pfca": {"level": "concept", "rows_sum_to_one": True, "reduces_to_shap": True, "shapley_based": True, "absorbs_duplicates": True, "resample_pool": True, "model_classes": True, "pareto": True},
    "grouped_shap": {"level": "concept", "rows_sum_to_one": True, "reduces_to_shap": True, "shapley_based": True, "absorbs_duplicates": True, "resample_pool": False, "model_classes": False, "pareto": False},
    "bootstrapped_shap": {"level": "feature", "rows_sum_to_one": True, "reduces_to_shap": True, "shapley_based": True, "absorbs_duplicates": False, "resample_pool": True, "model_classes": False, "pareto": False},
    "shap": {"level": "feature", "rows_sum_to_one": True, "reduces_to_shap": True, "shapley_based": True, "absorbs_duplicates": False, "resample_pool": False, "model_classes": False, "pareto": False},
}
PROPERTY_METHODS = ["pfca", "grouped_shap", "bootstrapped_shap", "shap"]
PROPERTIES = [
    (
        "efficiency at concept level",
        "the concept attributions sum to the difference between the prediction and the expected prediction for every membership matrix whose rows sum to one (Equation 1); a feature level method defines no concept attribution and satisfies efficiency at the feature level only",
        lambda t: t["level"] == "concept" and t["rows_sum_to_one"],
    ),
    (
        "reduction to SHAP",
        "with one crisp concept per feature (K equal to d and the identity membership matrix), one resample and one model class the method returns the SHAP values exactly",
        lambda t: t["reduces_to_shap"],
    ),
    (
        "symmetry",
        "two concepts, or two features, with identical aggregated contributions in every coalition receive identical attributions, inherited from the symmetry axiom of the Shapley value applied to every member of the pool",
        lambda t: t["shapley_based"],
    ),
    (
        "duplication invariance",
        "adding an exact copy of a feature leaves the attribution of the concept that absorbs the copy unchanged, whereas the feature level attribution of the copied feature is split between the copies",
        lambda t: t["absorbs_duplicates"],
    ),
    (
        "uncertainty representation",
        "the attribution is reported with its variability over resamples (a fuzzy number with sign confidence, or a confidence interval) instead of as a point estimate",
        lambda t: t["resample_pool"],
    ),
    (
        "model dependence quantified",
        "the variation of the attribution across model classes is measured (disagreement index) and represented (interval type 2 upgrade)",
        lambda t: t["model_classes"],
    ),
    (
        "Pareto selection",
        "the reported explanation is selected from a Pareto front trading off fidelity loss, complexity and instability (Equation 4), with a knee point default and a selection based membership",
        lambda t: t["pareto"],
    ),
]


def item_properties(tab_dir: Path) -> list[Path]:
    rows = []
    for name, definition, rule in PROPERTIES:
        row = {"property": name}
        for m in PROPERTY_METHODS:
            row[method_label(m)] = "yes" if rule(METHOD_TRAITS[m]) else "no"
        row["definition"] = definition
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = tab_dir / "properties.csv"
    df.to_csv(csv_path, index=False)
    md = ["## Properties satisfied by the compared methods (Section 5 of the guide)", ""]
    md.append(markdown_table(df.drop(columns=["definition"])))
    md += ["", "Definitions", ""]
    for name, definition, _ in PROPERTIES:
        md.append(f"1. {name}: {definition}.")
    md_path = write_text(tab_dir / "properties.md", "\n".join(md))
    return [csv_path, md_path]


# ----------------------------------------------------------------------
# item 2: pipeline schematic
# ----------------------------------------------------------------------

SCHEMATIC_BOXES = [
    ("Black box model", "reference model f fitted\non the training data X, y\n(Step 1)"),
    ("Bootstrap by\nmodel class pool", "B resamples x M model\nclasses, refit and Shapley\nvalues per member (Step 2)"),
    ("Fuzzy concepts", "fuzzy c means on feature\ncorrelation profiles,\nmembership U (Step 3)"),
    ("Fuzzy attributions", "aggregation by U (Eq. 1),\ntrapezoids (Eq. 2), sign\nconfidence, labels (Eq. 3)"),
    ("Pareto front", "fidelity loss, complexity,\ninstability (Eq. 4) over K,\nfuzzifier, sparsity, alpha;\nknee point"),
    ("Explanation", "linguistic concept\nattributions, selection\nmembership, feature vector"),
]


def item_schematic(fig_dir: Path) -> list[Path]:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    n = len(SCHEMATIC_BOXES)
    fig, ax = plt.subplots(figsize=(14.0, 3.0))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, gap = 0.82, 0.18
    for i, (title, body) in enumerate(SCHEMATIC_BOXES):
        x0 = i + gap / 2
        color = PALETTE[0] if i in (2, 3, 4) else MUTED
        ax.add_patch(FancyBboxPatch((x0, 0.14), w, 0.74, boxstyle="round,pad=0.01,rounding_size=0.04", facecolor="white", edgecolor=color, linewidth=1.4))
        ax.text(x0 + w / 2, 0.76, title, ha="center", va="center", fontsize=9, color=TEXT, linespacing=1.2)
        ax.text(x0 + w / 2, 0.40, body, ha="center", va="center", fontsize=6.9, color=MUTED, linespacing=1.45)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x0 + w + 0.015, 0.5), (x0 + w + gap - 0.015, 0.5), arrowstyle="-|>", mutation_scale=10, color=MUTED, linewidth=1.0))
    ax.text(gap / 2, 0.03, "Steps refer to the algorithm of Section 4. PFCA specific steps are framed in color; the reference model and the pool are shared with the baselines.", ha="left", va="center", fontsize=7, color=MUTED)
    return save_figure(fig, fig_dir, "pipeline_schematic")


# ----------------------------------------------------------------------
# item 3: recovery error and rank stability against rho (Phase A)
# ----------------------------------------------------------------------


def item_recovery_stability(dfa: pd.DataFrame, fig_dir: Path, tab_dir: Path, stability_col: str) -> list[Path]:
    need = ["family", "rho", "method", "seed", "recovery_error", stability_col]
    missing = [c for c in need if c not in dfa.columns]
    if missing:
        raise ValueError(f"columns {missing} are missing from the Phase A metrics")
    metrics = ["recovery_error", stability_col]
    summ = replication_summary(dfa, ["family", "method", "rho"], metrics, "seed")
    csv_path = tab_dir / "phase_a_recovery_stability.csv"
    summ.to_csv(csv_path, index=False)
    families = ordered_values(summ["family"], FAMILY_ORDER)
    methods = ordered_methods(summ["method"])
    rhos = sorted(summ["rho"].dropna().unique())
    panels = [("recovery_error", "attribution recovery error\n(lower is better)"), (stability_col, stability_col.replace("_", " ") + "\n(higher is better)")]
    fig, axes = plt.subplots(2, len(families), figsize=(3.6 * len(families) + 0.6, 6.0), sharex=True, squeeze=False)
    for j, fam in enumerate(families):
        for i, (metric, ylabel) in enumerate(panels):
            ax = axes[i, j]
            for m in methods:
                s = summ[(summ["family"] == fam) & (summ["method"] == m)].sort_values("rho")
                if s.empty or s[f"{metric}_mean"].isna().all():
                    continue
                x = s["rho"].to_numpy(float)
                y = s[f"{metric}_mean"].to_numpy(float)
                lo, hi = s[f"{metric}_ci_low"].to_numpy(float), s[f"{metric}_ci_high"].to_numpy(float)
                ok = np.isfinite(lo) & np.isfinite(hi)
                if ok.sum() >= 2:
                    ax.fill_between(x[ok], lo[ok], hi[ok], color=method_color(m), alpha=0.15, linewidth=0)
                ax.plot(x, y, color=method_color(m), linestyle=method_linestyle(m), linewidth=1.6, marker="o", markersize=3.5, label=method_label(m))
            ax.set_xticks(rhos)
            if i == 0:
                ax.set_title(f"{fam} family", loc="left")
            if i == len(panels) - 1:
                ax.set_xlabel("within block correlation")
            if j == 0:
                ax.set_ylabel(ylabel)
            style_axes(ax)
    shared_legend(fig, axes, ncol=4, bottom=-0.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save_figure(fig, fig_dir, "phase_a_recovery_stability") + [csv_path]


# ----------------------------------------------------------------------
# item 4: duplication experiment
# ----------------------------------------------------------------------


def item_duplication(dfa: pd.DataFrame, fig_dir: Path, tab_dir: Path) -> list[Path]:
    if "duplication_change" not in dfa.columns:
        raise ValueError("column duplication_change is missing from the Phase A metrics")
    red = dfa[(dfa["family"] == "redundancy") & dfa["duplication_change"].notna()]
    if red.empty:
        raise ValueError("no redundancy rows with a duplication_change value")
    summ = replication_summary(red, ["method"], ["duplication_change"], "seed")
    methods = ordered_methods(summ["method"])
    summ = summ.set_index("method").loc[methods].reset_index()
    csv_path = tab_dir / "phase_a_duplication.csv"
    summ.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(methods) + 1.5))
    y = np.arange(len(methods))[::-1]
    mean = summ["duplication_change_mean"].to_numpy(float)
    lo, hi = summ["duplication_change_ci_low"].to_numpy(float), summ["duplication_change_ci_high"].to_numpy(float)
    ax.barh(y, mean, color=[method_color(m) for m in methods], height=0.55)
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.errorbar(mean[ok], y[ok], xerr=[mean[ok] - lo[ok], hi[ok] - mean[ok]], fmt="none", ecolor=TEXT, elinewidth=1, capsize=3)
    for yi, v, h in zip(y, mean, hi):
        ax.text((h if np.isfinite(h) else v) + 0.01, yi, f"{v:.3f}", va="center", ha="left", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([method_label(m) for m in methods])
    n_seeds = int(summ["duplication_change_count"].max())
    ax.set_xlabel(f"relative change of the attribution absorbing the duplicated feature (lower is better)\nmean and 95 percent confidence interval over {n_seeds} seed{'s' if n_seeds != 1 else ''}", fontsize=8.5)
    ax.set_xlim(0, max(1.0, float(np.nanmax(np.where(np.isfinite(hi), hi, mean))) * 1.2))
    ax.set_title("Duplication experiment, redundancy family", loc="left", fontsize=9)
    style_axes(ax)
    return save_figure(fig, fig_dir, "phase_a_duplication") + [csv_path]


# ----------------------------------------------------------------------
# item 5: Phase B table
# ----------------------------------------------------------------------


def item_phase_b(dfb: pd.DataFrame, tab_dir: Path) -> list[Path]:
    metrics = [(c, l, h) for c, l, h in PHASE_B_METRICS if c in dfb.columns and dfb[c].notna().any()]
    if not metrics or "dataset" not in dfb.columns:
        raise ValueError("no faithfulness or stability metric is present in the Phase B metrics")
    cols = [c for c, _, _ in metrics]
    g = dfb.groupby(["dataset", "method"], sort=False)[cols]
    long = g.agg(["mean", "std", "count"])
    long.columns = [f"{m}_{s}" for m, s in long.columns]
    long = long.reset_index()
    paths = [tab_dir / "phase_b_metrics.csv"]
    long.to_csv(paths[0], index=False)
    means = g.mean().reset_index()
    datasets = list(dict.fromkeys(means["dataset"]))
    n_repeats = int(dfb.groupby(["dataset", "method"]).size().max())
    md = ["## Faithfulness and stability on the Phase B datasets", "", f"Mean over the repetitions of the split (up to {n_repeats}) per dataset and method. Average ranks come from the Friedman procedure across the {len(datasets)} dataset{'s' if len(datasets) != 1 else ''}, with the Nemenyi critical difference at the 5 percent level; two methods differ when their average ranks differ by more than the critical difference. The last column is the Holm adjusted p value of the two sided Wilcoxon signed rank test of every method against feature level SHAP across datasets. Methods that were not evaluated on every dataset are shown but excluded from the tests.", ""]
    friedman_rows, pair_rows, wilcoxon_rows = [], [], []
    for col, label, higher in metrics:
        wide = means.pivot(index="dataset", columns="method", values=col).reindex(index=datasets)
        wide = wide[ordered_methods(wide.columns)]
        complete = wide.dropna(axis=1, how="any")
        dropped = [m for m in wide.columns if m not in complete.columns]
        ranks, cd, fried = None, np.nan, {"friedman_statistic": np.nan, "p_value": np.nan}
        if complete.shape[1] >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    fried = friedman_nemenyi(complete, higher_is_better=higher)
                ranks, cd = fried["average_ranks"], float(fried["critical_difference"])
                sig = fried["significant"]
                for a in complete.columns:
                    for b in complete.columns:
                        if a < b:
                            pair_rows.append({"metric": col, "method_a": a, "method_b": b, "rank_difference": float(abs(ranks[a] - ranks[b])), "critical_difference": cd, "significant": bool(sig.loc[a, b])})
            except Exception as exc:
                note(f"Phase B {label}: Friedman test not computed ({type(exc).__name__}: {exc})")
        else:
            note(f"Phase B {label}: fewer than two methods evaluated on every dataset, tests skipped")
        friedman_rows.append({"metric": col, "higher_is_better": higher, "n_datasets": int(complete.shape[0]), "n_methods": int(complete.shape[1]), "friedman_statistic": fried.get("friedman_statistic", np.nan), "p_value": fried.get("p_value", np.nan), "critical_difference": cd, **{f"rank_{m}": (float(ranks[m]) if ranks is not None and m in ranks.index else np.nan) for m in wide.columns}})
        holm = {}
        if REFERENCE_METHOD in complete.columns and complete.shape[1] >= 2 and complete.shape[0] >= 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                w = wilcoxon_holm(complete, REFERENCE_METHOD, random_state=EFFECT_SIZE_RANDOM_STATE)
            for _, r in w.iterrows():
                rec = {"metric": col, "higher_is_better": higher}
                rec.update({k: (json.dumps([float(x) for x in v]) if isinstance(v, tuple) else v) for k, v in r.items()})
                wilcoxon_rows.append(rec)
                holm[r["method"]] = float(r["p_holm"])
        elif REFERENCE_METHOD not in complete.columns:
            note(f"Phase B {label}: feature level SHAP is not evaluated on every dataset, Wilcoxon tests skipped")
        table = wide.T.copy()
        table.columns = [str(c) for c in table.columns]
        table.insert(0, "method", [method_label(m) for m in table.index])
        table["average rank"] = [float(ranks[m]) if ranks is not None and m in ranks.index else np.nan for m in wide.columns]
        table["p (Holm) vs feature level SHAP"] = [holm.get(m, np.nan) if m != REFERENCE_METHOD else np.nan for m in wide.columns]
        md.append(f"### {label} ({'higher' if higher else 'lower'} is better)")
        md.append("")
        md.append(markdown_table(table.reset_index(drop=True)))
        stat_line = f"Friedman statistic {fried['friedman_statistic']:.3f}, p value {fried['p_value']:.4f}, " if np.isfinite(fried.get("friedman_statistic", np.nan)) else "Friedman test not available with these inputs, "
        md.append("")
        md.append(stat_line + (f"Nemenyi critical difference {cd:.3f} for {complete.shape[1]} methods on {complete.shape[0]} dataset{'s' if complete.shape[0] != 1 else ''}." if np.isfinite(cd) else "critical difference not available."))
        if dropped:
            md.append(f"Excluded from the tests because not evaluated on every dataset: {', '.join(method_label(m) for m in dropped)}.")
        if complete.shape[0] < 6:
            md.append("Fewer than the six datasets required by Section 6.2 are available, so the tests are reported for completeness only.")
        md.append("")
    paths.append(tab_dir / "phase_b_friedman.csv")
    pd.DataFrame(friedman_rows).to_csv(paths[-1], index=False)
    paths.append(tab_dir / "phase_b_friedman_pairwise.csv")
    pd.DataFrame(pair_rows, columns=["metric", "method_a", "method_b", "rank_difference", "critical_difference", "significant"]).to_csv(paths[-1], index=False)
    paths.append(tab_dir / "phase_b_wilcoxon_holm.csv")
    pd.DataFrame(wilcoxon_rows).to_csv(paths[-1], index=False)
    paths.append(write_text(tab_dir / "phase_b_table.md", "\n".join(md)))
    return paths


# ----------------------------------------------------------------------
# item 6: reliability diagrams and interval coverage
# ----------------------------------------------------------------------


def parse_bin_table(text) -> pd.DataFrame | None:
    """Decode a JSON encoded reliability table (dict of lists, list of records or split orient)."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and {"columns", "data"} <= set(obj):
        df = pd.DataFrame(obj["data"], columns=obj["columns"])
    else:
        df = pd.DataFrame(obj)
    need = {"bin_low", "bin_high", "count", "mean_confidence", "observed"}
    if not need <= set(df.columns):
        return None
    return df[sorted(need)].apply(pd.to_numeric, errors="coerce")


def pool_bins(tables: list[pd.DataFrame]) -> tuple[pd.DataFrame, float]:
    """Pool reliability bins across rows weighted by count; returns the pooled table and its expected calibration error."""
    allb = pd.concat(tables, ignore_index=True)
    allb["bin_low"], allb["bin_high"] = allb["bin_low"].round(6), allb["bin_high"].round(6)
    allb["count"] = allb["count"].fillna(0)
    allb["w_conf"] = allb["count"] * allb["mean_confidence"].fillna(0)
    allb["w_obs"] = allb["count"] * allb["observed"].fillna(0)
    g = allb.groupby(["bin_low", "bin_high"], sort=True)[["count", "w_conf", "w_obs"]].sum().reset_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        g["mean_confidence"] = np.where(g["count"] > 0, g["w_conf"] / g["count"], np.nan)
        g["observed"] = np.where(g["count"] > 0, g["w_obs"] / g["count"], np.nan)
    total = float(g["count"].sum())
    ece = float(np.nansum(g["count"] / total * np.abs(g["mean_confidence"] - g["observed"]))) if total > 0 else float("nan")
    return g[["bin_low", "bin_high", "count", "mean_confidence", "observed"]], ece


def item_reliability(sources: list[tuple[str, pd.DataFrame]], fig_dir: Path, tab_dir: Path) -> list[Path]:
    pooled: dict[tuple[str, str], tuple[pd.DataFrame, float]] = {}
    for phase, df in sources:
        if df is None or "sign_confidence_table" not in df.columns:
            note(f"reliability diagrams: {phase} has no sign_confidence_table column")
            continue
        for m, sub in df.groupby("method", sort=False):
            tables = [t for t in (parse_bin_table(x) for x in sub["sign_confidence_table"]) if t is not None and len(t)]
            if tables:
                pooled[(phase, m)] = pool_bins(tables)
    if not pooled:
        raise ValueError("no row holds a sign_confidence_table")
    phases = list(dict.fromkeys(p for p, _ in pooled))
    methods = ordered_methods({m for _, m in pooled})
    rows = []
    for (phase, m), (t, ece) in pooled.items():
        for _, r in t.iterrows():
            rows.append({"phase": phase, "method": m, **r.to_dict(), "ece_pooled": ece})
    csv_path = tab_dir / "reliability_bins.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    fig, axes = plt.subplots(len(phases), len(methods), figsize=(2.9 * len(methods) + 0.4, 2.9 * len(phases) + 0.4), squeeze=False)
    for i, phase in enumerate(phases):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            if (phase, m) not in pooled:
                ax.axis("off")
                continue
            t, ece = pooled[(phase, m)]
            plot_reliability(t, ax=ax, label=method_label(m))
            ax.lines[-1].set_color(method_color(m))
            ax.lines[-1].set_markersize(4)
            n = int(t["count"].sum())
            ax.text(0.52, 0.97, f"ECE {ece:.3f}, n = {n}", fontsize=7.5, color=MUTED, va="top")
            ax.set_title(f"{phase}: {method_label(m)}", loc="left", fontsize=8.5)
            if j > 0:
                ax.set_ylabel("")
            if i < len(phases) - 1:
                ax.set_xlabel("")
    fig.tight_layout()
    return save_figure(fig, fig_dir, "reliability_sign_confidence") + [csv_path]


def item_coverage(dfa: pd.DataFrame | None, dfb: pd.DataFrame | None, fig_dir: Path, tab_dir: Path) -> list[Path]:
    specs = [
        ("Phase A, true attribution", dfa, "support_coverage_truth", "core_coverage_truth", "seed"),
        ("Phase A, replicated attribution", dfa, "support_coverage_replicate", "core_coverage_replicate", "seed"),
        ("Phase B, replicated attribution", dfb, "support_coverage_replicate", "core_coverage_replicate", "dataset"),
    ]
    summaries = []
    for title, df, sup, core, rep in specs:
        if df is None or sup not in df.columns or core not in df.columns or rep not in df.columns:
            continue
        sub = df.dropna(subset=[sup, core], how="all")
        if sub.empty:
            continue
        s = replication_summary(sub, ["method"], [sup, core], rep)
        s = s.rename(columns={c: c.replace(sup, "support").replace(core, "core") for c in s.columns})
        s.insert(0, "source", title)
        s.insert(1, "replication", rep)
        summaries.append(s)
    if not summaries:
        raise ValueError("no coverage column is present in the Phase A or Phase B metrics")
    summ = pd.concat(summaries, ignore_index=True)
    csv_path = tab_dir / "interval_coverage.csv"
    summ.to_csv(csv_path, index=False)
    sources = list(dict.fromkeys(summ["source"]))
    fig, axes = plt.subplots(2, len(sources), figsize=(3.6 * len(sources) + 0.6, 6.2), squeeze=False, sharey=True)
    for j, src in enumerate(sources):
        s = summ[summ["source"] == src]
        methods = ordered_methods(s["method"])
        s = s.set_index("method").loc[methods]
        x = np.arange(len(methods))
        for i, level in enumerate(("support", "core")):
            ax = axes[i, j]
            nominal = NOMINAL[level]
            ax.axhspan(nominal - TOLERANCE, nominal + TOLERANCE, color=GRID, alpha=0.6, linewidth=0, zorder=0)
            ax.axhline(nominal, color=MUTED, linewidth=0.8, linestyle=":", zorder=1)
            mean = s[f"{level}_mean"].to_numpy(float)
            lo, hi = s[f"{level}_ci_low"].to_numpy(float), s[f"{level}_ci_high"].to_numpy(float)
            ax.bar(x, mean, color=[method_color(m) for m in methods], width=0.6, zorder=2)
            ok = np.isfinite(lo) & np.isfinite(hi)
            if ok.any():
                ax.errorbar(x[ok], mean[ok], yerr=[mean[ok] - lo[ok], hi[ok] - mean[ok]], fmt="none", ecolor=TEXT, elinewidth=1, capsize=3, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([method_label(m) for m in methods], rotation=30, ha="right", fontsize=7)
            ax.set_ylim(0, 1.0)
            if j == 0:
                ax.set_ylabel(f"{level} coverage (nominal {nominal:.2f})")
            if i == 0:
                ax.set_title(src, loc="left", fontsize=9)
            style_axes(ax)
    fig.suptitle("Interval coverage against the nominal level; the band marks the tolerance of five percentage points (Section 7.6)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, fig_dir, "interval_coverage") + [csv_path]


# ----------------------------------------------------------------------
# item 7: example front (produced by example_front.py)
# ----------------------------------------------------------------------


def item_example_front(fig_dir: Path) -> list[Path]:
    paths = [fig_dir / "example_front.png", fig_dir / "example_front.pdf"]
    present = [p for p in paths if p.exists()]
    if not present:
        raise ValueError(f"{paths[0]} not found; run python experiments/example_front.py --out {fig_dir}")
    return present


# ----------------------------------------------------------------------
# item 8: runtime and memory
# ----------------------------------------------------------------------

COST_COLUMNS = ["wall_seconds", "peak_mb", "pool_wall_seconds", "pool_peak_mb", "pfca_wall_seconds", "explain_wall_seconds"]


def item_runtime(dfa: pd.DataFrame | None, dfs: pd.DataFrame | None, tab_dir: Path) -> list[Path]:
    parts = []
    if dfa is not None and {"n_features", "method", "wall_seconds"} <= set(dfa.columns):
        cols = [c for c in COST_COLUMNS if c in dfa.columns]
        g = dfa.groupby(["method", "n_features"], sort=False)[cols].mean().reset_index()
        g.insert(0, "source", "Phase A")
        g.insert(1, "factor", "n_features")
        g.insert(2, "setting", g["n_features"].astype(int).astype(str))
        g["n"] = dfa.groupby(["method", "n_features"], sort=False).size().to_numpy()
        g["method"] = pd.Categorical(g["method"], categories=ordered_methods(g["method"]), ordered=True)
        parts.append(g.sort_values(["method", "n_features"]).drop(columns=["n_features"]).assign(method=lambda d: d["method"].astype(str)))
    if dfs is not None and {"factor", "setting", "wall_seconds"} <= set(dfs.columns):
        sub = dfs[dfs["factor"].isin(["resamples", "model_classes"])]
        if not sub.empty:
            cols = [c for c in COST_COLUMNS if c in sub.columns]
            keys = ["factor", "setting"] + [c for c in ("n_resamples", "n_model_classes") if c in sub.columns]
            g = sub.groupby(keys, sort=False)[cols].mean().reset_index()
            g["n"] = sub.groupby(keys, sort=False).size().to_numpy()
            g.insert(0, "source", "sensitivity")
            g.insert(3, "method", "pfca")
            g["factor"] = g["factor"].map({"resamples": "B", "model_classes": "M"})
            g["setting"] = g["setting"].astype(str).str.lstrip("BM")
            order = {"B": 0, "M": 1}
            g = g.assign(_o=g["factor"].map(order), _s=pd.to_numeric(g["setting"], errors="coerce")).sort_values(["_o", "_s"]).drop(columns=["_o", "_s"])
            parts.append(g)
    if not parts:
        raise ValueError("neither the Phase A nor the sensitivity metrics hold the cost columns")
    out = pd.concat(parts, ignore_index=True)
    front = ["source", "factor", "setting", "method", "n"] + [c for c in ("n_resamples", "n_model_classes") if c in out.columns]
    out = out[front + [c for c in out.columns if c not in front]]
    for c in ("n_resamples", "n_model_classes"):
        if c in out.columns:
            out[c] = out[c].astype("Int64")
    csv_path = tab_dir / "runtime_memory.csv"
    out.to_csv(csv_path, index=False)
    md = ["## Runtime and memory against d, B and M", ""]
    md.append("Mean over cells and seeds. wall_seconds and peak_mb are measured for the explanation step of the method from the shared pool (for feature level SHAP the pool time is divided by the number of pool members, for bootstrapped SHAP by the number of model classes); pool_wall_seconds and pool_peak_mb are the cost of fitting the pool of B resamples by M model classes and computing its Shapley values; pfca_wall_seconds is the sum of the pool and the explanation step. In the sensitivity rows the pool is fitted once at the largest B with every model class and subset for smaller B and M, so pool_wall_seconds refers to that pool. Memory is the peak of the Python heap in megabytes.")
    md.append("")
    show = out.copy()
    show["method"] = show["method"].map(method_label)
    for c in ("n_resamples", "n_model_classes"):
        if c in show.columns:
            show[c] = show[c].astype(object).where(show[c].notna(), None)
    md.append(markdown_table(show, int_cols=("n", "n_resamples", "n_model_classes")))
    return [csv_path, write_text(tab_dir / "runtime_memory.md", "\n".join(md))]


# ----------------------------------------------------------------------
# item 9: sensitivity summary
# ----------------------------------------------------------------------

SENSITIVITY_METRICS = ["recovery_error", "support_coverage_truth", "core_coverage_truth", "rank_stability", "n_front"]
SENSITIVITY_EXTRA = ["quantile_mad_to_largest_b", "sign_confidence_ece", "front_recovery", "hypervolume", "mean_support_width_rel", "label_agreement_with_default", "label_sign_agreement_with_default", "label_entropy", "fraction_negligible", "explain_wall_seconds"]


def item_sensitivity(dfs: pd.DataFrame, tab_dir: Path) -> list[Path]:
    if not {"factor", "setting"} <= set(dfs.columns):
        raise ValueError("columns factor and setting are missing from the sensitivity metrics")
    metrics = [c for c in SENSITIVITY_METRICS if c in dfs.columns]
    extra = [c for c in SENSITIVITY_EXTRA if c in dfs.columns]
    if not metrics:
        raise ValueError("none of the sensitivity metrics is present")
    g = dfs.groupby(["factor", "setting"], sort=False)
    summ = g[metrics + extra].mean().reset_index()
    summ.insert(2, "n", g.size().to_numpy())
    if "cell" in dfs.columns:
        summ.insert(3, "n_cells", g["cell"].nunique().to_numpy())
    summ["_o"] = summ["factor"].map({f: i for i, f in enumerate(FACTOR_ORDER)}).fillna(len(FACTOR_ORDER))
    summ = summ.sort_values(["_o"], kind="stable").drop(columns=["_o"]).reset_index(drop=True)
    csv_path = tab_dir / "sensitivity_summary.csv"
    summ.to_csv(csv_path, index=False)
    md = ["## Sensitivity analysis summary (Section 7.5)", ""]
    md.append("Mean over cells and seeds of the attribution recovery error, the coverage of the true attribution by the fuzzy support and core, the rank stability and the size of the Pareto front for every factor and setting. The default setting appears under every factor. Setting labels: B is the number of resamples, M the number of model classes, s and c the percentile levels of the support and the core.")
    md.append("")
    md.append(markdown_table(summ[["factor", "setting", "n"] + metrics], int_cols=("n",), col_formats={"n_front": "{:.1f}"}))
    if extra:
        md.append("")
        md.append("Additional factor specific quantities, when present: " + ", ".join(extra) + " (see sensitivity_summary.csv).")
    return [csv_path, write_text(tab_dir / "sensitivity_summary.md", "\n".join(md))]


# ----------------------------------------------------------------------
# item 10: Phase C figure
# ----------------------------------------------------------------------


def item_phase_c(run_dir: Path, fig_dir: Path) -> list[Path]:
    if not run_dir.exists():
        raise ValueError(f"{run_dir} not found")
    copies = [("example_repeat*_instance*_linguistic_bars.png", "phase_c_example_linguistic_bars.png"), ("overlap_repeat*.png", "phase_c_subscale_overlap.png")]
    out = []
    for pattern, target in copies:
        found = sorted(run_dir.glob(pattern))
        if not found:
            note(f"Phase C: no file matches {pattern} in {run_dir}")
            continue
        shutil.copyfile(found[0], fig_dir / target)
        note(f"Phase C: copied {found[0].name} to {fig_dir / target}")
        out.append(fig_dir / target)
    if not out:
        raise ValueError(f"no Phase C figure found in {run_dir}")
    return out


# ----------------------------------------------------------------------
# item 11: linear mixed models of Phase A (Section 7.4)
# ----------------------------------------------------------------------

MIXED_MODEL_METRICS = [
    ("recovery_error", "attribution recovery error", False),
    ("rank_stability", "rank stability", True),
    ("deletion_auc", "deletion AUC", False),
    ("insertion_auc", "insertion AUC", True),
    ("surrogate_r2", "surrogate R2", True),
    ("support_coverage_truth", "support coverage of the truth", None),
]


def _parse_term(term: str) -> tuple[str, str]:
    """Factor and level of a patsy term such as C(method, Treatment('shap'))[T.pfca]."""
    if term == "Intercept":
        return "intercept", ""
    level = term[term.index("[T.") + 3 : -1] if "[T." in term else ""
    inside = term[term.index("(") + 1 : term.index(")")] if term.startswith("C(") else term
    return inside.split(",")[0].strip(), level


def item_phase_a_mixed_models(dfa: pd.DataFrame, tab_dir: Path) -> list[Path]:
    import statsmodels.formula.api as smf

    need = {"method", "seed", "family"}
    if not need <= set(dfa.columns):
        raise ValueError(f"the Phase A metrics lack the columns {sorted(need - set(dfa.columns))}")
    metrics = [(c, l, h) for c, l, h in MIXED_MODEL_METRICS if c in dfa.columns and dfa[c].notna().any()]
    if not metrics:
        raise ValueError("none of the mixed model metrics is present in the Phase A metrics")
    families = [f for f in FAMILY_ORDER if f in set(dfa["family"])] + sorted(set(dfa["family"]) - set(FAMILY_ORDER))
    rows, md = [], ["## Linear mixed models of the Phase A metrics (Section 7.4)", ""]
    md.append("One model per family and metric: the metric is regressed on the method, the sample size, the dimension and the within block correlation as fixed effects (categorical), with a random intercept per replication seed; the method contrasts are relative to feature level SHAP. A design factor with a single level in the available data is dropped from the model. When a single seed is available no random effect can be estimated and an ordinary least squares model is fitted instead, which the model column records. Estimates are differences in the metric; a positive estimate means a larger value than the reference method.")
    md.append("")
    for family in families:
        sub = dfa[dfa["family"] == family]
        for metric, label, higher in metrics:
            d = sub[[metric, "method", "seed"] + [c for c in ("n_samples", "n_features", "rho") if c in sub.columns]].dropna(subset=[metric]).copy()
            methods = ordered_methods(d["method"].unique())
            if d.empty or len(methods) < 2:
                note(f"mixed models: {family}, {label}: fewer than two methods with values, skipped")
                continue
            ref = REFERENCE_METHOD if REFERENCE_METHOD in methods else methods[0]
            fixed = [f"C(method, Treatment('{ref}'))"] + [f"C({f})" for f in ("n_samples", "n_features", "rho") if f in d.columns and d[f].nunique() > 1]
            n_seeds = int(d["seed"].nunique())
            base = {"family": family, "metric": metric, "higher_is_better": higher, "reference_method": ref, "n_obs": int(len(d)), "n_seeds": n_seeds}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if n_seeds >= 2:
                        fit = mixed_model(d, metric, fixed, "seed")
                        model = "linear mixed model, random intercept per seed"
                        group_var = float(np.asarray(fit.cov_re)[0, 0])
                        params, bse, tvals, pvals, ci = fit.fe_params, fit.bse_fe, fit.tvalues, fit.pvalues, fit.conf_int()
                    else:
                        fit = smf.ols(f"{metric} ~ " + " + ".join(fixed), d).fit()
                        model = "ordinary least squares (single seed, no random effect)"
                        group_var = float("nan")
                        params, bse, tvals, pvals, ci = fit.params, fit.bse, fit.tvalues, fit.pvalues, fit.conf_int()
            except Exception as exc:
                note(f"mixed models: {family}, {label}: not fitted ({type(exc).__name__}: {exc})")
                rows.append({**base, "model": "not fitted", "error": f"{type(exc).__name__}: {exc}"})
                continue
            for term in params.index:
                factor, level = _parse_term(str(term))
                rows.append(
                    {
                        **base,
                        "model": model,
                        "group_variance": group_var,
                        "factor": factor,
                        "level": level,
                        "term": str(term),
                        "estimate": float(params[term]),
                        "se": float(bse[term]),
                        "statistic": float(tvals[term]),
                        "p_value": float(pvals[term]),
                        "ci_low": float(ci.loc[term].iloc[0]),
                        "ci_high": float(ci.loc[term].iloc[1]),
                    }
                )
            contrasts = pd.DataFrame([r for r in rows if r.get("family") == family and r.get("metric") == metric and r.get("factor") == "method"])
            md.append(f"### {family} family, {label} ({'higher' if higher else 'lower' if higher is not None else 'nominal 0.90'} is better)")
            md.append("")
            md.append(f"{model}; {len(d)} observations, {n_seeds} seed{'s' if n_seeds != 1 else ''}; contrasts relative to {method_label(ref)}.")
            md.append("")
            if len(contrasts):
                show = contrasts[["level", "estimate", "se", "p_value", "ci_low", "ci_high"]].copy()
                show.insert(0, "method", show.pop("level").map(method_label))
                md.append(markdown_table(show))
            md.append("")
    if not rows:
        raise ValueError("no mixed model could be fitted on the Phase A metrics")
    csv_path = tab_dir / "phase_a_mixed_models.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return [csv_path, write_text(tab_dir / "phase_a_mixed_models.md", "\n".join(md))]


# ----------------------------------------------------------------------
# item 12: pre registered success criteria (Section 7.6)
# ----------------------------------------------------------------------

CRITERIA_TEXT = {
    1: "lower attribution recovery error and higher rank stability than feature level SHAP in the redundancy family of Phase A, with at least a medium effect size",
    2: "faithfulness on the Phase B datasets matched or exceeded relative to feature level SHAP",
    3: "calibrated intervals: coverage within five percentage points of the nominal level",
}


def _pfca_vs_reference_phase_a(dfa: pd.DataFrame, metric: str) -> pd.DataFrame | None:
    keys = [k for k in PHASE_A_KEYS if k in dfa.columns]
    red = dfa[dfa["family"] == "redundancy"] if "family" in dfa.columns else dfa
    a = red[red["method"] == "pfca"].set_index(keys)[metric].rename("pfca")
    b = red[red["method"] == REFERENCE_METHOD].set_index(keys)[metric].rename("reference")
    if a.empty or b.empty:
        return None
    return pd.concat([a, b], axis=1, join="inner").dropna()


def item_success_criteria(dfa: pd.DataFrame | None, dfb: pd.DataFrame | None, tab_dir: Path) -> list[Path]:
    from scipy import stats

    if dfa is None and dfb is None:
        raise ValueError("neither the Phase A nor the Phase B metrics are available")
    rows = []
    # criterion 1: redundancy family of Phase A, paired by cell and seed
    for metric, higher, label in [("recovery_error", False, "attribution recovery error"), ("rank_stability", True, "rank stability")]:
        row = {"criterion": 1, "phase": "A, redundancy family", "quantity": metric, "direction": "higher" if higher else "lower", "threshold": f"favorable mean difference and |Cohen d| >= {MEDIUM_EFFECT}", "status": "not evaluated"}
        pair = _pfca_vs_reference_phase_a(dfa, metric) if dfa is not None and metric in dfa.columns and "method" in dfa.columns else None
        if pair is not None and len(pair) >= 2:
            es = paired_effect_sizes(pair["pfca"].to_numpy(), pair["reference"].to_numpy(), random_state=EFFECT_SIZE_RANDOM_STATE)
            favorable = es["mean_difference"] > 0 if higher else es["mean_difference"] < 0
            row.update({"pfca": float(pair["pfca"].mean()), "reference": float(pair["reference"].mean()), "difference": es["mean_difference"], "cohen_d": es["cohen_d"], "cohen_d_ci_low": es["cohen_d_ci"][0], "cohen_d_ci_high": es["cohen_d_ci"][1], "cliff_delta": es["cliff_delta"], "n": int(len(pair)), "status": "met" if favorable and abs(es["cohen_d"]) >= MEDIUM_EFFECT else "not met"})
        elif pair is not None:
            row["status"] = "not evaluated (fewer than two paired cells)"
        rows.append(row)
    # criterion 2: Phase B faithfulness, per dataset means
    for metric, label, higher in PHASE_B_METRICS[:3]:
        row = {"criterion": 2, "phase": "B", "quantity": metric, "direction": "higher" if higher else "lower", "threshold": f"mean over datasets not worse than the reference, or two sided Wilcoxon p >= 0.05 with at least {MIN_DATASETS} datasets", "status": "not evaluated"}
        if dfb is not None and {metric, "dataset", "method"} <= set(dfb.columns):
            means = dfb.groupby(["dataset", "method"])[metric].mean().unstack()
            if {"pfca", REFERENCE_METHOD} <= set(means.columns):
                pair = means[["pfca", REFERENCE_METHOD]].dropna()
                if len(pair):
                    diff = float((pair["pfca"] - pair[REFERENCE_METHOD]).mean())
                    favorable = diff >= 0 if higher else diff <= 0
                    p = float("nan")
                    if len(pair) >= 2 and not np.allclose(pair["pfca"], pair[REFERENCE_METHOD]):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            p = float(stats.wilcoxon(pair["pfca"], pair[REFERENCE_METHOD], alternative="two-sided", zero_method="wilcox").pvalue)
                    matches = np.isfinite(p) and p >= 0.05 and len(pair) >= MIN_DATASETS  # a non significant difference counts as matching only with enough datasets for the test to have power
                    row.update({"pfca": float(pair["pfca"].mean()), "reference": float(pair[REFERENCE_METHOD].mean()), "difference": diff, "p_value": p, "n": int(len(pair)), "status": "met" if favorable or matches else "not met"})
        rows.append(row)
    # criterion 3: coverage against the nominal levels
    for source, df in (("A", dfa), ("B", dfb)):
        if df is None or "method" not in df.columns:
            continue
        for col, nominal in [("support_coverage_truth", NOMINAL["support"]), ("core_coverage_truth", NOMINAL["core"]), ("support_coverage_replicate", NOMINAL["support"]), ("core_coverage_replicate", NOMINAL["core"])]:
            if col not in df.columns:
                continue
            vals = df.loc[df["method"] == "pfca", col].dropna()
            if vals.empty:
                continue
            mean = float(vals.mean())
            rows.append({"criterion": 3, "phase": source, "quantity": col, "direction": "nominal", "threshold": f"|coverage - {nominal:.2f}| <= {TOLERANCE:.2f}", "pfca": mean, "reference": nominal, "difference": mean - nominal, "n": int(len(vals)), "status": "met" if abs(mean - nominal) <= TOLERANCE else "not met"})
    if not any(r["status"] in ("met", "not met") for r in rows):
        raise ValueError("no criterion could be evaluated from the available metrics")
    df = pd.DataFrame(rows)
    evaluated = df["status"].isin(["met", "not met"])
    overall = "met" if evaluated.all() and (df["status"] == "met").all() else ("not met" if (df["status"] == "not met").any() else "not evaluated")
    csv_path = tab_dir / "success_criteria.csv"
    df.to_csv(csv_path, index=False)
    md = ["## Pre registered success criteria (Section 7.6)", ""]
    md.append("Relative to feature level SHAP the method is considered successful if it (1) " + CRITERIA_TEXT[1] + ", (2) " + CRITERIA_TEXT[2] + ", and (3) " + CRITERIA_TEXT[3] + ". Every criterion is evaluated from the saved metric files and failure is reported rather than concealed. Criterion 1 pairs PFCA and SHAP by cell and seed; the effect size is Cohen's d of the paired differences with a bootstrap confidence interval, and Cliff's delta is reported beside it. Criterion 2 compares the per dataset means; a mean not worse than the reference meets it, and so does a non significant two sided Wilcoxon test (p at least 0.05) when at least six datasets are available, the minimum of Section 6.2. Criterion 3 uses the mean coverage of the PFCA rows against the nominal levels of the support (0.90) and the core (0.50).")
    md.append("")
    show = df.copy()
    for c in ("pfca", "reference", "difference", "cohen_d", "cliff_delta", "p_value"):
        if c not in show.columns:
            show[c] = np.nan
    md.append(markdown_table(show[["criterion", "phase", "quantity", "direction", "pfca", "reference", "difference", "cohen_d", "cliff_delta", "p_value", "n", "status"]], int_cols=("criterion", "n")))
    md.append("")
    md.append(f"Overall: {overall}.")
    return [csv_path, write_text(tab_dir / "success_criteria.md", "\n".join(md))]


# ----------------------------------------------------------------------
# item 13: Pareto front diagnostics (Section 10, collapse of the front)
# ----------------------------------------------------------------------

FRONT_COLUMNS = ["n_front", "n_front_distinct", "front_collapsed", "n_feasible", "objective_corr_fidelity_complexity", "objective_corr_fidelity_instability", "objective_corr_complexity_instability"]


def _front_summary(sub: pd.DataFrame, by: str) -> pd.DataFrame:
    cols = [c for c in FRONT_COLUMNS if c in sub.columns]
    d = sub[[by] + cols].copy()
    if "front_collapsed" in d.columns:
        d["front_collapsed"] = pd.to_numeric(d["front_collapsed"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}), errors="coerce")
    g = d.groupby(by, sort=False)
    out = g[cols].mean().reset_index().rename(columns={by: "group", "front_collapsed": "fraction_collapsed"})
    out.insert(1, "n", g.size().to_numpy())
    return out


def item_front_diagnostics(dfa: pd.DataFrame | None, dfs: pd.DataFrame | None, tab_dir: Path) -> list[Path]:
    parts = []
    if dfa is not None and {"method", "n_front", "family"} <= set(dfa.columns):
        sub = dfa[dfa["method"] == "pfca"]
        if not sub.empty:
            fam = _front_summary(sub, "family")
            fam.insert(0, "source", "Phase A")
            allrows = _front_summary(sub.assign(_all="all families"), "_all")
            allrows.insert(0, "source", "Phase A")
            parts += [fam, allrows]
    if dfs is not None and {"factor", "setting", "n_front"} <= set(dfs.columns):
        sub = dfs[dfs["factor"] == "solver"]
        if not sub.empty:
            sol = _front_summary(sub, "setting")
            sol.insert(0, "source", "sensitivity, solver")
            parts.append(sol)
    if not parts:
        raise ValueError("no PFCA rows with the Pareto front size are available")
    out = pd.concat(parts, ignore_index=True)
    csv_path = tab_dir / "pareto_front_diagnostics.csv"
    out.to_csv(csv_path, index=False)
    md = ["## Pareto front diagnostics (Section 10 of the guide)", ""]
    md.append("Mean over cells and seeds of the number of Pareto optimal configurations (n_front), the number of distinct Pareto optimal explanations (one per partition and retained set), the fraction of runs in which the front collapsed to a single distinct explanation, the number of feasible evaluated configurations and the Spearman correlation between every pair of objectives of Equation 4 over the feasible configurations. A negative correlation means that the two objectives conflict; a front that collapses in most runs would mean that the objective set should be reduced, which the guide asks to report as a finding.")
    md.append("")
    md.append(markdown_table(out, int_cols=("n",)))
    return [csv_path, write_text(tab_dir / "pareto_front_diagnostics.md", "\n".join(md))]


# ----------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="directory holding the run directories")
    ap.add_argument("--phase-a", default="phase_a", help="name of the Phase A run directory")
    ap.add_argument("--phase-b", default="phase_b", help="name of the Phase B run directory")
    ap.add_argument("--sensitivity", default="sensitivity", help="name of the sensitivity run directory")
    ap.add_argument("--phase-c", default="phase_c", help="name of the Phase C run directory")
    ap.add_argument("--out", default=".", help="directory that receives figures/ and tables/")
    ap.add_argument("--figures", default=None, help="figure directory, overriding <out>/figures")
    ap.add_argument("--tables", default=None, help="table directory, overriding <out>/tables")
    ap.add_argument("--stability-column", default="rank_stability", choices=["rank_stability", "rank_stability_blocks"], help="stability metric of item 3")
    ap.add_argument("--name", default="make_figures", help="run directory (under --results) that logs the configuration and the produced items")
    args = ap.parse_args(argv)

    cfg = vars(args).copy()
    cfg["effect_size_random_state"] = EFFECT_SIZE_RANDOM_STATE
    run_dir = prepare_run(args.results, args.name, cfg)
    fig_dir = Path(args.figures) if args.figures else Path(args.out) / "figures"
    tab_dir = Path(args.tables) if args.tables else Path(args.out) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    note(f"figures in {fig_dir}, tables in {tab_dir}, log in {run_dir}; the only random component (bootstrap confidence intervals of the effect sizes) uses random state {EFFECT_SIZE_RANDOM_STATE}")
    results = Path(args.results)
    dfa = read_metrics(results / args.phase_a / "metrics.csv", "Phase A", ["family", "n_samples", "n_features", "rho", "seed", "method"])
    dfb = read_metrics(results / args.phase_b / "metrics.csv", "Phase B", ["dataset", "repeat", "method"])
    dfs = read_metrics(results / args.sensitivity / "metrics.csv", "sensitivity", ["cell", "seed", "factor", "setting"])

    def require(df, label):
        if df is None:
            raise ValueError(f"{label} metrics missing")
        return df

    items = [
        (1, "properties table", lambda: item_properties(tab_dir)),
        (2, "pipeline schematic", lambda: item_schematic(fig_dir)),
        (3, "recovery error and rank stability against rho (Phase A)", lambda: item_recovery_stability(require(dfa, "Phase A"), fig_dir, tab_dir, args.stability_column)),
        (4, "duplication experiment (Phase A)", lambda: item_duplication(require(dfa, "Phase A"), fig_dir, tab_dir)),
        (5, "Phase B table with Friedman ranks", lambda: item_phase_b(require(dfb, "Phase B"), tab_dir)),
        (6, "reliability diagrams", lambda: item_reliability([("Phase A", dfa), ("Phase B", dfb)], fig_dir, tab_dir)),
        (6, "interval coverage", lambda: item_coverage(dfa, dfb, fig_dir, tab_dir)),
        (7, "example Pareto front", lambda: item_example_front(fig_dir)),
        (8, "runtime and memory table", lambda: item_runtime(dfa, dfs, tab_dir)),
        (9, "sensitivity summary", lambda: item_sensitivity(require(dfs, "sensitivity"), tab_dir)),
        (10, "Phase C figure", lambda: item_phase_c(results / args.phase_c, fig_dir)),
        (11, "Phase A mixed models (Section 7.4)", lambda: item_phase_a_mixed_models(require(dfa, "Phase A"), tab_dir)),
        (12, "success criteria (Section 7.6)", lambda: item_success_criteria(dfa, dfb, tab_dir)),
        (13, "Pareto front diagnostics (Section 10)", lambda: item_front_diagnostics(dfa, dfs, tab_dir)),
    ]
    log_path = run_dir / "items.csv"
    if log_path.exists():
        log_path.unlink()
    writer = ResultWriter(log_path, ["item", "title"])
    n_done = 0
    t0 = time.time()
    for number, title, fn in items:
        t = time.time()
        try:
            paths = fn()
            status, detail = "written", ", ".join(str(p) for p in paths)
            n_done += 1
            note(f"item {number} ({title}): {detail}")
        except Exception as exc:
            status, detail = "skipped", f"{type(exc).__name__}: {exc}"
            note(f"item {number} ({title}) skipped: {exc}")
        writer.append([{"item": number, "title": title, "status": status, "detail": detail, "seconds": round(time.time() - t, 2)}])
    note(f"{n_done} of {len(items)} items written in {time.time() - t0:.1f}s; log in {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
