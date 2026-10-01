"""Turn the saved metric files of the reduced study into the tables and figures of the manuscript.

Usage: python3 build_results.py --results ../results --phase-a phase_a_reduced --phase-b phase_b_reduced
       --phase-c phase_c_reduced --sensitivity sensitivity_reduced --out .

Runs experiments/make_figures.py into <out>/generated (items of Table 2 of the guide and the three
extra items), then writes compact manuscript tables as CSV and as paper.md table blocks under
<out>/tables, and copies the figures used by the manuscript under <out>/figures.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in [HERE, *HERE.parents] if (p / "experiments" / "make_figures.py").exists())
LABELS = {"pfca": "PFCA", "shap": "SHAP", "bootstrapped_shap": "Bootstrapped SHAP", "grouped_shap": "Grouped SHAP", "lime": "LIME", "pfca_crisp": "PFCA, crisp", "pfca_single_model": "PFCA, single class", "pfca_fixed": "PFCA, fixed", "pfca_apriori": "PFCA, a priori", "integrated_gradients": "Integrated gradients"}
ORDER = ["pfca", "shap", "bootstrapped_shap", "grouped_shap", "lime", "pfca_crisp", "pfca_single_model", "pfca_fixed", "pfca_apriori"]


def fmt(m, s=None, nd=3):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return ""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} ({s:.{nd}f})"


def md_block(label, legend, df, note=None):
    lines = [f"::table {label} | {legend}", " | ".join(str(c) for c in df.columns)]
    for _, r in df.iterrows():
        lines.append(" | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.tolist()))
    if note:
        lines.append(f"Note: {note}")
    lines.append("::end")
    return "\n".join(lines)


def ordered(methods):
    return [m for m in ORDER if m in set(methods)] + sorted(set(methods) - set(ORDER))


def phase_a_tables(dfa: pd.DataFrame, tab_dir: Path, blocks: list):
    metrics = [("recovery_error", "recovery error"), ("rank_recovery", "rank recovery"), ("rank_stability", "rank stability"), ("deletion_auc", "deletion AUC"), ("insertion_auc", "insertion AUC"), ("surrogate_r2", "surrogate R2")]
    rows = []
    for fam in ["additive", "interaction", "redundancy"]:
        sub = dfa[dfa["family"] == fam]
        for m in ordered(sub["method"].unique()):
            s = sub[sub["method"] == m]
            row = {"Family": fam, "Method": LABELS.get(m, m), "n": int(len(s))}
            for col, lab in metrics:
                row[lab] = fmt(s[col].mean(), s[col].std()) if col in s.columns and s[col].notna().any() else ""
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "phase_a_by_family.csv", index=False)
    blocks.append(md_block("tab:phasea", "Correctness, faithfulness and stability per family and method in Phase A, mean (SD) over cells and seeds", df, "n is the number of cells times seeds. Lower is better for the recovery error and the deletion AUC, higher is better for the other metrics. Empty cells mark metrics that do not apply to a method."))
    # coverage table (PFCA variants and bootstrapped SHAP)
    cov_cols = [("support_coverage_truth", "support, truth"), ("core_coverage_truth", "core, truth"), ("support_coverage_replicate", "support, replicate"), ("core_coverage_replicate", "core, replicate"), ("sign_confidence_ece", "sign confidence ECE")]
    rows = []
    for m in ordered(dfa["method"].unique()):
        s = dfa[dfa["method"] == m]
        if not any(s[c].notna().any() for c, _ in cov_cols if c in s.columns):
            continue
        row = {"Method": LABELS.get(m, m)}
        for c, lab in cov_cols:
            row[lab] = fmt(s[c].mean(), s[c].std()) if c in s.columns and s[c].notna().any() else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "phase_a_coverage.csv", index=False)
    blocks.append(md_block("tab:coverage", "Interval coverage and sign confidence calibration in Phase A, mean (SD) over all cells and seeds", df, "Nominal coverage is 0.90 for the support and 0.50 for the core. ECE is the expected calibration error of the sign confidence against the sign agreement with the replicate attribution."))
    # duplication
    red = dfa[dfa["family"] == "redundancy"]
    if "duplication_change" in red.columns:
        rows = [{"Method": LABELS.get(m, m), "duplication change": fmt(red.loc[red.method == m, "duplication_change"].mean(), red.loc[red.method == m, "duplication_change"].std()), "n": int(red.loc[red.method == m, "duplication_change"].notna().sum())} for m in ordered(red["method"].unique()) if red.loc[red.method == m, "duplication_change"].notna().any()]
        df = pd.DataFrame(rows)
        df.to_csv(tab_dir / "phase_a_duplication.csv", index=False)
        blocks.append(md_block("tab:duplication", "Relative change of the attribution when the duplicate features are added, redundancy family of Phase A, mean (SD)", df, "The change is computed between the explanation of the problem without the duplicates and the explanation of the same instances with the duplicates, on the attribution of the item that contains the duplicated feature."))


def mixed_model_table(gen_tab: Path, tab_dir: Path, blocks: list):
    p = gen_tab / "phase_a_mixed_models.csv"
    if not p.exists():
        return
    mm = pd.read_csv(p)
    mm = mm[(mm["factor"] == "method") & mm["metric"].isin(["recovery_error", "rank_stability", "deletion_auc"])]
    rows = []
    for _, r in mm.iterrows():
        rows.append({"Family": r["family"], "Metric": r["metric"].replace("_", " "), "Method": LABELS.get(r["level"], r["level"]), "Estimate": f"{r['estimate']:+.3f}", "SE": f"{r['se']:.3f}", "95% CI": f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]", "p": ("<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}")})
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "phase_a_mixed_models.csv", index=False)
    model = mm["model"].iloc[0] if len(mm) else ""
    blocks.append(md_block("tab:mixed", "Linear mixed model contrasts of every method against feature level SHAP in Phase A, per family and metric", df, f"Model: {model}; sample size, dimension and correlation as categorical fixed effects. A negative estimate means a smaller value than SHAP."))


def criteria_table(gen_tab: Path, tab_dir: Path, blocks: list):
    p = gen_tab / "success_criteria.csv"
    if not p.exists():
        return
    sc = pd.read_csv(p)
    rows = []
    for _, r in sc.iterrows():
        rows.append({"Criterion": int(r["criterion"]), "Phase": r["phase"], "Quantity": str(r["quantity"]).replace("_", " "), "PFCA": fmt(r.get("pfca", np.nan)), "Reference": fmt(r.get("reference", np.nan)), "Effect or p": (fmt(r["cohen_d"]) + " (Cohen d)") if "cohen_d" in sc.columns and pd.notna(r.get("cohen_d", np.nan)) else ((fmt(r["p_value"]) + " (p)") if "p_value" in sc.columns and pd.notna(r.get("p_value", np.nan)) else ""), "Status": r["status"]})
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "success_criteria.csv", index=False)
    blocks.append(md_block("tab:criteria", "Pre registered success criteria evaluated on the reduced study", df, "Criterion 1: favorable paired difference with |Cohen d| of at least 0.5. Criterion 2: per dataset mean not worse than SHAP. Criterion 3: coverage within 0.05 of the nominal 0.90 (support) or 0.50 (core)."))


def phase_b_table(dfb: pd.DataFrame, tab_dir: Path, blocks: list):
    metrics = [("deletion_auc", "deletion AUC"), ("insertion_auc", "insertion AUC"), ("surrogate_r2", "surrogate R2"), ("rank_stability", "rank stability"), ("perturbation_stability", "perturbation stability"), ("support_coverage_replicate", "support coverage"), ("core_coverage_replicate", "core coverage"), ("wall_seconds", "time (s)")]
    rows = []
    for m in ordered(dfb["method"].unique()):
        s = dfb[dfb["method"] == m]
        row = {"Method": LABELS.get(m, m)}
        for c, lab in metrics:
            row[lab] = fmt(s[c].mean(), s[c].std()) if c in s.columns and s[c].notna().any() else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "phase_b.csv", index=False)
    n = int(dfb.groupby("method").size().max())
    blocks.append(md_block("tab:phaseb", f"Faithfulness, stability, calibration and cost on the breast cancer dataset, mean (SD) over {n} repetitions of the split", df, "Surrogate R2 is computed on the retained items at the sparsity of the PFCA knee. Time is the wall clock time of the explanation step after the shared pool."))


def phase_c_tables(results: Path, name: str, tab_dir: Path, blocks: list):
    p = results / name / "metrics.csv"
    if not p.exists():
        return
    dfc = pd.read_csv(p)
    metrics = [("ari_subscales", "ARI vs subscales"), ("deletion_auc", "deletion AUC"), ("insertion_auc", "insertion AUC"), ("surrogate_r2", "surrogate R2"), ("rank_stability", "rank stability"), ("support_coverage_replicate", "support coverage"), ("n_type2", "type 2 attributions")]
    rows = []
    for m in ordered(dfc["method"].unique()):
        s = dfc[dfc["method"] == m]
        row = {"Method": LABELS.get(m, m)}
        for c, lab in metrics:
            row[lab] = fmt(s[c].mean(), s[c].std(), 3 if c != "n_type2" else 1) if c in s.columns and s[c].notna().any() else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "phase_c.csv", index=False)
    n = int(dfc.groupby("method").size().max())
    blocks.append(md_block("tab:phasec", f"Concept recovery, faithfulness, stability and calibration on the simulated questionnaire, mean (SD) over {n} repetitions", df, "ARI vs subscales is the adjusted Rand index between the hardened partition of the method and the a priori subscales; it equals one by construction for the a priori partition and the grouped baseline."))
    rel = results / name / "reliability_summary.csv"
    if rel.exists():
        shutil.copy(rel, tab_dir / "phase_c_reliability_summary.csv")


def sensitivity_table(dfs: pd.DataFrame, tab_dir: Path, blocks: list):
    metrics = [("recovery_error", "recovery error"), ("rank_stability", "rank stability"), ("support_coverage_truth", "support coverage"), ("n_front_distinct", "distinct front size"), ("quantile_mad_to_largest_b", "quantile MAD to largest B"), ("front_recovery", "front recovery"), ("label_sign_agreement_with_default", "label sign agreement")]
    g = dfs.groupby(["factor", "setting"], sort=False)
    rows = []
    for (f, s), sub in g:
        row = {"Factor": f, "Setting": s, "n": int(len(sub))}
        for c, lab in metrics:
            row[lab] = fmt(sub[c].mean(), sub[c].std()) if c in sub.columns and sub[c].notna().any() else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "sensitivity.csv", index=False)
    blocks.append(md_block("tab:sensitivity", "Sensitivity of the explanation to the design choices of the method, mean (SD) over cells and seeds", df, "The default setting appears under every factor. Settings: B is the number of resamples, M the number of model classes, s and c the percentile levels of the support and the core."))


def front_table(gen_tab: Path, tab_dir: Path, blocks: list):
    p = gen_tab / "pareto_front_diagnostics.csv"
    if not p.exists():
        return
    fd = pd.read_csv(p)
    keep = {"source": "Source", "group": "Group", "n": "n", "n_front": "front size", "n_front_distinct": "distinct explanations", "fraction_collapsed": "fraction collapsed", "objective_corr_fidelity_complexity": "corr(fidelity, complexity)", "objective_corr_fidelity_instability": "corr(fidelity, instability)", "objective_corr_complexity_instability": "corr(complexity, instability)"}
    df = fd[[c for c in keep if c in fd.columns]].rename(columns=keep)
    for c in df.columns[3:]:
        df[c] = df[c].map(lambda v: fmt(v, None, 2))
    df.to_csv(tab_dir / "front_diagnostics.csv", index=False)
    blocks.append(md_block("tab:front", "Size of the Pareto front and conflict between the objectives", df, "Spearman correlations between the objectives over the feasible configurations; a negative value means that the two objectives conflict."))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--phase-a", default="phase_a_reduced")
    ap.add_argument("--phase-b", default="phase_b_reduced")
    ap.add_argument("--phase-c", default="phase_c_reduced")
    ap.add_argument("--sensitivity", default="sensitivity_reduced")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args(argv)
    results = Path(args.results)
    out = Path(args.out)
    gen = out / "generated"
    tab_dir, fig_dir = out / "tables", out / "figures"
    for d in (gen, tab_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    # the example front figure of make_figures is looked up in <gen>/figures
    (gen / "figures").mkdir(exist_ok=True)
    for name in ("example_front.png", "example_front.pdf"):
        src = fig_dir / name
        if src.exists():
            shutil.copy(src, gen / "figures" / name)
    cmd = [sys.executable, str(ROOT / "experiments" / "make_figures.py"), "--results", str(results), "--phase-a", args.phase_a, "--phase-b", args.phase_b, "--phase-c", args.phase_c, "--sensitivity", args.sensitivity, "--out", str(gen), "--name", "make_figures_paper"]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    blocks: list[str] = []
    pa = results / args.phase_a / "metrics.csv"
    if pa.exists():
        phase_a_tables(pd.read_csv(pa), tab_dir, blocks)
    mixed_model_table(gen / "tables", tab_dir, blocks)
    pb = results / args.phase_b / "metrics.csv"
    if pb.exists():
        phase_b_table(pd.read_csv(pb), tab_dir, blocks)
    phase_c_tables(results, args.phase_c, tab_dir, blocks)
    ps = results / args.sensitivity / "metrics.csv"
    if ps.exists():
        sensitivity_table(pd.read_csv(ps), tab_dir, blocks)
    front_table(gen / "tables", tab_dir, blocks)
    criteria_table(gen / "tables", tab_dir, blocks)
    (tab_dir / "blocks.md").write_text("\n\n".join(blocks) + "\n", encoding="utf8")
    # figures used by the manuscript
    for name in ("phase_a_recovery_stability.png", "phase_a_duplication.png", "reliability_sign_confidence.png", "interval_coverage.png"):
        src = gen / "figures" / name
        if src.exists():
            shutil.copy(src, fig_dir / name)
    crun = results / args.phase_c
    for pattern, target in (("example_repeat00_instance00_linguistic_bars.png", "phase_c_linguistic_bars.png"), ("example_repeat00_instance00_fuzzy_attribution.png", "phase_c_fuzzy_attribution.png"), ("overlap_repeat00.png", "phase_c_overlap.png"), ("membership_repeat00.png", "phase_c_membership.png")):
        src = crun / pattern
        if src.exists():
            shutil.copy(src, fig_dir / target)
    print(f"wrote {len(blocks)} table blocks to {tab_dir / 'blocks.md'}")


if __name__ == "__main__":
    main()
