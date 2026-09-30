"""Example Pareto front with three explanations taken from the front (Table 2 of the guide, item 7).

A synthetic additive problem (500 samples, 30 features in blocks of correlated
features, within block correlation 0.6) is generated with the Phase A
generator, PFCA is fitted with a moderate pool (20 resamples, gradient
boosting and random forest), 100 fresh instances are explained and the table
of evaluated configurations with the three objectives of Equation 4 is saved.
The figure shows the Pareto front of explanation configurations with the knee
point and, for one explained instance, the linguistic explanation at three
members of the front: the sparsest configuration, the knee point and the most
faithful configuration. The explanations at the other front members are
rebuilt with PFCAExplainer.explanation_at from the same pool, so the three
panels differ only in the configuration selected from the front.

Usage
-----
python experiments/example_front.py --out figures
python experiments/example_front.py --n-resamples 4 --n-explain 30 --out figures
python experiments/example_front.py --seed 1 --instance 3 --n-jobs 2 --apriori

Outputs
-------
<out>/example_front.png and <out>/example_front.pdf   the figure
<results>/<name>/config.yaml, environment.json        frozen arguments, package versions and command line
<results>/<name>/configurations.csv                   every evaluated configuration with objectives, front and knee flags
<results>/<name>/front_points.csv                     the three selected front members
<results>/<name>/explanations.txt                     linguistic description of the instance at the three points
<results>/<name>/explanation_knee.json                the knee explanation (PFCAExplanation.save)
<results>/<name>/seeds.json                           seed of the problem and seeds of every pool member
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import model_classes_from_names, prepare_run  # noqa: E402

from pfca import plotting  # noqa: E402
from pfca.evaluation.synthetic import make_synthetic  # noqa: E402
from pfca.explainer import PFCAExplainer, PFCAExplanation  # noqa: E402
from pfca.plotting import MUTED, PALETTE, TEXT, plot_pareto_front, style_axes  # noqa: E402

POINT_NAMES = ("sparsest", "knee", "most faithful")
POINT_MARKERS = {"sparsest": "s", "knee": "o", "most faithful": "^"}
POINT_LETTERS = {"sparsest": "a", "knee": "b", "most faithful": "c"}


def select_front_points(table: pd.DataFrame) -> dict[str, int]:
    """Indices (rows of the evaluated table) of the sparsest, knee and most faithful front members.

    The sparsest member retains the fewest concepts on the front (the sparsity
    level of Section 3.5), ties broken by complexity and fidelity loss; the
    most faithful member has the smallest fidelity loss, ties broken by
    complexity and instability.
    """
    front = table[table["pareto"]]
    if front.empty:
        raise ValueError("The evaluated table has no Pareto optimal configuration.")
    knee = int(table.index[table["knee"]][0])
    sparsest = int(front.sort_values(["n_retained", "complexity", "fidelity_loss"], kind="stable").index[0])
    faithful = int(front.sort_values(["fidelity_loss", "complexity", "instability"], kind="stable").index[0])
    return {"sparsest": sparsest, "knee": knee, "most faithful": faithful}


def short_concept_names(expl: PFCAExplanation, max_members: int = 2) -> list[str]:
    names = []
    for k in range(expl.n_concepts):
        mem = expl.partition.members(k)
        names.append(f"C{k + 1} ({', '.join(mem[:max_members])}{', ...' if len(mem) > max_members else ''})")
    return names


def draw_linguistic_panel(ax, expl: PFCAExplanation, instance: int, title: str, xlim: tuple[float, float]) -> None:
    """Compact linguistic bar panel: centroid bars, support whiskers and the best label per retained concept."""
    idx = sorted(list(expl.retained), key=lambda k: abs(expl.concept_centroids[instance, k]))
    names = short_concept_names(expl)
    label_idx, deg = expl.best_labels()
    y = np.arange(len(idx))
    cen = np.array([expl.concept_centroids[instance, k] for k in idx])
    lo = np.array([expl.quantiles[instance, k, 0] for k in idx])
    hi = np.array([expl.quantiles[instance, k, 4] for k in idx])
    colors = [PALETTE[0] if v >= 0 else PALETTE[1] for v in cen]
    ax.barh(y, cen, color=colors, height=0.55)
    ax.errorbar(cen, y, xerr=[np.maximum(cen - lo, 0), np.maximum(hi - cen, 0)], fmt="none", ecolor=TEXT, elinewidth=0.8, capsize=2)
    for yi, k in zip(y, idx):
        text = f"{expl.label_set.names[int(label_idx[instance, k])]} ({deg[instance, k]:.2f}); sign conf. {expl.sign_confidence[instance, k]:.2f}"
        ax.text(1.02, yi, text, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in idx], fontsize=7)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, max(len(idx) - 0.4, 0.6))
    ax.set_title(title, loc="left", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    style_axes(ax)


def make_figure(table: pd.DataFrame, explanations: dict[str, PFCAExplanation], points: dict[str, int], instance: int):
    """Front on the left, three linguistic panels on the right (one figure)."""
    plt = plotting._plt()
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1.0], hspace=0.7, wspace=0.45)
    ax_front = fig.add_subplot(gs[:, 0])
    plot_pareto_front(table, ax=ax_front)
    for name in POINT_NAMES:
        row = table.loc[points[name]]
        x, yv = float(row["complexity"]), float(row["fidelity_loss"])
        if name != "knee":
            ax_front.scatter([x], [yv], marker=POINT_MARKERS[name], s=90, facecolor="none", edgecolor=TEXT, linewidth=1.2, zorder=5, label=name)
        ax_front.annotate(f"({POINT_LETTERS[name]}) {name}", (x, yv), xytext=(7, 7), textcoords="offset points", fontsize=8, color=TEXT)
    ax_front.legend(fontsize=8)
    ax_front.set_title("Pareto front of explanation configurations", loc="left", fontsize=9)
    # common attribution axis across the three panels
    los, his = [0.0], [0.0]
    for e in explanations.values():
        for k in e.retained:
            los.append(float(e.quantiles[instance, k, 0]))
            his.append(float(e.quantiles[instance, k, 4]))
    span = max(max(his) - min(los), 1e-9)
    xlim = (min(los) - 0.05 * span, max(his) + 0.05 * span)
    for i, name in enumerate(POINT_NAMES):
        ax = fig.add_subplot(gs[i, 1])
        e = explanations[name]
        row = table.loc[points[name]]
        cfg = e.configuration
        same = [n for n in POINT_NAMES if n != name and points[n] == points[name]]
        extra = f", same configuration as the {' and the '.join(same)}" if same else ""
        title = f"({POINT_LETTERS[name]}) {name}: K = {cfg.n_concepts}, {int(e.retained.size)} retained, alpha = {cfg.alpha:.2f}, fidelity loss {row['fidelity_loss']:.3f}, complexity {row['complexity']:.2f}{extra}"
        draw_linguistic_panel(ax, e, instance, title, xlim)
        if i == len(POINT_NAMES) - 1:
            ax.set_xlabel(f"attribution to model output, instance {instance} (bar: centroid, whisker: support)", fontsize=8)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="additive", choices=["additive", "interaction", "redundancy"])
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--n-features", type=int, default=30)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--n-resamples", type=int, default=20, help="number of bootstrap resamples B of the pool")
    ap.add_argument("--n-explain", type=int, default=100, help="number of fresh instances explained")
    ap.add_argument("--background-size", type=int, default=100)
    ap.add_argument("--instance", type=int, default=0, help="explained instance shown in the linguistic panels")
    ap.add_argument("--apriori", action="store_true", help="include the true block partition as a candidate partition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", default="figures", help="directory of the figure")
    ap.add_argument("--results", default="results")
    ap.add_argument("--name", default="example_front")
    args = ap.parse_args(argv)
    if not (0 <= args.instance < args.n_explain):
        raise SystemExit("--instance must lie between 0 and --n-explain minus one")
    cfg = vars(args).copy()
    cfg.update({"model_classes": ["gbm", "rf"], "n_concepts_grid": [2, 3, 4, 5, 6, 8], "fuzzifier_grid": [1.5, 2.0, 2.5], "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0], "solver": "grid", "knee_method": "utopia"})
    out_dir = prepare_run(args.results, args.name, cfg)
    fig_dir = Path(args.out)
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"[example front] seed {args.seed}, {args.family} problem with {args.n_samples} samples, {args.n_features} features, rho {args.rho}; pool {args.n_resamples} x 2; {args.n_explain} instances; results in {out_dir}")

    t0 = time.time()
    problem = make_synthetic(args.family, n_samples=args.n_samples, n_features=args.n_features, rho=args.rho, random_state=args.seed)
    X_explain, _ = problem.sample(args.n_explain, random_state=args.seed + 2)
    classes = model_classes_from_names(cfg["model_classes"], "regression", args.seed)
    groups = [list(map(int, b)) for b in problem.blocks] if args.apriori else None
    explainer = PFCAExplainer(
        model_classes=classes,
        n_resamples=args.n_resamples,
        background_size=args.background_size,
        n_concepts_grid=tuple(k for k in cfg["n_concepts_grid"] if k < problem.n_features),
        fuzzifier_grid=tuple(cfg["fuzzifier_grid"]),
        alpha_grid=tuple(cfg["alpha_grid"]),
        apriori_groups=groups,
        partition_source="both" if groups is not None else "data",
        solver=cfg["solver"],
        knee_method=cfg["knee_method"],
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    explainer.fit(problem.X, problem.y, problem.feature_names)
    expl = explainer.explain(X_explain)
    print(f"[example front] pool and selection done in {time.time() - t0:.1f}s: {len(expl.selection.configurations)} configurations evaluated, {int(expl.selection.pareto.sum())} on the front")

    table = expl.selection.table
    table.to_csv(out_dir / "configurations.csv", index_label="index")
    points = select_front_points(table)
    explanations = {name: (expl if idx == expl.selection.knee else explainer.explanation_at(expl, idx)) for name, idx in points.items()}
    rows = []
    for name, idx in points.items():
        e = explanations[name]
        row = {"point": name, "index": idx, **e.configuration.as_dict(), "n_retained": int(e.retained.size)}
        row.update({c: float(table.loc[idx, c]) for c in ("fidelity_loss", "complexity", "instability")})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "front_points.csv", index=False)
    with open(out_dir / "explanations.txt", "w", encoding="utf8") as fh:
        for name, idx in points.items():
            fh.write(f"{name} (configuration index {idx})\n{explanations[name].describe(args.instance)}\n\n")
    expl.save(str(out_dir / "explanation_knee.json"))
    with open(out_dir / "seeds.json", "w", encoding="utf8") as fh:
        json.dump({"seed": args.seed, "explain_sample_seed": args.seed + 2, "pool_seeds": np.asarray(expl.attribution.seeds).tolist(), "model_classes": expl.attribution.model_class_names}, fh, indent=2)
    for name, idx in points.items():
        print(f"[example front] {name}: index {idx}, {explanations[name].configuration.as_dict()}, retained {int(explanations[name].retained.size)}")

    fig = make_figure(table, explanations, points, args.instance)
    written = []
    for ext in ("png", "pdf"):
        path = fig_dir / f"example_front.{ext}"
        fig.savefig(path, bbox_inches="tight")
        written.append(str(path))
    plotting._plt().close(fig)
    print(f"[example front] wrote {', '.join(written)} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
