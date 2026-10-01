"""Plotting helpers (matplotlib is an optional dependency).

The figures follow a small, fixed categorical palette assigned in order,
thin marks, recessive axes and direct labels where useful.
"""

from __future__ import annotations

import numpy as np

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting; install with 'pip install matplotlib'.") from exc
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )
    return plt


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    return ax


def plot_fuzzy_attribution(explanation, instance: int, retained_only: bool = True, ax=None, scale_axis: bool = False):
    """Membership functions of the fuzzy attributions of one instance, one curve per concept."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.2))
    names = explanation.concept_names
    idx = list(explanation.retained) if retained_only else list(range(explanation.n_concepts))
    lo = min(explanation.quantiles[instance, k, 0] for k in idx)
    hi = max(explanation.quantiles[instance, k, 4] for k in idx)
    pad = 0.1 * (hi - lo + 1e-9)
    z = np.linspace(lo - pad, hi + pad, 400)
    s = explanation.scale if scale_axis else 1.0
    for c, k in enumerate(idx):
        F = explanation.fuzzy_attribution(instance, k)
        color = PALETTE[c % len(PALETTE)]
        if hasattr(F, "members"):
            up, low = F.upper_membership(z), F.lower_membership(z)
            ax.fill_between(z / s, low, up, color=color, alpha=0.25, linewidth=0)
            ax.plot(z / s, up, color=color, linewidth=2, label=names[k] + " (type 2)")
        else:
            ax.plot(z / s, F.membership(z), color=color, linewidth=2, label=names[k])
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("standardized attribution" if scale_axis else "attribution to model output")
    ax.set_ylabel("membership")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    return ax


def plot_linguistic_bars(explanation, instance: int, retained_only: bool = True, ax=None):
    """Concept bars with the centroid, support as a whisker and the best label as text."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.2))
    names = explanation.concept_names
    idx = list(explanation.retained) if retained_only else list(range(explanation.n_concepts))
    idx = sorted(idx, key=lambda k: abs(explanation.concept_centroids[instance, k]))
    label_idx, deg = explanation.best_labels()
    y = np.arange(len(idx))
    cen = np.array([explanation.concept_centroids[instance, k] for k in idx])
    lo = np.array([explanation.quantiles[instance, k, 0] for k in idx])
    hi = np.array([explanation.quantiles[instance, k, 4] for k in idx])
    colors = [PALETTE[0] if v >= 0 else PALETTE[1] for v in cen]
    ax.barh(y, cen, color=colors, height=0.55)
    ax.errorbar(cen, y, xerr=[cen - lo, hi - cen], fmt="none", ecolor=TEXT, elinewidth=1, capsize=3)
    for yi, k, v in zip(y, idx, cen):
        text = f"{explanation.label_set.names[label_idx[instance, k]]} ({deg[instance, k]:.2f}); sign conf. {explanation.sign_confidence[instance, k]:.2f}"
        ax.text(hi[yi] if v >= 0 else lo[yi], yi, "  " + text if v >= 0 else text + "  ", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in idx])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("attribution to model output (bar: centroid, whisker: support)")
    style_axes(ax)
    return ax


def plot_shap_bars(values: np.ndarray, feature_names, ax=None, top: int = 15, title: str = "SHAP"):
    """Plain feature level bar plot for comparison formats."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.2))
    v = np.asarray(values, dtype=float)
    order = np.argsort(np.abs(v))[-top:]
    colors = [PALETTE[0] if x >= 0 else PALETTE[1] for x in v[order]]
    ax.barh(np.arange(order.size), v[order], color=colors, height=0.55)
    ax.set_yticks(np.arange(order.size))
    ax.set_yticklabels([feature_names[j] for j in order])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("attribution to model output")
    ax.set_title(title, loc="left")
    style_axes(ax)
    return ax


def plot_pareto_front(table, ax=None, highlight_knee: bool = True):
    """Fidelity loss against complexity for evaluated configurations, colored by instability."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.8))
    dom = table[~table["pareto"]]
    front = table[table["pareto"]].sort_values("complexity")
    ax.scatter(dom["complexity"], dom["fidelity_loss"], s=14, color=GRID, edgecolor=MUTED, linewidth=0.4, label="dominated")
    sc = ax.scatter(front["complexity"], front["fidelity_loss"], s=36, c=front["instability"], cmap="Blues", vmin=0, vmax=max(0.05, float(front["instability"].max())), edgecolor=TEXT, linewidth=0.5, label="Pareto front")
    ax.plot(front["complexity"], front["fidelity_loss"], color=PALETTE[0], linewidth=1, alpha=0.6)
    if highlight_knee and table["knee"].any():
        k = table[table["knee"]].iloc[0]
        ax.scatter([k["complexity"]], [k["fidelity_loss"]], s=120, facecolor="none", edgecolor=PALETTE[1], linewidth=2, label="knee point")
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("instability")
    ax.set_xlabel("complexity")
    ax.set_ylabel("fidelity loss")
    ax.legend(fontsize=8)
    style_axes(ax)
    return ax


def plot_membership(partition, ax=None):
    """Heat map of the feature by concept membership matrix."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    if ax is None:
        _, ax = plt.subplots(figsize=(4 + 0.3 * partition.n_concepts, 0.25 * partition.n_features + 1.2))
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    im = ax.imshow(partition.U, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(partition.n_concepts))
    ax.set_xticklabels([f"C{k + 1}" for k in range(partition.n_concepts)])
    ax.set_yticks(range(partition.n_features))
    ax.set_yticklabels(partition.feature_names, fontsize=7)
    ax.grid(False)
    plt.colorbar(im, ax=ax, pad=0.02, label="membership")
    return ax


def plot_reliability(table, ax=None, label: str = "sign confidence"):
    """Reliability diagram from the table of sign_confidence_calibration."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(3.8, 3.8))
    t = table.dropna()
    ax.plot([0.5, 1], [0.5, 1], color=MUTED, linewidth=0.8, linestyle=":")
    ax.plot(t["mean_confidence"], t["observed"], marker="o", markersize=5, color=PALETTE[0], linewidth=2, label=label)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("observed agreement")
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.5, 1)
    style_axes(ax)
    return ax
