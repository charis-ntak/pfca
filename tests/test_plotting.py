"""Tests of the plotting helpers (matplotlib is an optional dependency, so the module is skipped without it)."""

import sys

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402

from pfca import PFCAExplainer, plotting  # noqa: E402
from pfca.evaluation import metrics  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def explanations(small_problem):
    """Two explanations of the same pool: one without type 2 sets (threshold 1) and one where every disagreement upgrades the set (threshold 0)."""
    p, X_test = small_problem

    def fit(threshold):
        ex = PFCAExplainer(
            model_classes=[("gbm", GradientBoostingRegressor(n_estimators=30)), ("rf", RandomForestRegressor(n_estimators=30))],
            n_resamples=3,
            background_size=40,
            n_concepts_grid=(2, 3),
            fuzzifier_grid=(2.0,),
            alpha_grid=(0.0, 0.5),
            disagreement_threshold=threshold,
            random_state=0,
        )
        return ex.fit(p.X, p.y, feature_names=[f"item{j}" for j in range(p.n_features)]).explain(X_test[:5])

    return fit(1.0), fit(0.0)


def legend_texts(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_fuzzy_attribution_curves(explanations, tmp_path):
    e1, e2 = explanations
    ax = plotting.plot_fuzzy_attribution(e1, 0)
    assert legend_texts(ax) == [e1.concept_names[k] for k in e1.retained]
    assert ax.get_ylim() == (0.0, 1.05)
    assert ax.get_xlabel() == "attribution to model output"
    assert ax.get_ylabel() == "membership"
    out = tmp_path / "fuzzy.png"
    ax.figure.savefig(out)
    assert out.stat().st_size > 0
    # every concept on a standardized axis
    ax2 = plotting.plot_fuzzy_attribution(e1, 0, retained_only=False, scale_axis=True)
    assert legend_texts(ax2) == list(e1.concept_names)
    assert ax2.get_xlabel() == "standardized attribution"
    # a caller supplied axes is drawn on and returned
    _, mine = plt.subplots()
    assert plotting.plot_fuzzy_attribution(e1, 1, ax=mine) is mine
    # type 1 sets only when the threshold is never exceeded
    assert not any(hasattr(e1.fuzzy_attribution(0, k), "members") for k in range(e1.n_concepts))
    assert not ax2.collections


def test_fuzzy_attribution_type2_sets(explanations):
    _, e2 = explanations
    assert any(hasattr(e2.fuzzy_attribution(0, k), "members") for k in range(e2.n_concepts))
    ax = plotting.plot_fuzzy_attribution(e2, 0, retained_only=False)
    texts = legend_texts(ax)
    n_type2 = sum(hasattr(e2.fuzzy_attribution(0, k), "members") for k in range(e2.n_concepts))
    assert sum(t.endswith("(type 2)") for t in texts) == n_type2
    assert len(ax.collections) == n_type2  # one footprint between the lower and upper membership per type 2 set
    assert len(texts) == e2.n_concepts


def test_linguistic_bars(explanations):
    e1, _ = explanations
    ax = plotting.plot_linguistic_bars(e1, 0)
    idx = sorted(e1.retained, key=lambda k: abs(e1.concept_centroids[0, k]))
    assert len(ax.patches) == len(idx)
    assert [t.get_text() for t in ax.get_yticklabels()] == [e1.concept_names[k] for k in idx]
    assert len(ax.texts) == len(idx) and all("sign conf." in t.get_text() for t in ax.texts)
    for patch, k in zip(ax.patches, idx):
        expected = plotting.PALETTE[0] if e1.concept_centroids[0, k] >= 0 else plotting.PALETTE[1]
        assert patch.get_facecolor() == to_rgba(expected)
    ax_all = plotting.plot_linguistic_bars(e1, 0, retained_only=False)
    assert len(ax_all.patches) == e1.n_concepts


def test_shap_bars():
    rng = np.random.default_rng(0)
    values = rng.normal(size=12)
    values[3] = -5.0
    names = [f"f{j}" for j in range(12)]
    ax = plotting.plot_shap_bars(values, names, top=4, title="reference")
    order = np.argsort(np.abs(values))[-4:]
    assert len(ax.patches) == 4
    assert [t.get_text() for t in ax.get_yticklabels()] == [names[j] for j in order]
    assert ax.get_title(loc="left") == "reference"
    assert ax.patches[-1].get_facecolor() == to_rgba(plotting.PALETTE[1])  # the largest magnitude is negative
    ax_all = plotting.plot_shap_bars(values, names, top=100)
    assert len(ax_all.patches) == 12


def test_pareto_front(explanations):
    e1, _ = explanations
    table = e1.selection.table
    assert table["knee"].sum() == 1 and table["pareto"].any()
    ax = plotting.plot_pareto_front(table)
    assert legend_texts(ax) == ["dominated", "Pareto front", "knee point"]
    assert ax.get_xlabel() == "complexity" and ax.get_ylabel() == "fidelity loss"
    assert len(ax.figure.axes) == 2  # the colorbar of the instability
    ax_no = plotting.plot_pareto_front(table, highlight_knee=False)
    assert legend_texts(ax_no) == ["dominated", "Pareto front"]
    no_knee = table.copy()
    no_knee["knee"] = False
    assert legend_texts(plotting.plot_pareto_front(no_knee)) == ["dominated", "Pareto front"]


def test_membership_heat_map(explanations):
    e1, _ = explanations
    part = e1.partition
    ax = plotting.plot_membership(part)
    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape == (part.n_features, part.n_concepts)
    assert [t.get_text() for t in ax.get_xticklabels()] == [f"C{k + 1}" for k in range(part.n_concepts)]
    assert [t.get_text() for t in ax.get_yticklabels()] == list(part.feature_names)


def test_reliability_diagram():
    rng = np.random.default_rng(1)
    stated = rng.uniform(0.5, 1.0, size=200)
    agreement = (rng.uniform(size=200) < stated).astype(float)
    table = metrics.sign_confidence_calibration(stated, agreement, n_bins=5)["table"]
    ax = plotting.plot_reliability(table, label="stated")
    assert len(ax.get_lines()) == 2  # the diagonal and the curve
    assert ax.get_lines()[1].get_label() == "stated"
    assert ax.get_xlim() == (0.5, 1.0) and ax.get_ylim() == (0.5, 1.0)
    # empty bins carry missing values and are dropped before drawing
    sparse = pd.concat([table, pd.DataFrame([{"mean_confidence": np.nan, "observed": np.nan}])], ignore_index=True)
    ax2 = plotting.plot_reliability(sparse)
    x, y = ax2.get_lines()[1].get_data()
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_style_axes():
    _, ax = plt.subplots()
    assert plotting.style_axes(ax) is ax
    assert ax.get_axisbelow() is True


def test_plt_reports_a_missing_matplotlib(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match="matplotlib is required"):
        plotting._plt()
