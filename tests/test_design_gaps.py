"""Tests of the additions that close the gaps between the study design guide and the code.

Covered here: the LIME baseline output of the protocol (with the optional
package stubbed), the Pareto front diagnostics of Section 10, repeated cross
validation and the replicate choice of Section 7.3, the label set factor of
the sensitivity analysis, and the three items of the figure script that the
guide requires beyond Table 2 (linear mixed models of Section 7.4, the
success criteria of Section 7.6 and the front diagnostics).
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from pfca import PFCAExplainer
from pfca.attribution import AttributionEngine
from pfca.evaluation import datasets, protocol
from pfca.fuzzification import LinguisticLabelSet

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def _experiment_module(name: str):
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def engine(small_problem):
    p, _ = small_problem
    eng = AttributionEngine(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=20)), ("rf", RandomForestRegressor(n_estimators=20))], n_resamples=2, background_size=30, random_state=0)
    return eng.fit(p.X, p.y)


# ----------------------------------------------------------------------
# protocol: LIME output and objective conflict
# ----------------------------------------------------------------------


def test_lime_available_reflects_the_optional_package(monkeypatch):
    import importlib.util as iu

    monkeypatch.setattr(iu, "find_spec", lambda name: types.SimpleNamespace() if name == "lime" else None)
    assert protocol.lime_available() is True
    monkeypatch.setattr(iu, "find_spec", lambda name: None)
    assert protocol.lime_available() is False


def test_lime_output_with_a_stubbed_attribution(engine, small_problem, monkeypatch):
    p, X_test = small_problem
    calls = []

    def fake_lime(model, X_train, X_explain, task, num_samples=2000, random_state=0, feature_names=None):
        calls.append({"model": model, "n_train": X_train.shape[0], "task": task, "num_samples": num_samples, "random_state": random_state, "feature_names": feature_names})
        X_explain = np.asarray(X_explain, dtype=float)
        return X_explain * np.arange(1, X_explain.shape[1] + 1)

    monkeypatch.setattr(protocol, "lime_attribution", fake_lime)
    names = [f"item{j}" for j in range(p.n_features)]
    out = protocol.lime_output(engine, X_test[:6], "regression", feature_names=names, num_samples=50, random_state=3, n_retained=2)
    assert out.name == "lime" and out.level == "feature" and out.point.shape == (6, p.n_features)
    assert np.array_equal(out.U, np.eye(p.n_features)) and out.pool is None and out.quantiles is None
    assert out.retained is not None and out.retained.size == 2 and out.extra == {"num_samples": 50}
    assert calls[0]["model"] is engine.reference_model_ and calls[0]["n_train"] == p.X.shape[0]
    assert calls[0] == {**calls[0], "task": "regression", "num_samples": 50, "random_state": 3, "feature_names": names}
    np.testing.assert_allclose(out.attribution_fn(X_test[:2]), np.asarray(X_test[:2], dtype=float) * np.arange(1, p.n_features + 1))
    assert len(calls) == 2 and calls[1]["num_samples"] == 50 and out.wall_seconds >= 0
    # the output goes through the common evaluation like any other baseline
    ctx = protocol.EvaluationContext(model_fn=lambda X: engine.reference_model_.predict(X), X_explain=X_test[:6], background=p.X[:50], true_attr=p.true_concept_attributions(X_test[:6]), U_true=p.true_membership, perturbation_instances=0, n_perturbations=0)
    res = protocol.evaluate(out, ctx)
    assert res["method"] == "lime" and np.isfinite(res["recovery_error"]) and res["n_retained"] == 2


def test_lime_output_with_the_real_package(engine, small_problem):
    pytest.importorskip("lime")
    p, X_test = small_problem
    out = protocol.lime_output(engine, X_test[:3], "regression", num_samples=200, random_state=0)
    assert out.point.shape == (3, p.n_features) and np.isfinite(out.point).all()


class _FakeSelection:
    def __init__(self, objectives, pareto, representative=None):
        self.objectives = np.asarray(objectives, dtype=float)
        self.pareto = np.asarray(pareto, dtype=bool)
        self.representative = None if representative is None else np.asarray(representative, dtype=bool)


def test_objective_conflict_diagnostics():
    F = [[0.9, 1.0, 0.0], [0.6, 2.0, 0.1], [0.4, 3.0, 0.2], [0.3, 4.0, 0.3], [0.2, 5.0, 0.4], [np.inf, np.inf, np.inf]]
    pareto = [True, True, True, True, True, False]
    d = protocol.objective_conflict(_FakeSelection(F, pareto, representative=[True, True, True, False, False, False]))
    assert d["n_feasible"] == 5 and d["n_front_distinct"] == 3 and d["front_collapsed"] is False
    assert d["objective_corr_fidelity_complexity"] == pytest.approx(-1.0)
    assert d["objective_corr_fidelity_instability"] == pytest.approx(-1.0)
    assert d["objective_corr_complexity_instability"] == pytest.approx(1.0)
    # without representatives every front row counts; a constant objective has no correlation; fewer than three feasible rows neither
    d2 = protocol.objective_conflict(_FakeSelection([[0.5, 1.0, 0.0], [0.4, 2.0, 0.0], [0.3, 3.0, 0.0]], [True, True, True]))
    assert d2["n_front_distinct"] == 3 and np.isnan(d2["objective_corr_fidelity_instability"]) and d2["objective_corr_fidelity_complexity"] == pytest.approx(-1.0)
    d3 = protocol.objective_conflict(_FakeSelection([[0.5, 1.0, 0.0], [0.4, 2.0, 0.1]], [True, False]))
    assert d3["front_collapsed"] is True and np.isnan(d3["objective_corr_fidelity_complexity"])


def test_pfca_output_records_the_front_diagnostics(engine, small_problem):
    p, X_test = small_problem
    attr = engine.explain(X_test[:8])
    out = protocol.pfca_output("pfca", engine, attr, [f"x{j}" for j in range(p.n_features)], {"n_concepts_grid": (2, 3), "fuzzifier_grid": (2.0,), "alpha_grid": (0.0, 0.5)}, None)
    for key in ("n_feasible", "n_front_distinct", "front_collapsed", "objective_corr_fidelity_complexity"):
        assert key in out.extra
    assert 1 <= out.extra["n_front_distinct"] <= out.extra["n_front"] <= out.extra["n_feasible"]


# ----------------------------------------------------------------------
# datasets: repeated cross validation and the replicate choice
# ----------------------------------------------------------------------


def test_repeated_cv_splits():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=100) < 0.3).astype(int)
    splits = list(datasets.repeated_cv_splits(y, "classification", n_repeats=10, n_folds=5, random_state=1))
    assert [s[0] for s in splits] == list(range(10))
    tested = np.zeros(100, dtype=int)
    for r, tr, tu, te in splits:
        assert len(tr) == 60 and len(tu) == 20 and len(te) == 20
        assert set(tr) | set(tu) | set(te) == set(range(100)) and not (set(tr) & set(te)) and not (set(tu) & set(te)) and not (set(tr) & set(tu))
        assert abs(y[te].mean() - y.mean()) < 0.06  # stratified folds
        if r < 5:
            tested[te] += 1
    assert (tested == 1).all()  # every row is a test row once per shuffle
    # the test fold of repetition r is the tuning fold of repetition r - 1 within a shuffle
    assert np.array_equal(splits[1][3], splits[0][2])
    assert not np.array_equal(splits[5][3], splits[0][3])  # the second shuffle reorders the folds
    again = list(datasets.repeated_cv_splits(y, "classification", n_repeats=10, n_folds=5, random_state=1))
    assert all(np.array_equal(a[3], b[3]) for a, b in zip(splits, again))
    reg = list(datasets.repeated_cv_splits(np.arange(50, dtype=float), "regression", n_repeats=2, n_folds=5))
    assert len(reg) == 2 and len(reg[0][1]) == 30 and len(reg[0][3]) == 10
    with pytest.raises(ValueError, match="at least 3"):
        list(datasets.repeated_cv_splits(y, "classification", n_folds=2))


def test_make_splits_scheme_choice():
    y = np.arange(120, dtype=float)
    assert datasets.make_splits(y, "regression", n_repeats=2)[0] == "repeated_cv"
    assert datasets.make_splits(y, "regression", n_repeats=2, small_sample_rows=100)[0] == "holdout"
    used, splits = datasets.make_splits(y, "regression", n_repeats=3, scheme="holdout")
    assert used == "holdout" and len(splits) == 3 and len(splits[0][1]) == 72
    used, splits = datasets.make_splits(np.arange(1000, dtype=float), "regression", n_repeats=2, scheme="repeated_cv")
    assert used == "repeated_cv" and len(splits[0][3]) == 200
    with pytest.raises(ValueError, match="scheme must be"):
        datasets.make_splits(y, "regression", scheme="loo")


def test_replicate_repetition_prefers_the_split_that_holds_the_explained_rows_out():
    y = np.arange(100, dtype=float)
    _, cv = datasets.make_splits(y, "regression", n_repeats=5, scheme="repeated_cv")
    for r in range(5):
        j = datasets.replicate_repetition(cv, r, cv[r][3])
        assert j != r and not np.isin(cv[r][3], cv[j][1]).any()  # zero overlap exists with cross validation folds
    _, ho = datasets.make_splits(y, "regression", n_repeats=4, scheme="holdout")
    explain = ho[0][3]
    j = datasets.replicate_repetition(ho, 0, explain)
    overlaps = {k: int(np.isin(explain, ho[k][1]).sum()) for k in range(1, 4)}
    assert overlaps[j] == min(overlaps.values())
    assert datasets.replicate_repetition(ho[:1], 0, explain) is None
    # ties go to the next repetition in cyclic order
    same = [(0, np.arange(10), np.arange(10, 15), np.arange(15, 20))] * 3
    assert datasets.replicate_repetition(same, 2, np.arange(15, 20)) == 0


# ----------------------------------------------------------------------
# sensitivity: label set factor
# ----------------------------------------------------------------------


def test_label_set_specs_and_builder():
    rs = _experiment_module("run_sensitivity")
    specs = rs.label_set_specs({"label_sets": [{"name": "three", "centers": [-1, 0, 1], "names": ["neg", "zero", "pos"]}, {"name": "default"}, {"name": "wide", "centers": [-3, 0, 3]}]})
    assert list(specs) == ["default", "three", "wide"] and specs["default"] == {}
    assert specs["three"] == {"centers": [-1.0, 0.0, 1.0], "names": ["neg", "zero", "pos"]} and specs["wide"]["names"] is None
    assert rs.label_set_specs({}) == {"default": {}}
    assert rs.build_label_set({}).names == LinguisticLabelSet().names
    three = rs.build_label_set(specs["three"])
    assert three.names == ["neg", "zero", "pos"] and np.allclose(three.centers, [-1, 0, 1])
    assert rs.build_label_set(specs["wide"]).names == ["L1", "L2", "L3"]
    with pytest.raises(ValueError, match="needs a name"):
        rs.label_set_specs({"label_sets": [{"centers": [0, 1]}]})
    with pytest.raises(ValueError, match="at least two centers"):
        rs.label_set_specs({"label_sets": [{"name": "x", "centers": [0]}]})
    with pytest.raises(ValueError, match="names for"):
        rs.label_set_specs({"label_sets": [{"name": "x", "centers": [0, 1], "names": ["a"]}]})
    assert "labels" in rs.FACTORS
    default = rs.default_setting({"resamples": [2], "model_classes": ["gbm"]}, 1)
    plan = rs.build_plan({"resamples": [2], "label_sets": [{"name": "three", "centers": [-1, 0, 1]}]}, default, 1, ["labels"])
    assert [(f, n, s.label_set) for f, n, s in plan] == [("labels", "default", "default"), ("labels", "three", "three")]
    assert default.descriptors()["label_set"] == "default"


def test_label_statistics(small_problem):
    rs = _experiment_module("run_sensitivity")
    p, X_test = small_problem
    classes = [("gbm", GradientBoostingRegressor(n_estimators=20)), ("rf", RandomForestRegressor(n_estimators=20))]
    kw = dict(model_classes=classes, n_resamples=2, background_size=30, n_concepts_grid=(2, 3), fuzzifier_grid=(2.0,), alpha_grid=(0.0, 0.5), random_state=0)
    e_default = PFCAExplainer(**kw).fit(p.X, p.y).explain(X_test[:8])
    stats = rs.label_statistics(e_default)
    assert stats["n_labels"] == 5 and stats["label_agreement_with_default"] == 1.0 and stats["label_sign_agreement_with_default"] == 1.0
    assert 0.0 <= stats["label_entropy"] <= 1.0 and 0.0 <= stats["fraction_negligible"] <= 1.0 and 0.0 < stats["mean_label_compatibility"] <= 1.0
    three = LinguisticLabelSet.with_breakpoints([-1.0, 0.0, 1.0], ["negative", "negligible", "positive"])
    e_three = PFCAExplainer(linguistic_labels=three, **kw).fit(p.X, p.y).explain(X_test[:8])
    assert e_three.label_set.names == ["negative", "negligible", "positive"]
    s3 = rs.label_statistics(e_three)
    assert s3["n_labels"] == 3 and np.isnan(s3["label_agreement_with_default"]) and 0.0 <= s3["label_sign_agreement_with_default"] <= 1.0
    # the selection does not depend on the labels
    assert np.array_equal(e_three.retained, e_default.retained) and e_three.configuration == e_default.configuration


# ----------------------------------------------------------------------
# figure script: items 11 to 13
# ----------------------------------------------------------------------


def _phase_a_frame(n_seeds=3, pfca_shift=-0.3, rhos=(0.3, 0.9), noise=0.01):
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(n_seeds):
        for rho in rhos:
            for fam in ("additive", "redundancy"):
                base = rng.normal(0.8, 0.05)
                for method in ("pfca", "shap", "grouped_shap"):
                    shift = pfca_shift if method == "pfca" else 0.0
                    rows.append({"family": fam, "n_samples": 200, "n_features": 10, "rho": rho, "seed": seed, "method": method, "recovery_error": base + shift + rng.normal(0, noise), "rank_stability": 0.6 - shift + rng.normal(0, noise), "deletion_auc": 0.5 + rng.normal(0, noise), "support_coverage_truth": 0.91 if method == "pfca" else np.nan, "core_coverage_truth": 0.42 if method == "pfca" else np.nan, "n_front": 4 if method == "pfca" else np.nan, "n_front_distinct": 2 if method == "pfca" else np.nan, "front_collapsed": False if method == "pfca" else np.nan, "n_feasible": 10 if method == "pfca" else np.nan, "objective_corr_fidelity_complexity": -0.8 if method == "pfca" else np.nan})
    return pd.DataFrame(rows)


def test_item_phase_a_mixed_models(tmp_path):
    pytest.importorskip("statsmodels")
    mf = _experiment_module("make_figures")
    paths = mf.item_phase_a_mixed_models(_phase_a_frame(), tmp_path)
    assert [p.name for p in paths] == ["phase_a_mixed_models.csv", "phase_a_mixed_models.md"]
    df = pd.read_csv(paths[0])
    assert set(df["family"]) == {"additive", "redundancy"} and set(df["metric"]) >= {"recovery_error", "rank_stability", "deletion_auc"}
    assert (df["model"] == "linear mixed model, random intercept per seed").all() and df["n_seeds"].eq(3).all()
    red = df[(df["family"] == "redundancy") & (df["metric"] == "recovery_error") & (df["factor"] == "method")]
    assert set(red["level"]) == {"pfca", "grouped_shap"} and (red["reference_method"] == "shap").all()
    est = float(red.loc[red["level"] == "pfca", "estimate"].iloc[0])
    assert est == pytest.approx(-0.3, abs=0.05) and float(red.loc[red["level"] == "pfca", "p_value"].iloc[0]) < 0.01
    assert set(df.loc[df["metric"] == "recovery_error", "factor"]) == {"intercept", "method", "rho"}  # single level factors are dropped
    # a single seed falls back to ordinary least squares and says so
    single = pd.read_csv(mf.item_phase_a_mixed_models(_phase_a_frame(n_seeds=1), tmp_path / "single")[0]) if (tmp_path / "single").mkdir() is None else None
    assert (single["model"] == "ordinary least squares (single seed, no random effect)").all()
    with pytest.raises(ValueError, match="lack the columns"):
        mf.item_phase_a_mixed_models(pd.DataFrame({"method": ["pfca"], "recovery_error": [0.1]}), tmp_path)


def _phase_b_frame(pfca_gain=0.0):
    rows = []
    for ds in ("d1", "d2", "d3", "d4"):
        for rep in range(2):
            for method in ("pfca", "shap"):
                g = pfca_gain if method == "pfca" else 0.0
                rows.append({"dataset": ds, "repeat": rep, "method": method, "deletion_auc": 0.5 - g, "insertion_auc": 0.6 + g, "surrogate_r2": 0.8 + g, "support_coverage_replicate": 0.86 if method == "pfca" else np.nan, "core_coverage_replicate": 0.52 if method == "pfca" else np.nan})
    return pd.DataFrame(rows)


def test_item_success_criteria(tmp_path):
    mf = _experiment_module("make_figures")
    paths = mf.item_success_criteria(_phase_a_frame(), _phase_b_frame(pfca_gain=0.05), tmp_path)
    df = pd.read_csv(paths[0])
    assert sorted(df["criterion"].unique()) == [1, 2, 3]
    c1 = df[df["criterion"] == 1].set_index("quantity")
    assert c1.loc["recovery_error", "status"] == "met" and c1.loc["recovery_error", "cohen_d"] < -0.5 and c1.loc["recovery_error", "n"] == 6
    assert c1.loc["rank_stability", "status"] == "met"
    c2 = df[df["criterion"] == 2]
    assert (c2["status"] == "met").all() and c2["n"].eq(4).all()
    c3 = df[df["criterion"] == 3].set_index(["phase", "quantity"])
    assert c3.loc[("A", "support_coverage_truth"), "status"] == "met" and c3.loc[("A", "core_coverage_truth"), "status"] == "not met"
    assert c3.loc[("B", "support_coverage_replicate"), "status"] == "met" and c3.loc[("B", "core_coverage_replicate"), "status"] == "met"
    text = paths[1].read_text(encoding="utf8")
    assert "Overall: not met." in text and "Section 7.6" in text
    # no advantage of PFCA: criterion 1 fails, criterion 2 still holds because the methods tie
    tie = pd.read_csv(mf.item_success_criteria(_phase_a_frame(pfca_shift=0.0, noise=0.0), _phase_b_frame(0.0), tmp_path)[0])
    assert (tie.loc[tie["criterion"] == 1, "status"] == "not met").all() and (tie.loc[tie["criterion"] == 2, "status"] == "met").all()
    # a worse PFCA on every dataset fails criterion 2
    worse = pd.read_csv(mf.item_success_criteria(None, _phase_b_frame(-0.05), tmp_path)[0])
    assert (worse.loc[worse["criterion"] == 2, "status"] == "not met").all() and (worse.loc[worse["criterion"] == 1, "status"] == "not evaluated").all()
    with pytest.raises(ValueError, match="neither"):
        mf.item_success_criteria(None, None, tmp_path)


def test_item_front_diagnostics(tmp_path):
    mf = _experiment_module("make_figures")
    dfs = pd.DataFrame({"factor": ["solver", "solver", "knee"], "setting": ["grid", "nsga2", "utopia"], "n_front": [4, 7, 4], "n_front_distinct": [3, 5, 3], "front_collapsed": [False, False, False], "objective_corr_fidelity_complexity": [-0.5, -0.4, -0.5]})
    paths = mf.item_front_diagnostics(_phase_a_frame(), dfs, tmp_path)
    df = pd.read_csv(paths[0])
    assert list(df["source"]) == ["Phase A", "Phase A", "Phase A", "sensitivity, solver", "sensitivity, solver"]
    assert list(df["group"]) == ["additive", "redundancy", "all families", "grid", "nsga2"]
    assert df.loc[df["group"] == "all families", "n"].iloc[0] == 12 and df["fraction_collapsed"].eq(0).all()
    assert df.loc[df["group"] == "nsga2", "n_front"].iloc[0] == 7
    with pytest.raises(ValueError, match="no PFCA rows"):
        mf.item_front_diagnostics(_phase_a_frame().drop(columns=["n_front"]), None, tmp_path)
