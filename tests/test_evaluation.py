import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from pfca.attribution import AttributionEngine
from pfca.evaluation import baselines, budget, datasets, metrics, protocol, statistics
from pfca.evaluation.synthetic import make_synthetic, phase_a_grid


def test_synthetic_families():
    for fam in ("additive", "interaction", "redundancy"):
        p = make_synthetic(fam, n_samples=200, n_features=10, rho=0.5, random_state=0)
        assert p.X.shape[0] == 200 and p.y.shape == (200,)
        assert p.n_factors == 3
        A = p.true_concept_attributions(p.X)
        f = p.true_function(p.X)
        assert A.shape == (200, 3)
        np.testing.assert_allclose(A.sum(axis=1), f - p.true_intercept(), atol=1e-8)
        assert p.true_membership.shape == (p.n_features, 3)
        Xs, ys = p.sample(20, random_state=1)
        assert Xs.shape == (20, p.n_features)
    r = make_synthetic("redundancy", n_samples=100, n_features=10, rho=0.5, random_state=0)
    assert r.n_features == 13
    src, pos, sd = r.duplicates[0]
    np.testing.assert_allclose(r.X[:, pos], r.X[:, src])
    assert r.feature_names[pos].startswith("dup0_of_")
    with pytest.raises(ValueError):
        make_synthetic("other")
    assert len(phase_a_grid(n_seeds=2)) == 3 * 3 * 3 * 3 * 2


def test_correctness_metrics():
    rng = np.random.default_rng(0)
    U_true = np.repeat(np.eye(3), 2, axis=0)
    true = rng.normal(size=(20, 3))
    assert metrics.attribution_recovery_error(true, true, U_true, U_true) == 0.0
    perm = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    U_perm = U_true @ perm
    assert metrics.attribution_recovery_error(true @ perm, true, U_perm, U_true) == pytest.approx(0.0)
    assert metrics.feature_level_recovery_error(np.repeat(true, 2, axis=1) / 2, true, U_true) == pytest.approx(0.0)
    cr = metrics.concept_recovery(U_perm, U_true)
    assert cr["ari"] == 1.0 and cr["fpc"] == 1.0
    assert metrics.rank_recovery(np.array([3.0, 2.0, 1.0]), np.array([3.0, 2.0, 1.0])) == 1.0
    assert metrics.rank_recovery(np.array([1.0, 2.0, 3.0]) @ perm, np.array([1.0, 2.0, 3.0]), U_perm, U_true) == 1.0
    a = metrics.match_concepts_to_factors(U_perm, U_true, one_to_one=True)
    assert sorted(a.tolist()) == [0, 1, 2]


def test_faithfulness_and_stability_metrics():
    p = make_synthetic("additive", n_samples=300, n_features=6, rho=0.6, n_factors=2, random_state=1, nonlinear=False)
    A = p.true_concept_attributions(p.X[:20])
    order = np.argsort(-np.abs(A), axis=1)
    r = metrics.deletion_insertion_curves(p.true_function, p.X[:20], p.X[100:300], p.true_membership, order)
    assert 0 <= r["deletion_auc"] <= 1.5 and r["deletion_curve"].shape == (3,)
    r_m = metrics.deletion_insertion_curves(p.true_function, p.X[:20], p.X[100:300], p.true_membership, order[0], mode="marginal")
    assert np.isfinite(r_m["insertion_auc"])
    with pytest.raises(ValueError):
        metrics.deletion_insertion_curves(p.true_function, p.X[:20], p.X[100:300], p.true_membership, order, mode="bad")
    f = p.true_function(p.X[:60])
    assert metrics.surrogate_fidelity(p.true_concept_attributions(p.X[:60]), f) > 0.99
    assert np.isnan(metrics.surrogate_fidelity(np.zeros((60, 0)), f))
    ll = metrics.surrogate_fidelity(np.linspace(0, 1, 60)[:, None], np.linspace(0, 1, 60), metric="log_loss")
    assert ll > 0
    I = np.tile(np.array([[3.0, 2.0, 1.0]]), (4, 1))
    assert metrics.rank_stability(I) == 1.0
    assert np.isnan(metrics.rank_stability(I[:1]))
    ps = metrics.perturbation_stability(lambda X: X * 2, lambda X: X.sum(axis=1), p.X[:10], sigma=0.01, n_perturbations=2)
    assert ps >= 0


def test_calibration_and_robustness_metrics():
    rng = np.random.default_rng(0)
    Q = np.stack([np.full((5, 2), -1.0), np.full((5, 2), -0.5), np.zeros((5, 2)), np.full((5, 2), 0.5), np.full((5, 2), 1.0)], axis=-1)
    cov = metrics.interval_coverage(Q, np.full((5, 2), 0.7))
    assert cov == {"support_coverage": 1.0, "core_coverage": 0.0}
    stated = rng.uniform(0.5, 1, size=200)
    agree = (rng.uniform(size=200) < stated).astype(float)
    cal = metrics.sign_confidence_calibration(stated, agree, n_bins=5)
    assert cal["table"].shape[0] == 5 and 0 <= cal["ece"] <= 1
    assert metrics.observed_sign_agreement(np.array([1, -1]), np.array([2.0, 3.0])).tolist() == [1.0, 0.0]
    assert metrics.duplication_invariance(np.array([2.0, 2.0]), np.array([1.0, 1.0])) == 0.5


def test_grouped_shapley_exact_and_sampled():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    beta = np.array([1.0, 2.0, -1.0, 0.5, 0.0, 3.0])
    model = LinearRegression().fit(X, X @ beta)
    bg = X[:60]
    groups = [[0, 1], [2, 3], [4, 5]]
    g, ev = baselines.grouped_shapley(model.predict, X[:4], bg, groups)
    expected = np.stack([(beta[grp] * (X[:4][:, grp] - bg[:, grp].mean(axis=0))).sum(axis=1) for grp in groups], axis=1)
    np.testing.assert_allclose(g, expected, atol=1e-8)
    assert ev == pytest.approx(model.predict(bg).mean())
    gs, _ = baselines.grouped_shapley(model.predict, X[:2], bg, groups, max_exact_groups=2, n_permutations=40, random_state=0)
    np.testing.assert_allclose(gs, expected[:2], atol=1e-6)


def test_integrated_gradients_linear():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    beta = np.array([1.0, -1.0, 0.5])
    base = np.zeros(3)
    ig = baselines.integrated_gradients(lambda Z: Z @ beta, X, base, steps=10)
    np.testing.assert_allclose(ig, X * beta, atol=1e-6)


def test_bootstrapped_shap_and_feature_shap(small_problem):
    p, X_test = small_problem
    out = baselines.bootstrapped_shap(GradientBoostingRegressor(n_estimators=20), p.X, p.y, X_test[:5], n_resamples=3, background_size=30)
    assert out["pool"].shape == (3, 5, p.n_features)
    assert np.all(out["lower"] <= out["upper"])
    model = GradientBoostingRegressor(n_estimators=20).fit(p.X, p.y)
    v, e = baselines.feature_shap(model, p.X, X_test[:5], "regression", method="tree", background_size=30)
    assert v.shape == (5, p.n_features)
    from pfca.evaluation.baselines import ablation_explainer

    for kind in ("full", "crisp", "single_model", "fixed"):
        ex = ablation_explainer(kind, {"n_resamples": 2}, [("gbm", GradientBoostingRegressor(n_estimators=10))])
        assert ex.n_resamples == 2
    with pytest.raises(ValueError):
        ablation_explainer("apriori", {}, [])
    with pytest.raises(ValueError):
        ablation_explainer("unknown", {}, [])


def test_protocol_end_to_end(small_problem):
    p, X_test = small_problem
    eng = AttributionEngine(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=30)), ("rf", GradientBoostingRegressor(n_estimators=10))], n_resamples=3, background_size=40, random_state=0).fit(p.X, p.y)
    attr = eng.explain(X_test)
    groups = [list(b) for b in p.blocks]
    kw = {"n_concepts_grid": (2, 3), "fuzzifier_grid": (2.0,), "alpha_grid": (0.0, 0.5)}
    outs = [protocol.pfca_output(name, eng, attr, None, kw, groups) for name in ("pfca", "pfca_crisp", "pfca_single_model", "pfca_fixed", "pfca_apriori")]
    outs.append(protocol.shap_output(eng, attr, "regression"))
    outs.append(protocol.bootstrapped_shap_output(eng, attr))
    outs.append(protocol.grouped_shap_output(eng, attr, X_test, groups, "regression", background_size=20))
    from pfca.attribution import output_function

    rep_model = GradientBoostingRegressor(n_estimators=30, random_state=1).fit(*p.sample(300, random_state=7))
    from pfca.attribution import shapley_values

    rep_attr, _ = shapley_values(rep_model, X_test, p.X[:40], "regression", explainer="tree")
    ctx = protocol.EvaluationContext(
        model_fn=output_function(eng.reference_model_, "regression"),
        X_explain=X_test,
        background=p.X[:100],
        true_attr=p.true_concept_attributions(X_test),
        U_true=p.true_membership,
        true_importance=p.true_global_importance(X_test),
        replicate_feature_attr=rep_attr,
        perturbation_instances=3,
        n_perturbations=1,
    )
    rows = [protocol.evaluate(o, ctx) for o in outs]
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)} for r in rows])
    assert set(df["method"]) == {"pfca", "pfca_crisp", "pfca_single_model", "pfca_fixed", "pfca_apriori", "shap", "bootstrapped_shap", "grouped_shap"}
    for col in ("recovery_error", "ari", "rank_recovery", "deletion_auc", "insertion_auc", "surrogate_r2", "rank_stability", "perturbation_stability"):
        assert col in df.columns
        assert df[col].notna().all(), col
    assert df.loc[df["method"] == "pfca_apriori", "ari"].iloc[0] == 1.0
    assert df.loc[df["method"] == "pfca", "support_coverage_replicate"].notna().all()
    assert df.loc[df["method"] == "pfca", "sign_confidence_ece"].notna().all()
    assert protocol.duplication_change(outs[0], outs[0], 0, 0, p.true_membership) == 0.0
    assert protocol.duplication_change(outs[5], outs[5], 0, 0, p.true_membership) == 0.0


def test_statistics():
    adj = statistics.holm_correction([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    a = rng.normal(size=8)
    table = pd.DataFrame({"a": a, "b": a + 1 + 0.1 * rng.normal(size=8), "c": a + 2 + 0.1 * rng.normal(size=8)})
    w = statistics.wilcoxon_holm(table, "a")
    assert set(w["method"]) == {"b", "c"} and "p_holm" in w.columns
    f = statistics.friedman_nemenyi(table)
    assert f["n_datasets"] == 8 and f["significant"].shape == (3, 3)
    assert f["average_ranks"].index[0] == "a"
    es = statistics.paired_effect_sizes(table["a"], table["b"], n_boot=50)
    assert es["cohen_d"] < 0 and len(es["cohen_d_ci"]) == 2
    df = pd.DataFrame({"y": rng.normal(size=40), "method": np.tile(["m1", "m2"], 20), "seed": np.repeat(np.arange(20), 2)})
    res = statistics.mixed_model(df, "y", ["method"], "seed")
    assert hasattr(res, "params")
    s = statistics.summarize_by(df, ["method"], ["y"])
    assert ("y", "ci_low") in s.columns
    lik = pd.DataFrame({"rating": rng.integers(1, 6, size=60), "format": np.tile(["shap", "concept", "fuzzy"], 20), "expert": np.repeat(np.arange(10), 6)})
    om = statistics.ordinal_model(lik, "rating", ["format"])
    assert hasattr(om, "params")
    acc = pd.DataFrame({"correct": rng.integers(0, 2, size=60), "format": lik["format"], "expert": lik["expert"]})
    bm = statistics.binomial_mixed_model(acc, "correct", ["format"], "expert")
    assert hasattr(bm, "fe_mean")


def test_datasets_helpers(tmp_path):
    y = np.array([0, 1] * 50)
    splits = list(datasets.stratified_splits(y, "classification", n_repeats=2))
    assert len(splits) == 2
    r, tr, tu, te = splits[0]
    assert len(tr) == 60 and len(tu) == 20 and len(te) == 20
    assert set(tr) & set(te) == set()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(1, 6, size=(40, 6)), columns=[f"q{i}" for i in range(6)])
    df["age"] = rng.integers(18, 60, size=40)
    df["score"] = df[["q0", "q1", "q2"]].sum(axis=1) + rng.normal(size=40)
    csv = tmp_path / "q.csv"
    df.to_csv(csv, index=False)
    spec = {"name": "demo", "outcome": "score", "task": "regression", "subscales": {"A": ["q0", "q1", "q2"], "B": ["q3", "q4", "q5"]}, "covariates": ["age"]}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    ds = datasets.load_questionnaire(csv, spec_path)
    assert ds.n_features == 7 and ds.groups is not None and "covariates" in ds.groups
    assert len(datasets.BENCHMARKS) >= 6


def test_budget():
    with budget.measure() as b:
        _ = np.ones(10000)
    assert b.wall_seconds >= 0 and b.peak_python_mb >= 0


def test_make_synthetic_validates_arguments():
    kw = dict(n_samples=50, n_features=6, n_factors=2, random_state=0)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="rho"):
            make_synthetic("additive", rho=bad, **kw)
    for ok in (0.0, 1.0):
        p = make_synthetic("additive", rho=ok, **kw)
        assert np.isfinite(p.X).all() and np.isfinite(p.y).all()
    with pytest.raises(ValueError, match="coefficients"):
        make_synthetic("additive", rho=0.5, coefficients=[1.0], **kw)
    with pytest.raises(ValueError, match="link"):
        make_synthetic("additive", rho=0.5, links=["linear", "exp"], **kw)
    with pytest.raises(ValueError, match="link"):
        make_synthetic("additive", rho=0.5, links=["linear"], **kw)


def test_rank_recovery_aggregates_attributions_before_absolute_value():
    # Feature level attribution whose block sums equal the true factor attributions exactly,
    # with the first block split into contributions of opposite signs.
    p = make_synthetic("additive", n_samples=200, n_features=9, rho=0.5, n_factors=3, random_state=0, nonlinear=False, coefficients=[1.0, 1.6, 0.4])
    A = p.true_concept_attributions(p.X)
    U_true = p.true_membership
    phi = np.zeros((200, 9))
    phi[:, 0:3] = A[:, [0]] * np.array([1.6, -0.3, -0.3])
    phi[:, 3:6] = A[:, [1]] / 3.0
    phi[:, 6:9] = A[:, [2]] / 3.0
    assert metrics.feature_level_recovery_error(phi, A, U_true) == pytest.approx(0.0, abs=1e-12)
    true_imp = p.true_global_importance(p.X)
    # attribution matrix: the mapped attributions are summed before the absolute value
    assert metrics.rank_recovery(phi, true_imp, np.eye(9), U_true) == 1.0
    # the same matrix without memberships reduces to the mean absolute attribution per column
    assert metrics.rank_recovery(A, true_imp) == 1.0
    assert metrics.rank_recovery(np.abs(A).mean(axis=0), true_imp) == 1.0
    # importances alone remain accepted; summing per concept importances inflates the split block
    assert metrics.rank_recovery(np.abs(phi).mean(axis=0), true_imp, np.eye(9), U_true) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        metrics.rank_recovery(np.zeros((2, 2, 2)), true_imp)


def test_conditional_imputation_ignores_constant_background_columns():
    rng = np.random.default_rng(0)
    bg = rng.normal(size=(200, 4))
    bg[:, 3] = 1.0
    Xq = rng.normal(size=(5, 4))
    Xq[:, 3] = 0.0
    w = np.zeros((5, 4))
    w[:, 0] = 1.0
    imputed = metrics._impute(Xq, w, bg, "conditional", np.random.default_rng(0), k_neighbors=10)
    # reference k nearest neighbors on the informative kept features 1 and 2
    sd = bg[:, 1:3].std(axis=0)
    expected = []
    for i in range(5):
        dist = (((bg[:, 1:3] - Xq[i, 1:3]) / sd) ** 2).sum(axis=1)
        expected.append(bg[np.argsort(dist)[:10], 0].mean())
    np.testing.assert_allclose(imputed[:, 0], expected)
    np.testing.assert_allclose(imputed[:, 1:], Xq[:, 1:])
    assert np.ptp(imputed[:, 0]) > 0.05
    # a query whose only kept features are constant in the background receives the background mean
    w_all = np.ones((1, 4))
    w_all[0, 3] = 0.0
    np.testing.assert_allclose(metrics._impute(Xq[:1], w_all, bg, "conditional", np.random.default_rng(0))[0, :3], bg[:, :3].mean(axis=0))


def test_marginal_curves_reuse_replacement_rows():
    p = make_synthetic("additive", n_samples=300, n_features=6, rho=0.6, n_factors=2, random_state=1, nonlinear=False)
    A = p.true_concept_attributions(p.X[:20])
    order = np.argsort(-np.abs(A), axis=1)
    for draws in (1, 10):
        r = metrics.deletion_insertion_curves(p.true_function, p.X[:20], p.X[100:300], p.true_membership, order, mode="marginal", marginal_draws=draws)
        assert r["deletion_curve"][0] == pytest.approx(1.0)
        assert r["deletion_curve"][-1] == pytest.approx(0.0, abs=1e-12)
        assert r["insertion_curve"][0] == 0.0
        assert r["insertion_curve"][-1] == pytest.approx(1.0)
        assert np.all(np.isfinite(r["deletion_curve"])) and np.all(np.isfinite(r["insertion_curve"]))
    r1 = metrics.deletion_insertion_curves(p.true_function, p.X[:20], p.X[100:300], p.true_membership, order, mode="marginal", random_state=3)
    r2 = metrics.deletion_insertion_curves(p.true_function, p.X[:20], p.X[100:300], p.true_membership, order, mode="marginal", random_state=3)
    assert r1["deletion_auc"] == r2["deletion_auc"] and r1["insertion_auc"] == r2["insertion_auc"]


def test_curves_use_a_common_scale_across_instances():
    # Additive model with a zero background: removing a concept sets its feature to zero, so the
    # curves are known in closed form. The second instance has a total effect of zero with two
    # large contributions of opposite sign; a per instance normalization would let it dominate.
    beta = np.array([1.0, 1.0])
    X = np.array([[2.0, 0.5], [1.0, -1.0], [-0.5, -2.0]])
    bg = np.zeros((5, 2))
    U = np.eye(2)
    c = X * beta
    order = np.argsort(-np.abs(c), axis=1)
    r = metrics.deletion_insertion_curves(lambda Z: Z @ beta, X, bg, U, order, mode="conditional")
    total = c.sum(axis=1)
    scale = np.abs(total).mean()
    sign = np.where(total >= 0, 1.0, -1.0)
    first = c[np.arange(3), order[:, 0]]
    del_curve = np.stack([total, total - first, np.zeros(3)], axis=1) * sign[:, None] / scale
    ins_curve = np.stack([np.zeros(3), first, total], axis=1) * sign[:, None] / scale
    np.testing.assert_allclose(r["deletion_curve"], del_curve.mean(axis=0))
    np.testing.assert_allclose(r["insertion_curve"], ins_curve.mean(axis=0))
    area = lambda C: ((C[:, 0] + 2 * C[:, 1] + C[:, 2]) / 4).mean()  # trapezoid rule with step one half
    assert r["deletion_auc"] == pytest.approx(area(del_curve))
    assert r["insertion_auc"] == pytest.approx(area(ins_curve))
    assert r["deletion_curve"][0] == pytest.approx(1.0) and r["deletion_curve"][-1] == pytest.approx(0.0, abs=1e-12)
    assert r["insertion_curve"][0] == 0.0 and r["insertion_curve"][-1] == pytest.approx(1.0)
    # the zero effect instance contributes a bounded value, not a clipped one
    assert abs(del_curve[1, 1]) == pytest.approx(1.0 / scale)
    assert np.all(np.abs(del_curve) <= np.abs(c).sum(axis=1).max() / scale)


def test_surrogate_fidelity_checks_row_alignment():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(40, 2))
    y = Z @ np.array([1.0, 2.0])
    assert metrics.surrogate_fidelity(Z, y) > 0.99
    with pytest.raises(ValueError, match="same number of rows"):
        metrics.surrogate_fidelity(Z, np.concatenate([y, y[:20]]))
    with pytest.raises(ValueError, match="same number of rows"):
        metrics.surrogate_fidelity(Z, y[:30])
