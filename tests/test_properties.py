"""Theoretical properties of Section 5 of the study design guide, checked numerically."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from pfca import PFCAExplainer
from pfca.attribution import AttributionEngine, aggregate_to_concepts, shapley_values
from pfca.evaluation.baselines import grouped_shapley
from pfca.utils import normalize_rows


def test_efficiency_at_concept_level(small_problem):
    """Property 1: concept attributions sum to prediction minus expected prediction for any row stochastic U."""
    p, X_test = small_problem
    eng = AttributionEngine(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=40))], n_resamples=2, background_size=50, random_state=0)
    res = eng.fit_explain(p.X, p.y, X_test)
    rng = np.random.default_rng(0)
    for K in (1, 2, 5):
        U = normalize_rows(rng.random((p.n_features, K)))
        concept = aggregate_to_concepts(res.values, U)
        np.testing.assert_allclose(concept.sum(axis=-1) + res.expected[..., None], res.outputs, atol=1e-6)


def test_reduction_to_shap(small_problem):
    """Property 2: identity partition with B = M = 1 returns the SHAP values exactly."""
    p, X_test = small_problem
    d = p.n_features
    model = GradientBoostingRegressor(n_estimators=40, random_state=0)
    ex = PFCAExplainer(model_classes=[("gbm", model)], n_resamples=1, background_size=50, apriori_groups=[[j] for j in range(d)], partition_source="apriori", alpha_grid=(0.0,), random_state=0)
    ex.fit(p.X, p.y)
    e = ex.explain(X_test)
    ref = ex.reference_model_
    bg = ex.engine_.X_train_[np.random.default_rng(ex.engine_.seeds_[0, 0] + 1).choice(p.X.shape[0], size=50, replace=False)]
    sv, _ = shapley_values(ref, X_test, bg, "regression", explainer="tree")
    np.testing.assert_allclose(e.feature_attribution(), sv, atol=1e-8)
    np.testing.assert_allclose(e.concept_centroids, sv, atol=1e-8)
    # degenerate fuzzy numbers: all quantiles equal
    assert np.allclose(e.quantiles[..., 0], e.quantiles[..., 4])
    assert np.all(e.sign_confidence == 1.0)
    assert np.all(e.disagreement == 0.0)


def test_symmetry_at_concept_level(linear_model_cls):
    """Property 3: concepts with identical aggregated contributions in every coalition receive identical attributions."""
    model = linear_model_cls([1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    bg = rng.normal(size=(200, 4))
    bg -= bg.mean(axis=0)  # zero mean background so the coalition values depend on the block sums only
    x = np.array([[1.0, 2.0, 3.0, 0.0]])  # both blocks sum to 3
    groups = [[0, 1], [2, 3]]
    g, _ = grouped_shapley(model.predict, x, bg, groups)
    assert g[0, 0] == pytest.approx(g[0, 1], abs=1e-9)
    sv, _ = shapley_values(model, x, bg, "regression", explainer="exact")
    U = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    c = aggregate_to_concepts(sv, U)
    assert c[0, 0] == pytest.approx(c[0, 1], abs=1e-9)


def test_duplication_invariance(linear_model_cls):
    """Property 4: an exact copy of a feature halves its feature level SHAP value but leaves the concept attribution unchanged."""
    rng = np.random.default_rng(1)
    bg = rng.normal(size=(300, 2))
    x = np.array([[1.5, -0.5]])
    before = linear_model_cls([2.0, 1.0])
    sv_before, _ = shapley_values(before, x, bg, "regression", explainer="exact")
    # duplicate feature 0; an equivalent model spreads the coefficient over the two copies
    bg2 = np.hstack([bg, bg[:, [0]]])
    x2 = np.array([[1.5, -0.5, 1.5]])
    after = linear_model_cls([1.0, 1.0, 1.0])
    sv_after, _ = shapley_values(after, x2, bg2, "regression", explainer="exact")
    assert sv_after[0, 0] == pytest.approx(0.5 * sv_before[0, 0], abs=1e-9)
    assert sv_after[0, 2] == pytest.approx(0.5 * sv_before[0, 0], abs=1e-9)
    U_after = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
    c_after = aggregate_to_concepts(sv_after, U_after)
    assert c_after[0, 0] == pytest.approx(sv_before[0, 0], abs=1e-9)
    assert c_after[0, 1] == pytest.approx(sv_before[0, 1], abs=1e-9)


def test_duplication_invariance_fitted_model():
    """Property 4 on a fitted model: the concept absorbing the copy changes far less than the duplicated feature."""
    from pfca.evaluation.synthetic import make_synthetic
    from pfca.evaluation.metrics import duplication_invariance

    base = make_synthetic("additive", n_samples=600, n_features=6, rho=0.6, n_factors=2, random_state=3, nonlinear=False)
    dup = make_synthetic("redundancy", n_samples=600, n_features=6, rho=0.6, n_factors=2, random_state=3, nonlinear=False, n_duplicates=1, duplicate_noise=0.0)
    Xt, _ = base.sample(40, random_state=9)
    Xt2 = np.hstack([Xt, Xt[:, [dup.duplicates[0][0]]]])
    m1 = GradientBoostingRegressor(n_estimators=150, max_depth=2, random_state=0).fit(base.X, base.y)
    m2 = GradientBoostingRegressor(n_estimators=150, max_depth=2, random_state=0).fit(dup.X, dup.y)
    sv1, _ = shapley_values(m1, Xt, base.X[:100], "regression", explainer="tree")
    sv2, _ = shapley_values(m2, Xt2, dup.X[:100], "regression", explainer="tree")
    src = dup.duplicates[0][0]
    feature_change = duplication_invariance(sv1[:, src], sv2[:, src])
    c1 = aggregate_to_concepts(sv1, base.true_membership)
    c2 = aggregate_to_concepts(sv2, dup.true_membership)
    k = int(np.argmax(base.true_membership[src]))
    concept_change = duplication_invariance(c1[:, k], c2[:, k])
    assert concept_change < feature_change


def test_monotonicity_of_front_and_selection_membership(small_problem):
    """Property 6: complexity is non decreasing in alpha for fixed sparsity and selection membership lies in [0, 1]."""
    p, X_test = small_problem
    ex = PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=40))], n_resamples=6, background_size=50, n_concepts_grid=(3,), fuzzifier_grid=(2.0,), alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0), random_state=0)
    e = ex.fit(p.X, p.y).explain(X_test)
    t = e.selection.table
    K = 3
    sub = t[t["sparsity"] == K].sort_values("alpha")
    assert np.all(np.diff(sub["complexity"].to_numpy()) >= -1e-12)
    assert np.all((e.feature_selection_membership >= 0) & (e.feature_selection_membership <= 1))
    assert np.all((e.concept_selection_membership >= 0) & (e.concept_selection_membership <= 1))
