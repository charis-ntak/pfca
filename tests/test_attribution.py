import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

from pfca.attribution import TREE_TYPES, AttributionEngine, aggregate_to_concepts, choose_explainer, default_model_classes, redistribute_to_features, shapley_values
from pfca.utils import normalize_rows


def test_default_model_classes():
    for task in ("regression", "classification"):
        mc = default_model_classes(task)
        assert len(mc) == 3
        assert mc[0][0] == "gradient_boosting"
    with pytest.raises(ValueError):
        default_model_classes("other")


def test_engine_shapes_and_efficiency(small_problem):
    p, X_test = small_problem
    eng = AttributionEngine(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=30))], n_resamples=3, background_size=40, random_state=0)
    res = eng.fit_explain(p.X, p.y, X_test)
    assert res.values.shape == (3, 1, X_test.shape[0], p.n_features)
    assert res.expected.shape == (3, 1)
    assert res.outputs.shape == (3, 1, X_test.shape[0])
    # resample 0 is the original data
    assert np.array_equal(res.resample_indices[0], np.arange(p.X.shape[0]))
    # efficiency with respect to the background mean for the interventional tree explainer
    for b in range(3):
        np.testing.assert_allclose(res.values[b, 0].sum(axis=1) + res.expected[b, 0], res.outputs[b, 0], atol=1e-6)


def test_engine_classification_probability():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    eng = AttributionEngine(model_classes=[("gbm", GradientBoostingClassifier(n_estimators=30))], n_resamples=2, background_size=40, random_state=0)
    res = eng.fit_explain(X, y, X[:10])
    assert eng.task_ == "classification"
    assert np.all((res.outputs >= 0) & (res.outputs <= 1))
    np.testing.assert_allclose(res.values[0, 0].sum(axis=1) + res.expected[0, 0], res.outputs[0, 0], atol=1e-6)


def test_linear_and_exact_explainers_agree():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    beta = np.array([1.0, -2.0, 0.5, 0.0])
    y = X @ beta
    model = LinearRegression().fit(X, y)
    bg = X[:50]
    v_lin, e_lin = shapley_values(model, X[:5], bg, "regression", explainer="linear")
    v_ex, e_ex = shapley_values(model, X[:5], bg, "regression", explainer="exact")
    np.testing.assert_allclose(v_lin, v_ex, atol=1e-6)
    np.testing.assert_allclose(e_lin, e_ex, atol=1e-6)
    # for a linear model the Shapley value is beta_j (x_j - mean background_j)
    np.testing.assert_allclose(v_ex, beta * (X[:5] - bg.mean(axis=0)), atol=1e-6)


def test_aggregate_preserves_total():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(4, 2, 7, 6))
    U = normalize_rows(rng.random((6, 3)))
    concept = aggregate_to_concepts(values, U)
    assert concept.shape == (4, 2, 7, 3)
    np.testing.assert_allclose(concept.sum(axis=-1), values.sum(axis=-1))


def test_redistribute_identity_and_totals():
    rng = np.random.default_rng(3)
    C = rng.normal(size=(5, 4))
    np.testing.assert_allclose(redistribute_to_features(C, np.eye(4)), C)
    U = normalize_rows(rng.random((6, 4)))
    psi = redistribute_to_features(C, U)
    assert psi.shape == (5, 6)
    np.testing.assert_allclose(psi.sum(axis=1), C.sum(axis=1))


def test_subset():
    rng = np.random.default_rng(4)
    from pfca.attribution import AttributionResult

    res = AttributionResult(values=rng.normal(size=(3, 2, 4, 5)), expected=rng.normal(size=(3, 2)), outputs=rng.normal(size=(3, 2, 4)), reference_output=np.zeros(4), model_class_names=["a", "b"])
    sub = res.subset(model_classes=[1])
    assert sub.values.shape == (3, 1, 4, 5)
    assert sub.model_class_names == ["b"]
    np.testing.assert_allclose(sub.reference_output, res.outputs[0, 1])


# ----------------------------------------------------------------------
# regression tests for the attribution engine


def _efficiency_residual(res, b=0, m=0):
    return float(np.abs(res.values[b, m].sum(axis=1) + res.expected[b, m] - res.outputs[b, m]).max())


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(400, 6))
    y = 2 * X[:, 0] + X[:, 1] ** 2 - X[:, 2] + 0.1 * rng.normal(size=400)
    return X, y


def test_tree_explainer_uses_whole_background(regression_data):
    """A background of more than 100 rows is not subsampled by shap, so tree and exact explainers agree on the same background."""
    X, y = regression_data
    model = GradientBoostingRegressor(n_estimators=20, random_state=0).fit(X, y)
    v, e = shapley_values(model, X[:3], X[:250], "regression", explainer="tree")
    assert e == pytest.approx(model.predict(X[:250]).mean(), abs=1e-8)
    np.testing.assert_allclose(v.sum(axis=1) + e, model.predict(X[:3]), atol=1e-6)
    common = dict(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=20))], n_resamples=1, background_size=200, random_state=0)
    tree = AttributionEngine(explainer="tree", **common).fit_explain(X, y, X[:3])
    eng = AttributionEngine(explainer="exact", **common)
    exact = eng.fit_explain(X, y, X[:3])
    bg = X[np.random.default_rng(int(eng.seeds_[0, 0]) + 1).choice(X.shape[0], size=200, replace=False)]
    assert tree.expected[0, 0] == pytest.approx(eng.reference_model_.predict(bg).mean(), abs=1e-8)
    np.testing.assert_allclose(tree.values, exact.values, atol=1e-6)
    np.testing.assert_allclose(tree.expected, exact.expected, atol=1e-6)


def test_auto_task_follows_the_estimator(regression_data):
    """Classifiers with 1/2, -1/1 or three class codings are explained through probabilities; a regressor on a 0/1 target is regressed."""
    X, _ = regression_data
    yb = (X[:, 0] + X[:, 1] > 0).astype(int)
    for labels, target, col in (((1, 2), 2, 1), ((-1, 1), 1, 1)):
        yy = np.where(yb == 1, labels[1], labels[0])
        eng = AttributionEngine(model_classes=[("gbm", GradientBoostingClassifier(n_estimators=10))], n_resamples=1, background_size=30, target_class=target, random_state=0)
        res = eng.fit_explain(X, yy, X[:4])
        assert eng.task_ == "classification" and list(eng.classes_) == list(labels)
        assert res.explained_class == target
        np.testing.assert_allclose(res.outputs[0, 0], eng.reference_model_.predict_proba(X[:4])[:, col])
        assert _efficiency_residual(res) < 1e-6
    y3 = (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0).astype(int)
    eng = AttributionEngine(model_classes=[("gbm", GradientBoostingClassifier(n_estimators=10))], n_resamples=1, background_size=30, target_class=2, random_state=0)
    res = eng.fit_explain(X, y3, X[:4])
    assert eng.task_ == "classification"
    np.testing.assert_allclose(res.outputs[0, 0], eng.reference_model_.predict_proba(X[:4])[:, 2])
    assert _efficiency_residual(res) < 1e-6
    eng = AttributionEngine(model_classes=[("gbr", GradientBoostingRegressor(n_estimators=10))], n_resamples=1, background_size=30, random_state=0)
    res = eng.fit_explain(X, yb.astype(float), X[:4])
    assert eng.task_ == "regression" and res.classes is None and res.explained_class is None
    with pytest.raises(ValueError, match="contradicts"):
        AttributionEngine(model_classes=[("gbr", GradientBoostingRegressor(n_estimators=10))], task="classification").fit(X, yb)
    # without model classes the target decides
    assert AttributionEngine()._resolve_task(np.array([1, 2, 1, 2])) == "classification"
    assert AttributionEngine()._resolve_task(np.array([-1, 1, 1])) == "classification"
    assert AttributionEngine()._resolve_task(np.array([0.5, 1.2, 3.3])) == "regression"


def test_kernel_explainer_targets_full_shapley_vector():
    """Without the l1 preselection every attribution of a 20 feature linear model is recovered."""
    rng = np.random.default_rng(3)
    d = 20
    X = rng.normal(size=(300, d))
    beta = rng.normal(size=d)
    model = LinearRegression().fit(X, X @ beta)
    bg = X[:50]
    v, e = shapley_values(model, X[:3], bg, "regression", explainer="kernel")
    assert np.all(np.abs(v) > 1e-12)
    np.testing.assert_allclose(v, beta * (X[:3] - bg.mean(axis=0)), atol=1e-6)
    assert e == pytest.approx(model.predict(bg).mean(), abs=1e-8)


def test_kernel_explainer_reproducible_under_seed():
    """Kernel SHAP in the sampled regime depends only on the seed, and the global numpy state is left as it was."""
    rng = np.random.default_rng(4)
    d = 15
    X = rng.normal(size=(200, d))
    y = X @ rng.normal(size=d) + 0.5 * X[:, 0] * X[:, 1]
    model = GradientBoostingRegressor(n_estimators=10, random_state=0).fit(X, y)
    bg = X[:30]
    np.random.seed(123)
    v1, _ = shapley_values(model, X[:2], bg, "regression", explainer="kernel", seed=7)
    np.random.seed(999)
    v2, _ = shapley_values(model, X[:2], bg, "regression", explainer="kernel", seed=7)
    np.random.seed(999)
    v3, _ = shapley_values(model, X[:2], bg, "regression", explainer="kernel", seed=8)
    np.testing.assert_array_equal(v1, v2)
    assert not np.array_equal(v1, v3)
    np.random.seed(5)
    expected_draw = np.random.rand()
    np.random.seed(5)
    shapley_values(model, X[:1], bg, "regression", explainer="kernel", seed=7)
    assert np.random.rand() == expected_draw
    common = dict(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=10))], n_resamples=1, background_size=30, explainer="kernel", random_state=0)
    r1 = AttributionEngine(**common).fit_explain(X, y, X[:2])
    r2 = AttributionEngine(**common).fit_explain(X, y, X[:2])
    np.testing.assert_array_equal(r1.values, r2.values)


def test_hist_gradient_boosting_takes_model_agnostic_branch(regression_data):
    """Histogram based gradient boosting is not sent to TreeSHAP, which violates local accuracy for it (Property 1)."""
    X, y = regression_data
    assert "HistGradientBoostingRegressor" not in TREE_TYPES and "HistGradientBoostingClassifier" not in TREE_TYPES
    model = HistGradientBoostingRegressor(max_iter=30).fit(X, y)
    assert choose_explainer(model, X.shape[1]) == "exact"
    assert choose_explainer(model, 20) == "permutation"
    res = AttributionEngine(model_classes=[("hgb", HistGradientBoostingRegressor(max_iter=30))], n_resamples=1, background_size=40, random_state=0).fit_explain(X, y, X[:4])
    assert _efficiency_residual(res) < 1e-6


def test_multiclass_gradient_boosting_routing_and_default_classes(regression_data):
    """A multiclass GradientBoostingClassifier is not sent to TreeSHAP, so the default model classes handle three classes."""
    X, _ = regression_data
    X = X[:, :5]
    y3 = (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0).astype(int)
    gbc = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(X, y3)
    assert choose_explainer(gbc, 5) == "exact" and choose_explainer(gbc, 20) == "permutation"
    binary = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(X, (y3 > 0).astype(int))
    assert choose_explainer(binary, 5) == "tree"
    eng = AttributionEngine(n_resamples=1, background_size=30, target_class=2, random_state=0)
    res = eng.fit_explain(X, y3, X[:4])
    assert eng.model_classes_[0][0] == "gradient_boosting" and res.explained_class == 2
    np.testing.assert_allclose(res.reference_output, eng.reference_model_.predict_proba(X[:4])[:, 2])
    assert np.abs(res.values.sum(axis=-1) + res.expected[..., None] - res.outputs).max() < 1e-6


def test_string_labels_and_target_class_mapping(regression_data):
    """target_class given as a label is mapped through classes_, also for the tree branch of binary gradient boosting."""
    X, _ = regression_data
    yb = (X[:, 0] + X[:, 1] > 0).astype(int)
    ys = np.where(yb == 1, "yes", "no")
    for target, idx in (("no", 0), ("yes", 1), (1, 1), (0, 0)):
        eng = AttributionEngine(model_classes=[("gbm", GradientBoostingClassifier(n_estimators=10))], n_resamples=1, background_size=30, target_class=target, random_state=0)
        res = eng.fit_explain(X, ys, X[:4])
        assert list(eng.classes_) == ["no", "yes"] and eng.target_class_ == idx
        assert res.explained_class == ["no", "yes"][idx]
        np.testing.assert_allclose(res.outputs[0, 0], eng.reference_model_.predict_proba(X[:4])[:, idx])
        assert _efficiency_residual(res) < 1e-6
    with pytest.raises(ValueError, match="target_class"):
        AttributionEngine(model_classes=[("gbm", GradientBoostingClassifier(n_estimators=10))], target_class="maybe").fit(X, ys)
    # a random forest returns both classes from shap; the selected column follows the label as well
    eng = AttributionEngine(model_classes=[("rf", RandomForestClassifier(n_estimators=20))], n_resamples=1, background_size=30, target_class="no", random_state=0)
    res = eng.fit_explain(X, ys, X[:4])
    np.testing.assert_allclose(res.outputs[0, 0], eng.reference_model_.predict_proba(X[:4])[:, 0])
    assert _efficiency_residual(res) < 1e-6
    assert res.subset(resamples=[0]).explained_class == "no"
