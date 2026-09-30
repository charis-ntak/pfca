import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from pfca.attribution import AttributionEngine, aggregate_to_concepts, default_model_classes, redistribute_to_features, shapley_values
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
