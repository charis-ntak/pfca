import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from pfca import PFCAExplainer, PFCAExplanation
from pfca.fuzzification import IntervalType2FuzzyNumber, TrapezoidalFuzzyNumber


@pytest.fixture(scope="module")
def fitted(small_problem):
    p, X_test = small_problem
    ex = PFCAExplainer(
        model_classes=[("gbm", GradientBoostingRegressor(n_estimators=40)), ("rf", RandomForestRegressor(n_estimators=40))],
        n_resamples=5,
        background_size=50,
        n_concepts_grid=(2, 3),
        fuzzifier_grid=(2.0,),
        alpha_grid=(0.0, 0.5),
        apriori_groups=[list(b) for b in p.blocks],
        random_state=0,
    ).fit(p.X, p.y, feature_names=[f"item{j}" for j in range(p.n_features)])
    return ex, ex.explain(X_test)


def test_outputs(fitted, small_problem):
    p, X_test = small_problem
    ex, e = fitted
    assert isinstance(e, PFCAExplanation)
    n, K = X_test.shape[0], e.n_concepts
    assert e.quantiles.shape == (n, K, 5)
    assert e.concept_pool.shape == (10, n, K)
    assert e.sign_confidence.shape == (n, K) and np.all((e.sign_confidence >= 0.5) & (e.sign_confidence <= 1))
    assert e.disagreement.shape == (n, K)
    assert e.linguistic.shape == (n, K, 5)
    assert e.possibility.shape == (K, K)
    assert set(e.ranking.tolist()) == set(range(K))
    assert e.feature_attribution().shape == (n, p.n_features)
    np.testing.assert_allclose(e.feature_attribution().sum(axis=1), e.concept_centroids.sum(axis=1))
    assert e.retained.size >= 1
    assert len(e.concept_names) == K
    assert e.partition.feature_names[0] == "item0"
    F = e.fuzzy_attribution(0, 0)
    assert isinstance(F, (TrapezoidalFuzzyNumber, IntervalType2FuzzyNumber))
    assert isinstance(e.type2_number(0, 0), IntervalType2FuzzyNumber)


def test_frames_and_text(fitted, tmp_path):
    ex, e = fitted
    df = e.instance_frame(0)
    assert isinstance(df, pd.DataFrame) and len(df) == e.n_concepts
    g = e.global_frame()
    assert list(g["rank"]) == sorted(g["rank"])
    txt = e.describe(0)
    assert "Instance 0" in txt and "compatibility" in txt
    txt_all = e.describe(0, retained_only=False)
    assert len(txt_all) >= len(txt)
    front = e.front()
    assert front["pareto"].all()
    d = e.to_dict()
    json.dumps(d)
    e.save(tmp_path / "expl.json")
    assert (tmp_path / "expl.json").exists()


def test_explanation_at_other_front_member(fitted):
    ex, e = fitted
    idx = int(np.where(e.selection.pareto)[0][0])
    e2 = ex.explanation_at(e, idx)
    assert e2.configuration == e.selection.configurations[idx]
    assert e2.quantiles.shape[0] == e.quantiles.shape[0]


def test_fixed_and_apriori_solvers(small_problem):
    p, X_test = small_problem
    groups = [list(b) for b in p.blocks]
    ex = PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=30))], n_resamples=2, background_size=40, solver="fixed", fixed_configuration={"n_concepts": 3, "fuzzifier": 2.0, "sparsity": 2, "alpha": 0.0}, random_state=0)
    e = ex.fit(p.X, p.y).explain(X_test)
    assert e.configuration.sparsity == 2 and e.n_concepts == 3
    exa = PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=30))], n_resamples=2, background_size=40, apriori_groups=groups, partition_source="apriori", random_state=0)
    ea = exa.fit(p.X, p.y).explain(X_test)
    assert ea.partition.name == "apriori"
    assert np.array_equal(ea.partition.U, p.true_membership)
    with pytest.raises(ValueError):
        PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=10))], partition_source="apriori").fit(p.X, p.y)
    with pytest.raises(ValueError):
        PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=10))], n_resamples=1, solver="fixed").fit(p.X, p.y).explain(X_test)


def test_nsga2_solver(small_problem):
    p, X_test = small_problem
    ex = PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=30))], n_resamples=3, background_size=40, n_concepts_grid=(2, 3, 4), fuzzifier_grid=(1.5, 2.0), solver="nsga2", nsga_pop_size=8, nsga_generations=3, apriori_groups=[list(b) for b in p.blocks], random_state=0)
    e = ex.fit(p.X, p.y).explain(X_test)
    assert e.selection.pareto.sum() >= 1
    assert e.n_concepts in (2, 3, 4)


def test_classification_and_dataframe_input():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(250, 6))
    X[:, 1] = X[:, 0] + 0.3 * rng.normal(size=250)
    y = (X[:, 0] + X[:, 3] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(6)])
    ex = PFCAExplainer(model_classes=[("gbm", GradientBoostingClassifier(n_estimators=30))], n_resamples=2, background_size=40, n_concepts_grid=(2, 3), fuzzifier_grid=(2.0,), random_state=0)
    e = ex.fit(df, y).explain(df.iloc[:15])
    assert ex.task_ == "classification"
    assert np.all((e.reference_output >= 0) & (e.reference_output <= 1))
    assert e.partition.feature_names == [f"f{j}" for j in range(6)]


def test_from_engine_and_explain_with(small_problem):
    from pfca.attribution import AttributionEngine

    p, X_test = small_problem
    eng = AttributionEngine(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=30))], n_resamples=3, background_size=40, random_state=0).fit(p.X, p.y)
    attr = eng.explain(X_test)
    ex = PFCAExplainer.from_engine(eng, n_concepts_grid=(3,), fuzzifier_grid=(2.0,), alpha_grid=(0.0,))
    e = ex.explain_with(attr)
    assert e.n_concepts == 3
    e_single = ex.explain_with(attr.subset(resamples=[0]))
    assert e_single.concept_pool.shape[0] == 1


# ----------------------------------------------------------------------
# regression tests for the explainer


def test_from_engine_copies_engine_settings_and_accepts_overrides(small_problem):
    from pfca.attribution import AttributionEngine

    p, X_test = small_problem
    eng = AttributionEngine(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=20))], n_resamples=2, background_size=40, kernel_nsamples=300, permutation_max_evals=200, random_state=0).fit(p.X, p.y)
    ex = PFCAExplainer.from_engine(eng, n_concepts_grid=(3,), fuzzifier_grid=(2.0,), alpha_grid=(0.0,))
    params = ex.get_params()
    assert params["kernel_nsamples"] == 300 and params["permutation_max_evals"] == 200 and params["verbose"] == eng.verbose
    assert params["random_state"] == 0 and params["n_jobs"] == eng.n_jobs
    ex2 = PFCAExplainer.from_engine(eng, random_state=3, n_jobs=1, n_concepts_grid=(3,), fuzzifier_grid=(2.0,), alpha_grid=(0.0,))
    assert ex2.random_state == 3 and ex2.engine_ is eng
    e = ex2.explain_with(eng.explain(X_test))
    assert e.n_concepts == 3


def test_classification_exposes_explained_class():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(250, 4))
    y = np.where(X[:, 0] + X[:, 1] > 0, "normal", "abnormal")
    ex = PFCAExplainer(model_classes=[("gbm", GradientBoostingClassifier(n_estimators=10))], n_resamples=1, background_size=30, n_concepts_grid=(2,), fuzzifier_grid=(2.0,), alpha_grid=(0.0,), target_class="abnormal", random_state=0).fit(X, y)
    assert list(ex.classes_) == ["abnormal", "normal"] and ex.target_class_ == 0
    e = ex.explain(X[:5])
    np.testing.assert_allclose(e.reference_output, ex.reference_model_.predict_proba(X[:5])[:, 0])
    d = e.to_dict()
    assert d["classes"] == ["abnormal", "normal"] and d["explained_class"] == "abnormal"
    json.dumps(d)


def test_single_feature_uses_identity_partition():
    rng = np.random.default_rng(0)
    X1 = rng.normal(size=(300, 1))
    y1 = 2.0 * X1[:, 0] + 0.1 * rng.normal(size=300)
    ex = PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=10))], n_resamples=2, background_size=30, random_state=0)
    with pytest.warns(UserWarning, match="n_concepts_grid"):
        ex.fit(X1, y1)
    e = ex.explain(X1[:5])
    assert e.n_concepts == 1 and e.partition.method == "identity"
    assert e.feature_attribution().shape == (5, 1)
    np.testing.assert_allclose(e.feature_attribution()[:, 0], e.concept_centroids[:, 0])


def test_two_features_with_default_grid_and_both_solvers():
    rng = np.random.default_rng(1)
    X2 = rng.normal(size=(300, 2))
    y2 = X2[:, 0] - X2[:, 1] + 0.1 * rng.normal(size=300)
    for solver in ("grid", "nsga2"):
        ex = PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=10))], n_resamples=2, background_size=30, solver=solver, nsga_pop_size=6, nsga_generations=2, random_state=0)
        with pytest.warns(UserWarning, match="n_concepts_grid"):
            ex.fit(X2, y2)
        e = ex.explain(X2[:5])
        assert e.partition.method == "identity" and e.n_concepts == 2


def test_identity_partition_when_k_equals_d(small_problem):
    """Requesting K = d gives the identity partition (Property 2) instead of a silent K = d - 1."""
    p, X_test = small_problem
    d = p.n_features
    ex = PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=20))], n_resamples=1, background_size=40, n_concepts_grid=(3, d), fuzzifier_grid=(2.0,), alpha_grid=(0.0,), random_state=0).fit(p.X, p.y)
    assert sorted(part.n_concepts for part in ex.former_.candidates_) == [3, d]
    e = ex.explain(X_test)
    table = e.selection.table
    ident = ex.explanation_at(e, int(table.index[table["partition"] == "identity"][0]))
    assert ident.n_concepts == d
    np.testing.assert_allclose(ident.feature_attribution(), e.attribution.values[0, 0], atol=1e-8)
    # a fixed configuration with K = d is honored as well
    exf = PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=20))], n_resamples=1, background_size=40, solver="fixed", fixed_configuration={"n_concepts": d, "sparsity": d, "alpha": 0.0}, random_state=0).fit(p.X, p.y)
    ef = exf.explain(X_test)
    assert ef.partition.method == "identity" and ef.n_concepts == d
    # values above d are dropped with a warning and no substitute partition is invented
    with pytest.warns(UserWarning, match=r"\[12\]"):
        exw = PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=10))], n_resamples=1, background_size=40, n_concepts_grid=(3, 12), fuzzifier_grid=(2.0,), random_state=0).fit(p.X, p.y)
    assert [part.n_concepts for part in exw.former_.candidates_] == [3]
    # NSGA II offers the identity partition as an extra source
    exn = PFCAExplainer(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=10))], n_resamples=2, background_size=40, n_concepts_grid=(2, 3, d), fuzzifier_grid=(2.0,), solver="nsga2", nsga_pop_size=8, nsga_generations=3, random_state=0).fit(p.X, p.y)
    en = exn.explain(X_test)
    assert "identity" in set(en.selection.table["partition"])


def test_single_feature_fits_and_reduces_to_shap():
    """With one feature, B = 1 and M = 1, the explainer fits on the identity partition and the concept attribution is the SHAP value."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 1))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=100)
    ex = PFCAExplainer(model_classes=[("lin", LinearRegression())], n_resamples=1, background_size=20, explainer="linear", random_state=0).fit(X[:80], y[:80])
    assert [c.method for c in ex.former_.candidates_] == ["identity"]
    e = ex.explain(X[80:])
    assert e.n_concepts == 1 and e.partition.method == "identity" and e.partition.U.tolist() == [[1.0]]
    assert e.concept_pool.shape == (1, 20, 1)
    np.testing.assert_allclose(e.feature_attribution(), e.concept_centroids)
    np.testing.assert_allclose(e.concept_pool[0], e.concept_centroids)
    assert e.retained.tolist() == [0]
