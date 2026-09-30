import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor

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
