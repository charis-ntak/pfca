"""Regression tests for the evaluation protocol, the baselines, the budget and the Phase D statistics."""

import tracemalloc

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pfca.attribution import AttributionEngine, output_function, shapley_values
from pfca.evaluation import baselines, budget, metrics, protocol, statistics

KW = {"n_concepts_grid": (2, 3), "fuzzifier_grid": (2.0,), "alpha_grid": (0.0, 0.5)}


@pytest.fixture(scope="module")
def pool(small_problem):
    """A small pool of two tree model classes on the additive problem."""
    p, X_test = small_problem
    eng = AttributionEngine(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=30)), ("gbm2", GradientBoostingRegressor(n_estimators=10))], n_resamples=3, background_size=40, random_state=0).fit(p.X, p.y)
    return p, X_test, eng, eng.explain(X_test)


def _candidates(out):
    return {c.partition for c in out.extra["explanation"].selection.configurations}


def test_data_driven_variants_do_not_evaluate_the_apriori_partition(pool):
    p, X_test, eng, attr = pool
    groups = [list(map(int, b)) for b in p.blocks]
    for name in ("pfca", "pfca_crisp", "pfca_single_model", "pfca_fixed"):
        out = protocol.pfca_output(name, eng, attr, None, KW, groups)
        assert "apriori" not in _candidates(out), name
        assert out.extra["partition"] != "apriori"
    # the ablation evaluates the a priori partition only, and an explicit request keeps it as a candidate
    assert _candidates(protocol.pfca_output("pfca_apriori", eng, attr, None, KW, groups)) == {"apriori"}
    both = protocol.pfca_output("pfca", eng, attr, None, {**KW, "partition_source": "both"}, groups)
    assert "apriori" in _candidates(both) and len(_candidates(both)) > 1
    with pytest.raises(ValueError):
        protocol.pfca_output("pfca_apriori", eng, attr, None, KW, None)


def test_surrogate_fidelity_uses_the_retained_items(pool):
    p, X_test, eng, attr = pool
    model_fn = output_function(eng.reference_model_, "regression")
    ctx = protocol.EvaluationContext(model_fn=model_fn, X_explain=X_test, background=p.X[:100], perturbation_instances=0)
    output = model_fn(X_test)
    # an efficient feature level attribution reproduces the output exactly when every feature is retained
    res_all = protocol.evaluate(protocol.shap_output(eng, attr, "regression"), ctx)
    assert res_all["surrogate_r2"] == pytest.approx(1.0, abs=1e-6) and res_all["n_retained"] == p.n_features
    top = protocol.shap_output(eng, attr, "regression", n_retained=2)
    np.testing.assert_array_equal(top.retained, np.argsort(-top.importance, kind="stable")[:2])
    res_top = protocol.evaluate(top, ctx)
    assert res_top["n_retained"] == 2 and res_top["surrogate_r2_all"] == pytest.approx(1.0, abs=1e-6)
    assert res_top["surrogate_r2"] == pytest.approx(metrics.surrogate_fidelity(top.point[:, top.retained], output))
    assert res_top["surrogate_r2"] < 0.99
    boot = protocol.bootstrapped_shap_output(eng, attr, n_retained=3)
    assert boot.retained.size == 3 and protocol.evaluate(boot, ctx)["n_retained"] == 3
    # PFCA: the retained concepts of the knee configuration, not every concept of the partition
    out = protocol.pfca_output("pfca", eng, attr, None, {**KW, "sparsity_grid": (1,)})
    expl = out.extra["explanation"]
    np.testing.assert_array_equal(out.retained, expl.retained)
    assert out.retained.size == 1 < out.n_items
    res = protocol.evaluate(out, ctx)
    assert res["surrogate_r2"] == pytest.approx(metrics.surrogate_fidelity(out.point[:, expl.retained], output))
    assert res["surrogate_r2_all"] == pytest.approx(metrics.surrogate_fidelity(out.point, output))
    assert res["surrogate_r2"] < res["surrogate_r2_all"] and res["n_retained"] == 1
    # log loss for probability outputs
    ctx_ll = protocol.EvaluationContext(model_fn=lambda Z: 1.0 / (1.0 + np.exp(-model_fn(Z))), X_explain=X_test, background=p.X[:100], perturbation_instances=0, surrogate_metric="log_loss")
    res_ll = protocol.evaluate(top, ctx_ll)
    assert res_ll["surrogate_log_loss"] > 0 and "surrogate_r2" not in res_ll
    with pytest.raises(ValueError):
        protocol.EvaluationContext(model_fn=model_fn, X_explain=X_test, background=p.X[:100], surrogate_metric="bad")


def test_measure_times_without_tracing_and_traces_reentrantly(pool):
    assert not tracemalloc.is_tracing()
    with budget.measure() as b:
        tracing_inside = tracemalloc.is_tracing()
        _ = np.ones(10000)
    assert not tracing_inside and b.peak_python_mb == 0.0
    assert b.wall_seconds >= 0 and b.max_rss_mb > 1.0 and b.rss_increase_mb >= 0.0
    # nested tracing blocks: the inner block neither stops the tracing nor hides the peak of the outer block
    with budget.measure(trace_memory=True) as outer:
        big = bytearray(40 * 2**20)
        del big
        with budget.measure(trace_memory=True) as inner:
            small = bytearray(8 * 2**20)
            del small
        still_tracing = tracemalloc.is_tracing()
    assert still_tracing and not tracemalloc.is_tracing()
    assert 8.0 <= inner.peak_python_mb < 30.0
    assert outer.peak_python_mb >= 40.0
    # a plain block inside a tracing block leaves it untouched
    with budget.measure(trace_memory=True) as outer2:
        with budget.measure() as plain:
            x = bytearray(8 * 2**20)
            del x
    assert outer2.peak_python_mb >= 8.0 and plain.peak_python_mb == 0.0 and not tracemalloc.is_tracing()
    # tracing started by the caller is left running
    tracemalloc.start()
    try:
        with budget.measure(trace_memory=True) as b2:
            y = bytearray(4 * 2**20)
            del y
        assert tracemalloc.is_tracing() and b2.peak_python_mb >= 4.0
    finally:
        tracemalloc.stop()
    # the protocol fills the memory column only on request
    p, X_test, eng, attr = pool
    groups = [list(map(int, b)) for b in p.blocks]
    assert protocol.grouped_shap_output(eng, attr, X_test[:2], groups, "regression").peak_mb == 0.0
    assert protocol.grouped_shap_output(eng, attr, X_test[:2], groups, "regression", trace_memory=True).peak_mb > 0.0


def test_max_rss_unit_depends_on_platform(monkeypatch):
    linux = budget.max_rss_mb()
    assert 1.0 < linux < 1e6
    monkeypatch.setattr(budget.sys, "platform", "darwin")
    assert budget.max_rss_mb() == pytest.approx(linux / 1024.0)


def test_bootstrapped_shap_forwards_task_and_target_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 4))
    y = np.where(X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=150) > 0, 2, 1)  # binary labels coded 1 and 2
    clf = GradientBoostingClassifier(n_estimators=15, random_state=0)
    out2 = baselines.bootstrapped_shap(clf, X, y, X[:5], n_resamples=2, background_size=30, task="classification", target_class=2)
    assert out2["engine"].task_ == "classification" and out2["result"].explained_class == 2
    probs = out2["result"].outputs[0, 0]
    assert np.all((probs >= 0) & (probs <= 1)) and 0 < out2["expected"][0] < 1
    out1 = baselines.bootstrapped_shap(clf, X, y, X[:5], n_resamples=2, background_size=30, task="classification", target_class=1)
    assert out1["result"].explained_class == 1
    np.testing.assert_allclose(out1["pool"], -out2["pool"], atol=1e-8)
    np.testing.assert_allclose(out1["expected"], 1.0 - out2["expected"], atol=1e-8)
    with pytest.raises(ValueError):
        baselines.bootstrapped_shap(clf, X, y, X[:5], n_resamples=2, background_size=30, task="regression")


def test_grouped_shap_and_perturbation_functions_share_the_pool_background(pool):
    p, X_test, eng, attr = pool
    groups = [list(map(int, b)) for b in p.blocks]
    bg = protocol.pool_background(eng)
    assert bg.shape == (40, p.n_features)
    np.testing.assert_allclose(shapley_values(eng.reference_model_, X_test, bg, "regression", explainer="tree")[0], attr.values[0, 0], atol=1e-8)
    out = protocol.grouped_shap_output(eng, attr, X_test, groups, "regression")
    assert out.extra["expected"] == pytest.approx(attr.expected[0, 0])
    np.testing.assert_allclose(out.point.sum(axis=1), attr.values[0, 0].sum(axis=1), atol=1e-6)
    shap_out = protocol.shap_output(eng, attr, "regression")
    np.testing.assert_allclose(shap_out.attribution_fn(X_test), attr.values[0, 0], atol=1e-8)
    # an integer background size subsamples the pool background
    small = protocol.grouped_shap_output(eng, attr, X_test[:3], groups, "regression", background_size=20, random_state=1)
    bg_small = bg[np.random.default_rng(1).choice(40, size=20, replace=False)]
    g = output_function(eng.reference_model_, "regression")
    ref, expected = baselines.grouped_shapley(g, X_test[:3], bg_small, groups)
    np.testing.assert_allclose(small.point, ref, atol=1e-8)
    assert small.extra["expected"] == pytest.approx(expected)


def test_grouped_shap_gives_uncovered_features_their_own_group(pool):
    p, X_test, eng, attr = pool
    groups = [list(map(int, b)) for b in p.blocks]
    out = protocol.grouped_shap_output(eng, attr, X_test, groups[:-1], "regression")
    assert out.n_items == len(groups) and out.U.shape == (p.n_features, len(groups))
    np.testing.assert_array_equal(np.flatnonzero(out.U[:, -1]), groups[-1])
    assert np.abs(out.point[:, -1]).mean() > 0.01
    np.testing.assert_allclose(out.point.sum(axis=1), attr.values[0, 0].sum(axis=1), atol=1e-6)
    full = protocol.grouped_shap_output(eng, attr, X_test, groups, "regression")
    np.testing.assert_allclose(out.point, full.point, atol=1e-8)
    np.testing.assert_allclose(out.attribution_fn(X_test[:2]), full.point[:2], atol=1e-8)
    named = protocol.grouped_shap_output(eng, attr, X_test[:2], {"a": groups[0], "b": groups[1]}, "regression")
    assert named.n_items == 3
    g = output_function(eng.reference_model_, "regression")
    est, _ = baselines.grouped_shapley(g, X_test[:2], protocol.pool_background(eng), groups, max_exact_groups=2, n_permutations=1)
    assert np.all(np.isfinite(est))


def test_integrated_gradients_require_a_smooth_model_class(pool):
    p, X_test, eng, attr = pool
    with pytest.raises(ValueError, match="smooth model class"):
        protocol.integrated_gradients_output(eng, X_test, "regression")
    explicit = protocol.integrated_gradients_output(eng, X_test[:3], "regression", model_index=0)
    assert explicit.extra["model_class"] == "gbm"
    net = Pipeline([("scale", StandardScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(8,), max_iter=300, random_state=0))])
    eng2 = AttributionEngine(model_classes=[("gbm", GradientBoostingRegressor(n_estimators=10)), ("net", net)], n_resamples=1, background_size=40, random_state=0).fit(p.X, p.y)
    out = protocol.integrated_gradients_output(eng2, X_test[:5], "regression")
    assert out.extra["model_class"] == "net"
    g = output_function(eng2.models_[0][1], "regression")
    target = g(X_test[:5]) - g(p.X.mean(axis=0)[None, :])
    assert np.abs(out.point.sum(axis=1) - target).max() < 0.05 * np.abs(target).mean()


def test_evaluation_context_coerces_inputs_and_derives_true_importance(pool):
    p, X_test, eng, attr = pool
    model_fn = output_function(eng.reference_model_, "regression")
    true_attr = p.true_concept_attributions(X_test)
    ctx = protocol.EvaluationContext(model_fn=model_fn, X_explain=pd.DataFrame(X_test), background=pd.DataFrame(p.X[:100]), true_attr=pd.DataFrame(true_attr), U_true=p.true_membership, perturbation_instances=0)
    assert isinstance(ctx.X_explain, np.ndarray) and isinstance(ctx.background, np.ndarray) and isinstance(ctx.true_attr, np.ndarray)
    np.testing.assert_allclose(ctx.true_importance, p.true_global_importance(X_test))
    res = protocol.evaluate(protocol.shap_output(eng, attr, "regression"), ctx)
    assert np.isfinite(res["rank_recovery"]) and np.isfinite(res["deletion_auc"]) and np.isfinite(res["recovery_error"])


def test_ordinal_model_clusters_the_covariance_by_expert():
    rng = np.random.default_rng(0)
    expert = np.repeat(np.arange(10), 6)
    fmt = np.tile(["shap", "concept", "fuzzy"], 20)
    latent = rng.normal(size=10)[expert] + np.where(fmt == "fuzzy", 0.8, 0.0) + rng.normal(size=60)
    lik = pd.DataFrame({"rating": np.digitize(latent, [-1.0, -0.2, 0.5, 1.2]) + 1, "format": fmt, "expert": expert})
    assert lik["rating"].nunique() == 5
    plain = statistics.ordinal_model(lik, "rating", ["format"])
    clustered = statistics.ordinal_model(lik, "rating", ["format"], group="expert")
    assert plain.cov_type == "nonrobust" and clustered.cov_type == "cluster"
    np.testing.assert_allclose(clustered.params.values, plain.params.values, atol=1e-5)
    assert not np.allclose(clustered.bse.values[:2], plain.bse.values[:2])
    fixed = statistics.ordinal_model(lik, "rating", ["format"], group="expert", group_effects=True)
    assert fixed.params.size == plain.params.size + 9 and fixed.cov_type == "cluster"
