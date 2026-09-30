import numpy as np
import pandas as pd
import pytest

from pfca.concepts import ConceptPartition
from pfca.fuzzification import fuzzify_array
from pfca.selection import (
    ExplanationConfiguration,
    ParetoSelector,
    PartitionPool,
    complexity,
    concept_selection_membership,
    fidelity_loss,
    instability,
    knee_point,
    pareto_mask,
    retained_concepts,
    selection_membership,
)


def make_pool(seed=0, P=30, n=40, K=4, noise=0.5):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, K)) * np.array([2.0, 1.0, 0.5, 0.1])
    pool = base[None] + noise * rng.normal(size=(P, n, K)) * np.array([1.0, 1.0, 1.0, 2.0])
    Q = fuzzify_array(pool, axis=0)
    G = fuzzify_array(np.abs(pool).mean(axis=1), axis=0)
    U = np.eye(K)
    part = ConceptPartition(U, "id")
    return PartitionPool(part, pool, Q, G), base.sum(axis=1)


def test_pareto_mask_and_knee():
    F = np.array([[1, 5, 0], [2, 3, 0], [3, 1, 0], [2.5, 2.5, 0], [4, 4, 0]])
    m = pareto_mask(F)
    assert m.tolist() == [True, True, True, True, False]
    k = knee_point(F, m, "utopia")
    assert k in (1, 3)
    kh = knee_point(F, m, "hyperplane")
    assert kh in (1, 3)
    assert knee_point(np.array([[1.0, 1.0]])) == 0
    with pytest.raises(ValueError):
        knee_point(F, np.zeros(5, dtype=bool))


def test_retention_monotone_in_alpha():
    """Property 6: the set passing the alpha cut is nested in alpha; complexity cannot decrease."""
    pp, _ = make_pool()
    K = pp.partition.n_concepts
    prev, prev_c = set(), -1.0
    for alpha in np.linspace(0, 1, 11):
        r, activity, passing = retained_concepts(pp.quantiles, pp.global_quantiles, alpha, K)
        cur = set(np.where(passing)[0].tolist())
        assert prev.issubset(cur)
        c = complexity(pp.partition.U, r)
        assert c >= prev_c - 1e-12
        prev, prev_c = cur, c
        assert np.all((activity >= 0) & (activity <= 1))
    # sparsity cap
    r2, _, _ = retained_concepts(pp.quantiles, pp.global_quantiles, 0.0, 2)
    assert r2.size == 2


def test_complexity_crisp_equals_count():
    U = np.eye(3)
    assert complexity(U, np.array([0, 2])) == 2.0
    assert complexity(U, np.array([])) == 0.0
    Uf = np.full((3, 3), 1 / 3)
    assert complexity(Uf, np.array([0])) == pytest.approx(2.0)


def test_fidelity_loss():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 3))
    y = Z @ np.array([1.0, 2.0, 0.0]) + 3
    assert fidelity_loss(Z, np.array([0, 1]), y) < 1e-6
    assert fidelity_loss(Z, np.array([2]), y) > 0.9
    assert fidelity_loss(Z, np.array([]), y) == 1.0
    assert fidelity_loss(Z, np.array([0]), np.ones(60)) == 0.0
    assert fidelity_loss(Z, np.array([0, 1]), y, surrogate="tree") < 0.5


def test_instability():
    same = np.tile(np.array([[3.0, 2.0, 1.0]])[:, None, :], (5, 4, 1))
    assert instability(same, np.array([0, 1, 2])) == 0.0
    rev = np.stack([same[0], same[0][:, ::-1]])
    assert instability(rev, np.array([0, 1, 2])) == pytest.approx(2.0)
    assert instability(same, np.array([0])) == 0.0


def test_selection_membership_bounds():
    rng = np.random.default_rng(1)
    partitions = {"a": ConceptPartition(np.eye(3), "a"), "b": ConceptPartition(np.array([[1, 0], [0.5, 0.5], [0, 1]]), "b")}
    configs = [ExplanationConfiguration("a", 3, 2.0, 2, 0.0), ExplanationConfiguration("b", 2, 2.0, 1, 0.0), ExplanationConfiguration("b", 2, 2.0, 2, 0.5)]
    retained = [np.array([0, 1]), np.array([1]), np.array([0, 1])]
    mask = np.array([True, True, False])
    fsm = selection_membership(configs, retained, partitions, mask)
    assert fsm.shape == (3,)
    assert np.all((fsm >= 0) & (fsm <= 1))
    np.testing.assert_allclose(fsm, [0.5, 0.75, 0.5])
    csm = concept_selection_membership(partitions["b"].U, fsm)
    assert np.all((csm >= 0) & (csm <= 1))


def test_selector_grid():
    pp, output = make_pool()
    sel = ParetoSelector(alpha_grid=(0.0, 0.5, 1.0), random_state=0)
    res = sel.run_grid({"id": pp}, output)
    assert isinstance(res.table, pd.DataFrame)
    assert res.pareto.sum() >= 1
    assert res.table.loc[res.knee, "pareto"]
    assert res.front.shape[0] == res.pareto.sum()
    assert set(res.table.columns) >= {"fidelity_loss", "complexity", "instability", "retained", "knee"}
    # the configuration retaining every concept has the lowest fidelity loss
    full = res.table[res.table["sparsity"] == 4]
    assert full["fidelity_loss"].min() <= res.table["fidelity_loss"].min() + 1e-9
    assert np.all((res.feature_selection_membership >= 0) & (res.feature_selection_membership <= 1))


def test_selector_nsga2():
    pools = {}
    pp, output = make_pool()

    def provider(K, m, source):
        return pp

    sel = ParetoSelector(solver="nsga2", nsga_pop_size=8, nsga_generations=3, random_state=0)
    res = sel.run_nsga2(provider, (4, 4), (2.0,), output)
    assert res.pareto.sum() >= 1
    assert len(res.configurations) >= 1
