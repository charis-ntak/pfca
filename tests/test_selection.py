import numpy as np
import pandas as pd
import pytest

import pfca.selection as selection_module
from pfca.concepts import ConceptPartition
from pfca.fuzzification import alpha_cut_array, fuzzify_array
from pfca.selection import (
    ExplanationConfiguration,
    ParetoSelector,
    PartitionPool,
    activation_threshold,
    complexity,
    concept_selection_membership,
    concept_selection_membership_on_front,
    fidelity_loss,
    instability,
    knee_point,
    pareto_mask,
    representative_mask,
    retained_concepts,
    selection_membership,
)


def make_pool(seed=0, P=30, n=40, K=4, noise=0.5, U=None, name="id"):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, K)) * np.array([2.0, 1.0, 0.5, 0.1])
    pool = base[None] + noise * rng.normal(size=(P, n, K)) * np.array([1.0, 1.0, 1.0, 2.0])
    Q = fuzzify_array(pool, axis=0)
    G = fuzzify_array(np.abs(pool).mean(axis=1), axis=0)
    U = np.eye(K) if U is None else np.asarray(U, dtype=float)
    part = ConceptPartition(U, name)
    return PartitionPool(part, pool, Q, G), base.sum(axis=1)


def make_named_pool(U, seed, name, n=40, P=20):
    """Crisp or fuzzy partition with K = U.shape[1] concepts of decreasing importance."""
    rng = np.random.default_rng(seed)
    K = U.shape[1]
    base = rng.normal(size=(n, K)) * np.linspace(2.0, 0.2, K)
    pool = base[None] + 0.5 * rng.normal(size=(P, n, K))
    part = ConceptPartition(U, name)
    return PartitionPool(part, pool, fuzzify_array(pool, axis=0), fuzzify_array(np.abs(pool).mean(axis=1), axis=0)), base.sum(axis=1)


def front_keys(table: pd.DataFrame) -> set:
    return set(zip(table["partition"], table["retained"]))


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


def test_activation_threshold_matches_alpha_cut():
    """A number is active at alpha exactly when its support excludes zero or its threshold is below alpha."""
    rng = np.random.default_rng(5)
    Q = np.sort(rng.normal(size=(200, 6, 5)), axis=-1)
    t, always = activation_threshold(Q)
    assert t.shape == (200, 6) and np.all(t >= 0)
    assert np.all(t[always] == 0.0)
    core_has_zero = (Q[..., 1] <= 0) & (Q[..., 3] >= 0)
    assert np.all(np.isinf(t[core_has_zero]))
    assert np.all(t[~core_has_zero & ~always] < 1.0)
    for alpha in rng.random(30):
        lo, hi = alpha_cut_array(Q, alpha)
        direct = (lo > 0) | (hi < 0)
        derived = always | (t < alpha)
        disagreement = direct != derived
        assert np.all(np.abs(t[disagreement] - alpha) < 1e-9)
    # closed forms
    t1, a1 = activation_threshold(np.array([[-0.1, 0.3, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5, 0.6], [-2.0, -1.0, 0.0, 1.0, 3.0], [-3.0, -2.0, -1.5, -1.0, 1.0]]))
    np.testing.assert_allclose(t1, [0.25, 0.0, np.inf, 0.5])
    assert a1.tolist() == [False, True, False, False]


def test_retention_monotone_in_alpha():
    """Property 6: the passing set and the retained set are nested in alpha for every sparsity; complexity cannot decrease."""
    U = np.array([[0.7, 0.2, 0.1, 0.0], [0.1, 0.8, 0.1, 0.0], [0.0, 0.3, 0.6, 0.1], [0.05, 0.05, 0.1, 0.8]])
    for seed in range(4):
        pp, _ = make_pool(seed=seed, U=U, noise=0.9)
        K = pp.partition.n_concepts
        for sparsity in range(1, K + 1):
            prev, prev_r, prev_c = set(), set(), -1.0
            for alpha in np.linspace(0, 1, 21):
                r, activity, passing = retained_concepts(pp.quantiles, pp.global_quantiles, alpha, sparsity)
                cur = set(np.where(passing)[0].tolist())
                assert prev.issubset(cur)
                cur_r = set(r.tolist())
                assert prev_r.issubset(cur_r), (seed, sparsity, alpha)
                assert len(cur_r) <= sparsity
                c = complexity(pp.partition.U, r)
                assert c >= prev_c - 1e-12
                prev, prev_r, prev_c = cur, cur_r, c
                assert np.all((activity >= 0) & (activity <= 1))
    # sparsity cap
    pp, _ = make_pool()
    r2, _, _ = retained_concepts(pp.quantiles, pp.global_quantiles, 0.0, 2)
    assert r2.size == 2


def test_retention_nested_under_binding_cap():
    """A concept entering at a higher alpha cannot displace a retained one, so complexity does not drop."""
    U = np.array([[0.6, 0.4], [0.6, 0.4], [0.0, 1.0], [0.0, 1.0]])
    # concept 0: narrow support excluding zero; concept 1: wide support crossing zero but larger importance
    Q = np.zeros((1, 2, 5))
    Q[0, 0] = [0.2, 0.3, 0.4, 0.5, 0.6]
    Q[0, 1] = [-0.5, 0.5, 1.0, 1.5, 2.0]
    G = np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.5, 0.8, 1.0, 1.2, 1.5]])
    r_low, _, passing_low = retained_concepts(Q, G, 0.5, 1)
    r_high, _, passing_high = retained_concepts(Q, G, 0.75, 1)
    assert passing_low.tolist() == [True, False] and passing_high.tolist() == [True, True]
    assert r_low.tolist() == [0] and r_high.tolist() == [0]
    assert complexity(U, r_high) >= complexity(U, r_low)
    # with room for both, the concept that became active first still ranks first
    r_two, _, _ = retained_concepts(Q, G, 0.75, 2)
    assert r_two.tolist() == [0, 1]
    # a support that excludes zero ranks ahead of one that touches zero, whatever the importance
    Q2 = np.zeros((1, 2, 5))
    Q2[0, 0] = [0.1, 0.2, 0.3, 0.4, 0.5]
    Q2[0, 1] = [0.0, 0.2, 1.0, 1.5, 2.0]
    G2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.8, 1.0, 1.2, 1.5]])
    assert retained_concepts(Q2, G2, 0.0, 1)[0].tolist() == [0]
    assert retained_concepts(Q2, G2, 0.5, 1)[0].tolist() == [0]
    assert retained_concepts(Q2, G2, 0.5, 2)[0].tolist() == [0, 1]


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
    with pytest.raises(ValueError):
        fidelity_loss(Z, np.array([]), y, surrogate="foo")


def test_instability():
    same = np.tile(np.array([[3.0, 2.0, 1.0]])[:, None, :], (5, 4, 1))
    assert instability(same, np.array([0, 1, 2])) == 0.0
    rev = np.stack([same[0], same[0][:, ::-1]])
    assert instability(rev, np.array([0, 1, 2])) == pytest.approx(2.0)
    assert instability(same, np.array([0])) == 0.0


def test_selection_membership_bounds():
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
    # Section 3.5 membership on the same front: indicator for own partition, overlap degree across partitions
    on_front = concept_selection_membership_on_front(configs, retained, partitions, mask)
    np.testing.assert_allclose(on_front["a"], [0.5, 0.75, 0.5])
    np.testing.assert_allclose(on_front["b"], [0.5, (1.0 + 1.0 / 3.0) / 2.0])


def test_concept_selection_membership_is_proportion_of_front():
    """Section 3.5: the proportion of Pareto optimal explanations retaining the concept, also for fuzzy partitions."""
    U = np.array([[1, 0], [0.5, 0.5], [0, 1]])
    parts = {"b": ConceptPartition(U, "b")}
    configs = [ExplanationConfiguration("b", 2, 2.0, 1, a) for a in (0.0, 0.5, 1.0)]
    retained = [np.array([0]), np.array([0]), np.array([0])]
    mask = np.ones(3, dtype=bool)
    fsm = selection_membership(configs, retained, parts, mask)
    np.testing.assert_allclose(fsm, [1.0, 0.5, 0.0])
    # the round trip through a fuzzy U mixes the concepts, the front based value does not
    assert not np.allclose(concept_selection_membership(U, fsm), [1.0, 0.0])
    np.testing.assert_allclose(concept_selection_membership_on_front(configs, retained, parts, mask)["b"], [1.0, 0.0])
    two = [np.array([0]), np.array([1])]
    np.testing.assert_allclose(concept_selection_membership_on_front(configs[:2], two, parts, mask[:2])["b"], [0.5, 0.5])
    Un = np.array([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5], [0.5, 0.5]])
    near = {"n": ConceptPartition(Un, "n")}
    cn = [ExplanationConfiguration("n", 2, 2.0, 1, 0.0), ExplanationConfiguration("n", 2, 2.0, 1, 0.5)]
    np.testing.assert_allclose(concept_selection_membership_on_front(cn, [np.array([0]), np.array([0])], near, np.ones(2, dtype=bool))["n"], [1.0, 0.0])
    # crisp partition: the front based value and the feature level round trip coincide
    I = {"i": ConceptPartition(np.eye(3), "i")}
    ci = [ExplanationConfiguration("i", 3, None, 1, 0.0), ExplanationConfiguration("i", 3, None, 2, 0.0), ExplanationConfiguration("i", 3, None, 3, 0.0)]
    ri = [np.array([0]), np.array([0, 1]), np.array([0, 2])]
    mi = np.ones(3, dtype=bool)
    fsm_i = selection_membership(ci, ri, I, mi)
    np.testing.assert_allclose(concept_selection_membership_on_front(ci, ri, I, mi)["i"], concept_selection_membership(np.eye(3), fsm_i))
    np.testing.assert_allclose(fsm_i, [1.0, 1 / 3, 1 / 3])


def test_selection_membership_counts_distinct_explanations():
    """Rows that retain the same concepts of the same partition are one explanation and are counted once."""
    parts = {"i": ConceptPartition(np.eye(3), "i")}
    configs = [ExplanationConfiguration("i", 3, None, 1, a) for a in (0.0, 0.25, 0.5, 0.75, 1.0)] + [ExplanationConfiguration("i", 3, None, 3, 0.0)]
    retained = [np.array([0])] * 5 + [np.array([0, 1, 2])]
    mask = np.ones(6, dtype=bool)
    rep = representative_mask(configs, retained, mask)
    assert rep.tolist() == [True, False, False, False, False, True]
    np.testing.assert_allclose(selection_membership(configs, retained, parts, mask), [1.0, 0.5, 0.5])
    np.testing.assert_allclose(concept_selection_membership_on_front(configs, retained, parts, mask)["i"], [1.0, 0.5, 0.5])
    # the same explanation reached with the retained concepts in another order is the same key
    other = [np.array([0])] * 5 + [np.array([2, 0, 1])]
    assert representative_mask(configs, other, mask).tolist() == rep.tolist()


def test_selector_grid():
    pp, output = make_pool()
    sel = ParetoSelector(alpha_grid=(0.0, 0.5, 1.0), random_state=0)
    res = sel.run_grid({"id": pp}, output)
    assert isinstance(res.table, pd.DataFrame)
    assert res.pareto.sum() >= 1
    assert res.table.loc[res.knee, "pareto"]
    assert res.front.shape[0] == res.pareto.sum()
    assert set(res.table.columns) >= {"fidelity_loss", "complexity", "instability", "retained", "knee", "representative"}
    # the configuration retaining every concept has the lowest fidelity loss
    full = res.table[res.table["sparsity"] == 4]
    assert full["fidelity_loss"].min() <= res.table["fidelity_loss"].min() + 1e-9
    assert np.all((res.feature_selection_membership >= 0) & (res.feature_selection_membership <= 1))
    # one representative per distinct front explanation, and the membership is computed over those rows
    distinct = res.distinct_front
    assert len(front_keys(distinct)) == len(distinct) == len(front_keys(res.front))
    assert distinct["representative"].all() and distinct["pareto"].all()
    counts = np.zeros(4)
    for r in distinct["retained"]:
        counts[list(r)] += 1
    np.testing.assert_allclose(res.concept_selection_memberships["id"], counts / len(distinct))
    np.testing.assert_allclose(res.concept_membership("id"), res.feature_selection_membership)
    # duplicating the alpha grid does not change the memberships
    res2 = ParetoSelector(alpha_grid=(0.0, 0.0, 0.5, 0.5, 1.0, 1.0), random_state=0).run_grid({"id": pp}, output)
    np.testing.assert_allclose(res2.feature_selection_membership, res.feature_selection_membership)
    np.testing.assert_allclose(res2.concept_selection_memberships["id"], res.concept_selection_memberships["id"])


def test_selector_argument_validation():
    pp, output = make_pool()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ParetoSelector(alpha_grid=(-0.5, 0.0, 1.0, 1.5))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ParetoSelector(alpha_grid=(0.0, 1.0001))
    with pytest.raises(ValueError, match="cv"):
        ParetoSelector(cv=1)
    with pytest.raises(ValueError, match="surrogate"):
        ParetoSelector(surrogate="foo")
    with pytest.raises(ValueError, match="knee_method"):
        ParetoSelector(knee_method="elbow")
    with pytest.raises(ValueError, match="No configuration to evaluate"):
        ParetoSelector(sparsity_grid=(10,)).run_grid({"id": pp}, output)
    with pytest.raises(ValueError, match="No configuration to evaluate"):
        ParetoSelector().run_grid({}, output)
    # boundary values are accepted
    ParetoSelector(alpha_grid=(0.0, 1.0), cv=2)


def test_selector_nsga2():
    pools = {}
    pp, output = make_pool()

    def provider(K, m, source):
        return pp

    sel = ParetoSelector(solver="nsga2", nsga_pop_size=8, nsga_generations=3, random_state=0)
    res = sel.run_nsga2(provider, (4, 4), (2.0,), output)
    assert res.pareto.sum() >= 1
    assert len(res.configurations) >= 1
    assert "representative" in res.table.columns


def test_nsga2_evaluates_at_the_stored_alpha(monkeypatch):
    """The alpha used for the retention rule is the rounded alpha stored in the configuration."""
    pp, output = make_pool()
    seen = []
    original = selection_module.retained_concepts

    def spy(Q, G, alpha, sparsity):
        seen.append(float(alpha))
        return original(Q, G, alpha, sparsity)

    monkeypatch.setattr(selection_module, "retained_concepts", spy)
    sel = ParetoSelector(solver="nsga2", nsga_pop_size=10, nsga_generations=4, random_state=0)
    res = sel.run_nsga2(lambda K, m, source: pp, (4, 4), (2.0,), output)
    monkeypatch.undo()
    assert len(seen) >= 1
    assert all(round(a, 3) == a for a in seen)
    for cfg, r in zip(res.configurations, res.retained_sets):
        assert round(cfg.alpha, 3) == cfg.alpha
        rebuilt, _, _ = retained_concepts(pp.quantiles, pp.global_quantiles, cfg.alpha, cfg.sparsity)
        assert np.array_equal(rebuilt, r)


def test_grid_and_nsga2_memberships_agree():
    """Memberships from the two solvers agree when their fronts contain the same explanations."""
    pp, output = make_named_pool(np.eye(4), 0, "id")
    grid = ParetoSelector(alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0), random_state=0).run_grid({"id": pp}, output)
    for seed in (0, 1):
        nsga = ParetoSelector(solver="nsga2", nsga_pop_size=20, nsga_generations=10, random_state=seed).run_nsga2(lambda K, m, source: pp, (4, 4), (2.0,), output)
        assert front_keys(grid.distinct_front) == front_keys(nsga.distinct_front)
        assert len(nsga.front) > len(nsga.distinct_front)
        np.testing.assert_allclose(nsga.feature_selection_membership, grid.feature_selection_membership)
        np.testing.assert_allclose(nsga.concept_selection_memberships["id"], grid.concept_selection_memberships["id"])


def test_nsga2_sparsity_reaches_the_size_of_an_extra_partition():
    """An a priori partition with more concepts than the FCM upper bound can be retained in full."""
    pools, output = {}, None
    for Kp in (2, 3, 5):
        U = np.zeros((10, Kp))
        for j in range(10):
            U[j, j % Kp] = 1.0
        pools[Kp], output = make_named_pool(U, Kp, "apriori" if Kp == 5 else f"fcm_K{Kp}")

    def provider(K, m, source):
        return pools[5] if source == "apriori" else pools[int(K)]

    sel = ParetoSelector(solver="nsga2", nsga_pop_size=20, nsga_generations=15, random_state=0)
    res = sel.run_nsga2(provider, (2, 3), (2.0,), output, extra_partitions=("apriori",))
    apriori = res.table[res.table["partition"] == "apriori"]
    assert apriori["sparsity"].max() == 5
    assert apriori["n_retained"].max() == 5
    assert set(res.concept_selection_memberships) == {"apriori", "fcm_K2", "fcm_K3"}
    assert res.concept_selection_memberships["apriori"].shape == (5,)
    # the FCM partitions are still clipped to their own size
    for Kp in (2, 3):
        rows = res.table[res.table["partition"] == f"fcm_K{Kp}"]
        assert rows["sparsity"].max() <= Kp
