import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from pfca.concepts import ConceptFormer, ConceptPartition, apriori_membership, consensus_membership, correlation_profile_embedding, fuzzy_c_means, identity_partition, loading_membership
from pfca.utils import concept_fuzziness, fuzzy_partition_coefficient, harden, row_entropy


def test_fcm_recovers_blocks(small_problem):
    p, _ = small_problem
    Z = correlation_profile_embedding(p.X)
    U, centers, obj = fuzzy_c_means(Z, 3, 2.0, random_state=0)
    assert U.shape == (9, 3)
    np.testing.assert_allclose(U.sum(axis=1), 1.0)
    assert adjusted_rand_score(p.block_labels, harden(U)) == 1.0
    assert centers.shape == (3, 9)
    assert obj >= 0


def test_fcm_validation():
    Z = np.eye(4)
    with pytest.raises(ValueError):
        fuzzy_c_means(Z, 4)
    with pytest.raises(ValueError):
        fuzzy_c_means(Z, 2, fuzzifier=1.0)


def test_fcm_deterministic_with_seed(small_problem):
    p, _ = small_problem
    Z = correlation_profile_embedding(p.X)
    U1, _, _ = fuzzy_c_means(Z, 3, 2.0, random_state=5)
    U2, _, _ = fuzzy_c_means(Z, 3, 2.0, random_state=5)
    np.testing.assert_allclose(U1, U2)


def test_apriori_membership_adds_other_concept():
    U = apriori_membership([[0, 1], [2]], 5)
    assert U.shape == (5, 3)
    assert U[3, 2] == 1.0 and U[4, 2] == 1.0
    U2 = apriori_membership({"a": ["f0", "f1"], "b": ["f2"]}, 3, feature_names=["f0", "f1", "f2"])
    assert U2.shape == (3, 2)


def test_loading_and_consensus(small_problem):
    p, _ = small_problem
    U = loading_membership(p.X, 3, random_state=0)
    assert U.shape == (9, 3)
    np.testing.assert_allclose(U.sum(axis=1), 1.0)
    Uc = consensus_membership(p.X, 3, n_resamples=5, random_state=0)
    assert adjusted_rand_score(p.block_labels, harden(Uc)) == 1.0


def test_concept_former_candidates(small_problem):
    p, _ = small_problem
    cf = ConceptFormer(n_concepts_grid=(2, 3, 20), fuzzifier_grid=(1.5, 2.0), apriori_groups=[list(b) for b in p.blocks], random_state=0).fit(p.X)
    names = [c.name for c in cf.candidates_]
    assert len(cf.candidates_) == 5  # 2 K values x 2 fuzzifiers + apriori (K = 20 is skipped)
    assert "apriori" in names
    part = cf.partition(3, 2.0)
    assert part is cf.partition(3, 2.0)
    assert part.n_concepts == 3
    crisp = ConceptFormer(n_concepts_grid=(3,), fuzzifier_grid=(2.0,), crisp=True, random_state=0).fit(p.X)
    assert np.all(np.isin(crisp.candidates_[0].U, [0.0, 1.0]))
    load = ConceptFormer(n_concepts_grid=(3,), distance="loading", random_state=0).fit(p.X)
    assert load.candidates_[0].method == "loading"


def test_partition_helpers():
    U = np.array([[1, 0], [0.5, 0.5], [0, 1]])
    part = ConceptPartition(U, "p", feature_names=["a", "b", "c"])
    assert part.n_concepts == 2 and part.n_features == 3
    assert part.members(0) == ["a", "b"]
    assert part.hardened().U.tolist() == [[1, 0], [1, 0], [0, 1]]
    assert part.to_frame().shape == (3, 2)
    ident = identity_partition(3)
    assert ident.n_concepts == 3
    with pytest.raises(ValueError):
        ConceptPartition(np.array([[0.5, 0.2]]), "bad")


def test_entropy_and_fuzziness():
    crisp = np.eye(3)
    assert np.allclose(row_entropy(crisp), 0.0)
    assert np.allclose(concept_fuzziness(crisp), 0.0)
    assert fuzzy_partition_coefficient(crisp) == 1.0
    uniform = np.full((3, 3), 1 / 3)
    assert np.allclose(row_entropy(uniform), 1.0)
    assert np.allclose(concept_fuzziness(uniform), 1.0)
