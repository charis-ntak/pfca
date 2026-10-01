import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from pfca.concepts import ConceptFormer, ConceptPartition, apriori_membership, consensus_membership, correlation_matrix, correlation_profile_embedding, fuzzy_c_means, identity_partition, loading_membership
from pfca.evaluation.synthetic import make_synthetic
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


def test_loading_membership_recovers_blocks_after_rotation():
    """Unrotated maximum likelihood loadings are not identified; the varimax rotated memberships recover the known blocks."""
    for d, F, rho, seed in ((9, 3, 0.8, 2), (30, 10, 0.6, 0)):
        p = make_synthetic("additive", n_samples=300 if d == 9 else 500, n_features=d, rho=rho, n_factors=F, random_state=seed)
        U = loading_membership(p.X, F, random_state=0)
        assert U.shape == (d, F)
        np.testing.assert_allclose(U.sum(axis=1), 1.0)
        assert adjusted_rand_score(p.block_labels, harden(U)) == 1.0
        cf = ConceptFormer(n_concepts_grid=(F,), distance="loading", random_state=0).fit(p.X)
        assert adjusted_rand_score(p.block_labels, cf.candidates_[0].labels) == 1.0


def test_correlation_matrix_is_square_for_every_dimension():
    rng = np.random.default_rng(0)
    X1, X2, X9 = rng.normal(size=(200, 1)), rng.normal(size=(200, 2)), rng.normal(size=(120, 9))
    for method in ("pearson", "spearman"):
        np.testing.assert_array_equal(correlation_matrix(X1, method), np.ones((1, 1)))
        assert correlation_matrix(X2, method).shape == (2, 2)
    np.testing.assert_allclose(correlation_matrix(X2, "spearman")[0, 1], spearmanr(X2).correlation)
    np.testing.assert_allclose(correlation_matrix(X9, "spearman"), spearmanr(X9).correlation)
    Xt = np.round(X9)  # tied ranks
    np.testing.assert_allclose(correlation_matrix(Xt, "spearman"), spearmanr(Xt).correlation)
    for j in (0, 1, 5):
        Xc = X9.copy()
        Xc[:, j] = 3.0
        for method in ("pearson", "spearman"):
            R = correlation_matrix(Xc, method)
            assert R.shape == (9, 9)
            assert R[j, j] == 1.0
            assert np.all(np.delete(R[j], j) == 0.0) and np.all(np.delete(R[:, j], j) == 0.0)
    with pytest.raises(ValueError):
        correlation_matrix(X9, "kendall")


def test_concept_former_spearman_on_two_features_and_constant_feature():
    rng = np.random.default_rng(1)
    X2 = rng.normal(size=(200, 2))
    cf = ConceptFormer(n_concepts_grid=(1, 2), fuzzifier_grid=(2.0,), correlation="spearman").fit(X2)
    assert [c.name for c in cf.candidates_] == ["fcm_K1_m2"]
    Xk = rng.normal(size=(120, 9))
    Xk[:, 0] = 0.0
    cf = ConceptFormer(n_concepts_grid=(3,), fuzzifier_grid=(2.0,), correlation="spearman", random_state=0).fit(Xk)
    assert cf.embedding_.shape == (9, 9) and cf.candidates_[0].n_concepts == 3


def test_correlation_matrix_rejects_nonfinite(small_problem):
    p, _ = small_problem
    for bad in (np.nan, np.inf):
        X = p.X.copy()
        X[3, 1] = bad
        for method in ("pearson", "spearman"):
            with pytest.raises(ValueError, match="NaN or infinite"):
                correlation_matrix(X, method)
        for distance in ("correlation", "loading"):
            with pytest.raises(ValueError):
                ConceptFormer(n_concepts_grid=(3,), fuzzifier_grid=(2.0,), distance=distance, random_state=0).fit(X)


def test_fcm_fuzzifier_close_to_one(small_problem):
    p, _ = small_problem
    Z = correlation_profile_embedding(p.X)
    for m in (1.001, 1.0001):
        U, centers, obj = fuzzy_c_means(Z, 3, m, random_state=0)
        assert np.all(np.isfinite(U)) and np.all(np.isfinite(centers)) and np.isfinite(obj)
        np.testing.assert_allclose(U.sum(axis=1), 1.0)
        assert adjusted_rand_score(p.block_labels, harden(U)) == 1.0
        ConceptPartition(U, "near_crisp", fuzzifier=m)


def test_concept_former_validates_distance_and_correlation():
    with pytest.raises(ValueError, match="distance"):
        ConceptFormer(distance="cosine")
    with pytest.raises(ValueError, match="correlation"):
        ConceptFormer(correlation="kendall")


def test_partition_accepts_any_name_sequence_and_checks_length(small_problem):
    p, _ = small_problem
    df = pd.DataFrame(p.X, columns=[f"f{j}" for j in range(p.n_features)])
    assert identity_partition(p.n_features, df.columns).feature_names == list(df.columns)
    assert ConceptPartition(np.eye(3), "a", feature_names=np.array(["a", "b", "c"])).members(1) == ["b"]
    assert ConceptPartition(np.eye(3), "a", feature_names=("a", "b", "c")).feature_names == ["a", "b", "c"]
    assert ConceptPartition(np.eye(3), "a", feature_names=None).feature_names == ["x0", "x1", "x2"]
    assert identity_partition(2, None).feature_names == ["x0", "x1"]
    with pytest.raises(ValueError, match="feature_names"):
        ConceptPartition(np.eye(3), "a", feature_names=["a"])


def test_concept_former_single_feature_yields_identity():
    X1 = np.random.default_rng(0).normal(size=(50, 1))
    cf = ConceptFormer(random_state=0).fit(X1, ["only"])
    assert len(cf.candidates_) == 1
    part = cf.candidates_[0]
    assert part.method == "identity" and part.feature_names == ["only"] and part.U.tolist() == [[1.0]]
