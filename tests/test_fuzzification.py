import numpy as np
import pytest

from pfca.fuzzification import (
    IntervalType2FuzzyNumber,
    LinguisticLabelSet,
    TrapezoidalFuzzyNumber,
    alpha_cut_array,
    centroid_array,
    compatibility,
    disagreement_index,
    fuzzify,
    fuzzify_array,
    fuzzy_ranking,
    possibility_degree,
    possibility_matrix,
    sign_confidence,
    standardization_scale,
    tfn_from_quantiles,
)


def brute_sup_min(A, B, lo=-10, hi=10, n=200001):
    z = np.linspace(lo, hi, n)
    return float(np.max(np.minimum(A.membership(z), B.membership(z))))


def brute_possibility(A, B, lo=-10, hi=10, n=1501):
    z = np.linspace(lo, hi, n)
    mua, mub = A.membership(z), B.membership(z)
    # sup over x >= y of min(mu_A(x), mu_B(y)) = sup_x min(mu_A(x), max_{y <= x} mu_B(y))
    cummax_b = np.maximum.accumulate(mub)
    return float(np.max(np.minimum(mua, cummax_b)))


def test_trapezoid_basics():
    F = TrapezoidalFuzzyNumber(0, 1, 2, 3)
    np.testing.assert_allclose(F.membership([-1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4]), [0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])
    assert F.alpha_cut(0) == (0, 3)
    assert F.alpha_cut(1) == (1, 2)
    assert F.alpha_cut(0.5) == (0.5, 2.5)
    assert F.centroid() == pytest.approx(1.5)
    assert F.width() == 3
    assert F.scaled(2).parameters == (0, 0.5, 1, 1.5)
    assert F.shifted(1).parameters == (1, 2, 3, 4)
    assert F.in_support(0.0) and not F.in_core(0.5)
    assert TrapezoidalFuzzyNumber.crisp(2.0).centroid() == 2.0
    assert TrapezoidalFuzzyNumber.triangular(0, 1, 2).core == (1, 1)
    with pytest.raises(ValueError):
        TrapezoidalFuzzyNumber(1, 0, 2, 3)


def test_excludes_zero_monotone_in_alpha():
    F = TrapezoidalFuzzyNumber(-0.5, 0.2, 1.0, 2.0)
    flags = [F.excludes_zero(a) for a in np.linspace(0, 1, 11)]
    assert flags == sorted(flags)  # once excluded, stays excluded as alpha grows
    assert not flags[0] and flags[-1]


def test_fuzzify_matches_percentiles():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    F = fuzzify(x)
    q = np.percentile(x, [5, 25, 75, 95])
    np.testing.assert_allclose(F.parameters, q)
    T = fuzzify(x, shape="triangular")
    assert T.b == T.c == pytest.approx(np.median(x))
    with pytest.raises(ValueError):
        fuzzify(x, shape="other")
    with pytest.raises(ValueError):
        fuzzify([])


def test_fuzzify_array_consistency():
    rng = np.random.default_rng(1)
    S = rng.normal(size=(50, 4, 3))
    Q = fuzzify_array(S, axis=0)
    assert Q.shape == (4, 3, 5)
    F = fuzzify(S[:, 2, 1])
    assert tfn_from_quantiles(Q[2, 1]).parameters == pytest.approx(F.parameters)
    lo, hi = alpha_cut_array(Q, 0.5)
    assert np.all(lo <= hi)
    c = centroid_array(Q)
    assert c[2, 1] == pytest.approx(F.centroid())
    Qt = fuzzify_array(S, axis=0, shape="triangular")
    np.testing.assert_allclose(Qt[..., 1], Qt[..., 2])


def test_consistency_of_fuzzy_summary():
    """Property 5: with B growing the trapezoid converges to the population quantiles."""
    rng = np.random.default_rng(2)
    from scipy.stats import norm

    target = norm.ppf([0.05, 0.25, 0.75, 0.95])
    errors = []
    for B in (20, 200, 20000):
        errs = [np.abs(np.array(fuzzify(rng.normal(size=B)).parameters) - target).max() for _ in range(20)]
        errors.append(np.mean(errs))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.05


@pytest.mark.parametrize("seed", range(30))
def test_compatibility_closed_form(seed):
    rng = np.random.default_rng(seed)
    A = TrapezoidalFuzzyNumber(*np.sort(rng.uniform(-5, 5, 4)))
    B = TrapezoidalFuzzyNumber(*np.sort(rng.uniform(-5, 5, 4)))
    assert compatibility(A, B) == pytest.approx(brute_sup_min(A, B), abs=2e-3)
    assert compatibility(A, B) == pytest.approx(compatibility(B, A))


@pytest.mark.parametrize("seed", range(30))
def test_possibility_closed_form(seed):
    rng = np.random.default_rng(seed)
    A = TrapezoidalFuzzyNumber(*np.sort(rng.uniform(-5, 5, 4)))
    B = TrapezoidalFuzzyNumber(*np.sort(rng.uniform(-5, 5, 4)))
    assert possibility_degree(A, B) == pytest.approx(brute_possibility(A, B), abs=2e-2)
    assert max(possibility_degree(A, B), possibility_degree(B, A)) == 1.0


def test_possibility_matrix_and_ranking():
    nums = [TrapezoidalFuzzyNumber(0, 1, 1, 2), TrapezoidalFuzzyNumber(3, 4, 4, 5), TrapezoidalFuzzyNumber(1.5, 2.5, 2.5, 3.5)]
    P = possibility_matrix(nums)
    assert P.shape == (3, 3)
    assert np.all(np.diag(P) == 1.0)
    assert P[1, 0] == 1.0 and P[0, 1] == 0.0
    order, dominance = fuzzy_ranking(nums)
    assert list(order) == [1, 2, 0]
    assert dominance[1] == 1.0


def test_sign_confidence_and_disagreement():
    S = np.array([[1, 2, -1, 3, 4]]).T.astype(float)
    assert sign_confidence(S, axis=0)[0] == pytest.approx(0.8)
    V = np.zeros((10, 2, 3))
    V[:, 0, :] = 1.0
    V[:, 1, :] = -1.0
    D = disagreement_index(V)
    assert D.shape == (3,)
    np.testing.assert_allclose(D, 1.0)
    rng = np.random.default_rng(0)
    W = rng.normal(size=(50, 3, 4))
    D2 = disagreement_index(W)
    assert np.all((D2 >= 0) & (D2 <= 1))
    assert np.all(disagreement_index(W[:, :1]) == 0)


def test_type2_union():
    A = TrapezoidalFuzzyNumber(0, 1, 2, 3)
    B = TrapezoidalFuzzyNumber(2, 3, 4, 5)
    T = IntervalType2FuzzyNumber([A, B])
    z = np.linspace(-1, 6, 100)
    np.testing.assert_allclose(T.upper_membership(z), np.maximum(A.membership(z), B.membership(z)))
    np.testing.assert_allclose(T.lower_membership(z), np.minimum(A.membership(z), B.membership(z)))
    assert T.footprint().support == (0, 5)
    assert T.alpha_cut(0) == (0, 5)
    lo, hi = T.compatibility(TrapezoidalFuzzyNumber(4, 4.5, 4.5, 6))
    assert lo <= hi
    assert T.centroid_interval() == (A.centroid(), B.centroid())
    with pytest.raises(ValueError):
        IntervalType2FuzzyNumber([])


def test_linguistic_labels_ruspini():
    L = LinguisticLabelSet()
    z = np.linspace(-3, 3, 61)
    total = sum(A.membership(z) for _, A in L.labels)
    np.testing.assert_allclose(total, 1.0)
    assert L.best_label(TrapezoidalFuzzyNumber.crisp(2.5)) == ("strongly positive", 1.0)
    assert L.best_label(TrapezoidalFuzzyNumber.crisp(0.0)) == ("negligible", 1.0)
    prof = L.profile(TrapezoidalFuzzyNumber(0.4, 0.6, 0.8, 1.0))
    assert prof["negligible"] == pytest.approx(0.6, abs=1e-9) or prof["weakly positive"] >= 0.5
    custom = LinguisticLabelSet.with_breakpoints([-1, 0, 1], ["neg", "zero", "pos"])
    assert custom.names == ["neg", "zero", "pos"]
    Q = np.array([[[0, 0, 0, 0, 0], [2, 2.2, 2.4, 2.6, 3]]], dtype=float)
    arr = L.profile_array(Q)
    assert arr.shape == (1, 2, 5)
    assert arr[0, 0, 2] == 1.0 and arr[0, 1, 4] == 1.0
    assert standardization_scale(np.array([[1.0, -3.0]]), "mean_abs") == 2.0
    assert standardization_scale(None, 2.5) == 2.5
    assert standardization_scale(np.zeros((2, 2)), "output_sd", np.array([0.0, 2.0])) == 1.0
