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


def exact_centroid(F):
    """Area weighted centroid of the left triangle, the rectangle and the right triangle."""
    a, b, c, d = F.parameters
    parts = [((b - a) / 2.0, a + 2.0 * (b - a) / 3.0), (c - b, (b + c) / 2.0), ((d - c) / 2.0, c + (d - c) / 3.0)]
    area = sum(w for w, _ in parts)
    if area == 0:
        return 0.5 * (a + d)
    return sum(w * z for w, z in parts) / area


@pytest.mark.parametrize("seed", range(20))
def test_centroid_matches_area_decomposition(seed):
    rng = np.random.default_rng(seed)
    F = TrapezoidalFuzzyNumber(*np.sort(rng.uniform(-5, 5, 4)))
    assert F.centroid() == pytest.approx(exact_centroid(F), abs=1e-12)
    assert F.a <= F.centroid() <= F.d
    G = F.shifted(1e6)
    assert G.centroid() == pytest.approx(exact_centroid(F) + 1e6, abs=1e-8)
    assert G.a <= G.centroid() <= G.d


@pytest.mark.parametrize(
    "offset, width",
    [(1000.0, 1e-9), (1e4, 1e-9), (1e5, 1e-9), (1e6, 1e-9), (1e6, 1e-6), (50.0, 1e-13), (1000.0, 1e-12), (-1000.0, 1e-9)],
)
def test_centroid_nearly_degenerate_stays_in_support(offset, width):
    """Nearly coincident quantiles must not push the defuzzified value outside [a, d]."""
    F = TrapezoidalFuzzyNumber(offset, offset, offset, offset + width)
    cen = F.centroid()
    assert F.a <= cen <= F.d
    assert cen == pytest.approx(offset + width / 3.0, abs=width * 1e-6)
    Q = np.array([offset, offset, offset, offset, offset + width])
    arr = centroid_array(Q)
    assert F.a <= arr <= F.d
    assert arr == pytest.approx(cen, abs=width * 1e-6)


@pytest.mark.parametrize("mag, spread", [(50.0, 1e-11), (1000.0, 1e-12), (1e4, 1e-11), (1e6, 1e-9)])
def test_centroid_of_two_point_pool_is_midpoint(mag, spread):
    """Pool members that coincide up to floating point noise give a centroid at their midpoint."""
    x = np.array([mag, mag + spread])
    F = fuzzify(x)
    tol = max(spread * 1e-3, 4.0 * np.spacing(mag))  # the spread may span only a few representable values
    assert F.centroid() == pytest.approx(mag + spread / 2.0, abs=tol)
    assert F.a <= F.centroid() <= F.d
    Q = fuzzify_array(x[:, None], axis=0)
    c = centroid_array(Q)
    assert c.shape == (1,)
    assert c[0] == pytest.approx(mag + spread / 2.0, abs=tol)
    assert Q[0, 0] <= c[0] <= Q[0, 4]


def test_centroid_array_inside_support_random():
    rng = np.random.default_rng(3)
    S = 1e4 + 1e-9 * rng.normal(size=(40, 6, 5))
    Q = fuzzify_array(S, axis=0)
    c = centroid_array(Q)
    assert np.all(c >= Q[..., 0]) and np.all(c <= Q[..., 4])
    for i in range(6):
        for j in range(5):
            assert c[i, j] == pytest.approx(exact_centroid(tfn_from_quantiles(Q[i, j])), abs=1e-14 * 1e4)


def test_best_label_tie_break_by_core_distance():
    """Labels tied at compatibility one are separated by the distance between centroid and label core."""
    L = LinguisticLabelSet()
    symmetric = [
        TrapezoidalFuzzyNumber(-2, -1, 1, 2),
        TrapezoidalFuzzyNumber(-1.5, -1, 1, 1.5),
        TrapezoidalFuzzyNumber(-3, -2.1, 2.1, 3),
        TrapezoidalFuzzyNumber(-1.5, -1.2, 1.3, 1.6),
    ]
    for F in symmetric:
        prof = L.profile(F)
        assert prof["weakly negative"] == 1.0 and prof["negligible"] == 1.0 and prof["weakly positive"] == 1.0
        assert L.best_label(F) == ("negligible", 1.0)
    assert L.best_label(TrapezoidalFuzzyNumber(0.6, 0.95, 2.1, 2.5)) == ("strongly positive", 1.0)
    assert L.best_label(TrapezoidalFuzzyNumber(-2.5, -2.1, -0.95, -0.6)) == ("strongly negative", 1.0)
    assert L.best_label(TrapezoidalFuzzyNumber(0.2, 0.9, 1.2, 1.4)) == ("weakly positive", 1.0)
    # the scale is applied before the distance is measured
    assert L.best_label(TrapezoidalFuzzyNumber(-4, -2, 2, 4), scale=2.0) == ("negligible", 1.0)
    # unchanged behavior when a single label attains the maximum
    assert L.best_label(TrapezoidalFuzzyNumber.crisp(2.5)) == ("strongly positive", 1.0)
    assert L.best_label(TrapezoidalFuzzyNumber.crisp(-0.9)) == ("weakly negative", pytest.approx(0.9))
    np.testing.assert_allclose(L.centers, [-2.0, -1.0, 0.0, 1.0, 2.0])
    custom = LinguisticLabelSet.with_breakpoints([-1, 0, 1], ["neg", "zero", "pos"])
    np.testing.assert_allclose(custom.centers, [-1.0, 0.0, 1.0])
    assert custom.best_label(TrapezoidalFuzzyNumber(-1.5, -1, 1, 1.5)) == ("zero", 1.0)


def test_best_label_array_matches_best_label():
    L = LinguisticLabelSet()
    rng = np.random.default_rng(4)
    S = 2.5 * rng.normal(size=(30, 7, 4))
    Q = fuzzify_array(S, axis=0)
    idx, deg = L.best_label_array(Q, scale=1.3)
    assert idx.shape == (7, 4) and deg.shape == (7, 4)
    prof = L.profile_array(Q, scale=1.3)
    idx2, deg2 = L.best_label_array(Q, scale=1.3, profile=prof)
    np.testing.assert_array_equal(idx, idx2)
    np.testing.assert_allclose(deg, deg2)
    for i in range(7):
        for j in range(4):
            name, degree = L.best_label(tfn_from_quantiles(Q[i, j]), scale=1.3)
            assert L.names[idx[i, j]] == name
            assert deg[i, j] == pytest.approx(degree)
    tied = np.array([[-2, -1, 0, 1, 2], [-3, -2.1, 0, 2.1, 3], [0.6, 0.95, 1.5, 2.1, 2.5]], dtype=float)
    idx_t, deg_t = L.best_label_array(tied)
    assert [L.names[i] for i in idx_t] == ["negligible", "negligible", "strongly positive"]
    np.testing.assert_allclose(deg_t, 1.0)


def test_standardization_scale_accepts_numpy_scalars():
    X = np.array([[1.0, -3.0]])
    for m in (np.float64(2.0), np.float32(2.0), np.int64(2), np.int32(2), 2, 2.0):
        assert standardization_scale(X, m) == 2.0
        assert isinstance(standardization_scale(X, m), float)
    with pytest.raises(ValueError):
        standardization_scale(X, np.float32(0.0))
    with pytest.raises(ValueError):
        standardization_scale(X, np.int64(-1))
    for flag in (True, np.bool_(True)):
        with pytest.raises(ValueError):
            standardization_scale(X, flag)
    with pytest.raises(ValueError):
        standardization_scale(X, "unknown")
