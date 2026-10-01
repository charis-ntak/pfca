"""Fuzzification of attribution distributions.

The concept attributions obtained across resamples and model classes form an
empirical distribution which is summarized by a trapezoidal fuzzy number whose
support is given by the 5th and 95th percentiles and whose core by the 25th and
75th percentiles (Equation 2). Sign confidence, the disagreement index between
model classes, interval type 2 upgrades, linguistic labels through the fuzzy
compatibility measure (Equation 3), and the possibility degree used for fuzzy
ranking are implemented here.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Sequence

import numpy as np

BIG = 1e12
QUANTILE_LEVELS = (5.0, 25.0, 50.0, 75.0, 95.0)


@dataclass(frozen=True)
class TrapezoidalFuzzyNumber:
    """Trapezoidal fuzzy number with support [a, d] and core [b, c]."""

    a: float
    b: float
    c: float
    d: float

    def __post_init__(self):
        a, b, c, d = float(self.a), float(self.b), float(self.c), float(self.d)
        if not (a <= b + 1e-12 and b <= c + 1e-12 and c <= d + 1e-12):
            raise ValueError(f"Trapezoid parameters must be ordered a <= b <= c <= d, got {(a, b, c, d)}")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", min(max(b, a), d))
        object.__setattr__(self, "c", min(max(c, b), d))
        object.__setattr__(self, "d", d)

    # ------------------------------------------------------------------
    @classmethod
    def triangular(cls, a: float, m: float, d: float) -> "TrapezoidalFuzzyNumber":
        return cls(a, m, m, d)

    @classmethod
    def crisp(cls, v: float) -> "TrapezoidalFuzzyNumber":
        return cls(v, v, v, v)

    @property
    def support(self) -> tuple[float, float]:
        return (self.a, self.d)

    @property
    def core(self) -> tuple[float, float]:
        return (self.b, self.c)

    @property
    def parameters(self) -> tuple[float, float, float, float]:
        return (self.a, self.b, self.c, self.d)

    def membership(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        mu = np.zeros_like(z)
        a, b, c, d = self.parameters
        left = (z >= a) & (z < b)
        if b > a:
            mu = np.where(left, (z - a) / (b - a), mu)
        mu = np.where((z >= b) & (z <= c), 1.0, mu)
        right = (z > c) & (z <= d)
        if d > c:
            mu = np.where(right, (d - z) / (d - c), mu)
        return mu

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return (self.a + alpha * (self.b - self.a), self.d - alpha * (self.d - self.c))

    def centroid(self) -> float:
        """Centroid of the trapezoid, always inside the support [a, d].

        The closed form is evaluated after translating the trapezoid to its
        left endpoint a, which avoids the catastrophic cancellation of the
        raw formula when the four parameters are nearly equal but far from
        zero, as happens when pool members coincide up to floating point
        noise. A degenerate (crisp) number returns its single value.
        """
        a, b, c, d = self.parameters
        b, c, d = b - a, c - a, d - a
        den = 3.0 * (d + c - b)
        if den <= 0.0:
            return a + 0.5 * d
        cen = a + (d * d + c * c + c * d - b * b) / den
        return float(min(max(cen, self.a), self.d))

    def width(self) -> float:
        return self.d - self.a

    def scaled(self, s: float) -> "TrapezoidalFuzzyNumber":
        if s <= 0:
            raise ValueError("Scale must be positive.")
        return TrapezoidalFuzzyNumber(self.a / s, self.b / s, self.c / s, self.d / s)

    def shifted(self, t: float) -> "TrapezoidalFuzzyNumber":
        return TrapezoidalFuzzyNumber(self.a + t, self.b + t, self.c + t, self.d + t)

    def in_support(self, z: float) -> bool:
        return bool(self.a <= z <= self.d)

    def in_core(self, z: float) -> bool:
        return bool(self.b <= z <= self.c)

    def excludes_zero(self, alpha: float = 0.0) -> bool:
        lo, hi = self.alpha_cut(alpha)
        return bool(lo > 0.0 or hi < 0.0)

    def possibility_geq(self, other: "TrapezoidalFuzzyNumber") -> float:
        return possibility_degree(self, other)

    def compatibility(self, other: "TrapezoidalFuzzyNumber") -> float:
        return compatibility(self, other)

    def __repr__(self) -> str:
        return f"TFN(a={self.a:.4g}, b={self.b:.4g}, c={self.c:.4g}, d={self.d:.4g})"


def fuzzify(
    samples,
    shape: str = "trapezoidal",
    support_percentiles: tuple[float, float] = (5.0, 95.0),
    core_percentiles: tuple[float, float] = (25.0, 75.0),
) -> TrapezoidalFuzzyNumber:
    """Equation 2: trapezoidal (or triangular) fuzzy number from an empirical sample."""
    x = np.asarray(samples, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Cannot fuzzify an empty sample.")
    lo, hi = np.percentile(x, support_percentiles)
    if shape == "trapezoidal":
        b, c = np.percentile(x, core_percentiles)
    elif shape == "triangular":
        b = c = float(np.median(x))
    else:
        raise ValueError("shape must be 'trapezoidal' or 'triangular'")
    return TrapezoidalFuzzyNumber(lo, b, c, hi)


def fuzzify_array(
    samples: np.ndarray,
    axis: int = 0,
    shape: str = "trapezoidal",
    support_percentiles: tuple[float, float] = (5.0, 95.0),
    core_percentiles: tuple[float, float] = (25.0, 75.0),
) -> np.ndarray:
    """Vectorized Equation 2.

    Returns an array whose last axis holds the five quantiles
    (support low, core low, median, core high, support high). For the
    triangular shape the core values equal the median.
    """
    samples = np.asarray(samples, dtype=float)
    levels = [support_percentiles[0], core_percentiles[0], 50.0, core_percentiles[1], support_percentiles[1]]
    Q = np.percentile(samples, levels, axis=axis)
    Q = np.moveaxis(Q, 0, -1)
    if shape == "triangular":
        Q[..., 1] = Q[..., 2]
        Q[..., 3] = Q[..., 2]
    elif shape != "trapezoidal":
        raise ValueError("shape must be 'trapezoidal' or 'triangular'")
    return Q


def tfn_from_quantiles(q: Sequence[float]) -> TrapezoidalFuzzyNumber:
    """Build a fuzzy number from a five quantile vector produced by fuzzify_array."""
    q = np.asarray(q, dtype=float)
    return TrapezoidalFuzzyNumber(q[0], q[1], q[3], q[4])


def alpha_cut_array(Q: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Alpha cut of every fuzzy number stored as a five quantile vector."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    lo = Q[..., 0] + alpha * (Q[..., 1] - Q[..., 0])
    hi = Q[..., 4] - alpha * (Q[..., 4] - Q[..., 3])
    return lo, hi


def centroid_array(Q: np.ndarray) -> np.ndarray:
    """Centroid of every trapezoid stored as a five quantile vector.

    The same translated closed form as ``TrapezoidalFuzzyNumber.centroid`` is
    used, so that every value lies inside the support of its trapezoid even
    when the quantiles coincide up to floating point noise.
    """
    Q = np.asarray(Q, dtype=float)
    a = Q[..., 0]
    b, c, d = Q[..., 1] - a, Q[..., 3] - a, Q[..., 4] - a
    den = 3.0 * (d + c - b)
    degenerate = den <= 0.0
    num = d * d + c * c + c * d - b * b
    with np.errstate(invalid="ignore", divide="ignore"):
        cen = a + np.where(degenerate, 0.5 * d, num / np.where(degenerate, 1.0, den))
    return np.minimum(np.maximum(cen, Q[..., 0]), Q[..., 4])


def sign_confidence(samples: np.ndarray, axis: int = 0) -> np.ndarray:
    """Proportion of pool members whose attribution has the sign of the median."""
    samples = np.asarray(samples, dtype=float)
    med = np.median(samples, axis=axis, keepdims=True)
    sign_med = np.sign(med)
    agree = np.sign(samples) == sign_med
    agree = np.where(sign_med == 0, samples == 0, agree)
    return agree.mean(axis=axis)


def disagreement_index(values: np.ndarray) -> np.ndarray:
    """Ratio of the between model class variance to the total variance.

    ``values`` has shape (B, M, ...) with resamples on axis 0 and model classes
    on axis 1. Returns an array of shape values.shape[2:], in [0, 1]. When the
    total variance is zero the index is zero.
    """
    values = np.asarray(values, dtype=float)
    B, M = values.shape[0], values.shape[1]
    flat = values.reshape(B * M, *values.shape[2:])
    total = flat.var(axis=0)
    class_means = values.mean(axis=0)
    between = class_means.var(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.where(total > 1e-18, between / np.where(total > 1e-18, total, 1.0), 0.0)
    if M == 1:
        idx = np.zeros_like(idx)
    return np.clip(idx, 0.0, 1.0)


def compatibility(A: TrapezoidalFuzzyNumber, B: TrapezoidalFuzzyNumber) -> float:
    """Equation 3: sup_z min(mu_A(z), mu_B(z)) for two trapezoidal fuzzy numbers.

    The supremum is attained either inside the intersection of the cores or at
    the crossing of the facing linear edges, for which a closed form exists.
    """
    a1, a2, a3, a4 = A.parameters
    b1, b2, b3, b4 = B.parameters
    if a2 <= b3 and b2 <= a3:
        return 1.0
    if a3 < b2:
        left, right = (a3, a4), (b1, b2)
    else:
        left, right = (b3, b4), (a1, a2)
    lc, ld = left
    ra, rb = right
    if ld <= ra:
        return 0.0
    den = (ld - lc) + (rb - ra)
    if den <= 0:
        return 1.0
    return float(np.clip((ld - ra) / den, 0.0, 1.0))


def possibility_degree(A: TrapezoidalFuzzyNumber, B: TrapezoidalFuzzyNumber) -> float:
    """Possibility that A is at least B: sup_{x >= y} min(mu_A(x), mu_B(y))."""
    a1, a2, a3, a4 = A.parameters
    b1, b2, b3, b4 = B.parameters
    if a3 >= b2:
        return 1.0
    if a4 <= b1:
        return 0.0
    den = (a4 - a3) + (b2 - b1)
    if den <= 0:
        return 1.0
    return float(np.clip((a4 - b1) / den, 0.0, 1.0))


def possibility_matrix(numbers: Sequence[TrapezoidalFuzzyNumber]) -> np.ndarray:
    """Pairwise matrix P[i, j] = Poss(number_i >= number_j)."""
    K = len(numbers)
    P = np.ones((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                P[i, j] = possibility_degree(numbers[i], numbers[j])
    return P


def fuzzy_ranking(numbers: Sequence[TrapezoidalFuzzyNumber]) -> tuple[np.ndarray, np.ndarray]:
    """Rank fuzzy numbers by the minimum possibility of exceeding every other number.

    Returns the order (indices from most to least important) and the dominance
    degree of every number, defined as min_j Poss(F_i >= F_j).
    """
    P = possibility_matrix(numbers)
    K = len(numbers)
    if K == 1:
        return np.array([0]), np.array([1.0])
    dominance = np.array([np.min(np.delete(P[i], i)) for i in range(K)])
    centroids = np.array([f.centroid() for f in numbers])
    order = np.lexsort((-centroids, -dominance))
    return order, dominance


class IntervalType2FuzzyNumber:
    """Interval type 2 fuzzy number whose footprint of uncertainty is the union of type 1 numbers.

    The upper membership is the pointwise maximum and the lower membership the
    pointwise minimum of the member fuzzy numbers, one per model class.
    """

    def __init__(self, members: Sequence[TrapezoidalFuzzyNumber]):
        if len(members) == 0:
            raise ValueError("At least one member fuzzy number is required.")
        self.members = list(members)

    def upper_membership(self, z) -> np.ndarray:
        return np.max([m.membership(z) for m in self.members], axis=0)

    def lower_membership(self, z) -> np.ndarray:
        return np.min([m.membership(z) for m in self.members], axis=0)

    def footprint(self) -> TrapezoidalFuzzyNumber:
        """Trapezoidal envelope of the footprint (outer support, outer core)."""
        a = min(m.a for m in self.members)
        b = min(m.b for m in self.members)
        c = max(m.c for m in self.members)
        d = max(m.d for m in self.members)
        return TrapezoidalFuzzyNumber(a, min(b, c), max(b, c), d)

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        cuts = [m.alpha_cut(alpha) for m in self.members]
        return (min(c[0] for c in cuts), max(c[1] for c in cuts))

    def footprint_width(self) -> float:
        env = self.footprint()
        inner = max(min(m.d for m in self.members) - max(m.a for m in self.members), 0.0)
        return env.width() - inner

    def compatibility(self, A: TrapezoidalFuzzyNumber) -> tuple[float, float]:
        """Compatibility interval (lower, upper) with a linguistic set A."""
        vals = [compatibility(m, A) for m in self.members]
        return (float(min(vals)), float(max(vals)))

    def centroid_interval(self) -> tuple[float, float]:
        cs = [m.centroid() for m in self.members]
        return (float(min(cs)), float(max(cs)))

    def __repr__(self) -> str:
        return f"IT2FN(members={len(self.members)}, footprint={self.footprint()!r})"


class LinguisticLabelSet:
    """Fixed fuzzy sets over the standardized attribution axis.

    The default is a Ruspini partition with five labels: strongly negative,
    weakly negative, negligible, weakly positive and strongly positive, whose
    cores are centered at -2, -1, 0, 1 and 2 units of the standardization
    scale. Alternative label sets can be supplied for the sensitivity analysis.
    """

    DEFAULT = (
        ("strongly negative", TrapezoidalFuzzyNumber(-BIG, -BIG, -2.0, -1.0)),
        ("weakly negative", TrapezoidalFuzzyNumber(-2.0, -1.0, -1.0, 0.0)),
        ("negligible", TrapezoidalFuzzyNumber(-1.0, 0.0, 0.0, 1.0)),
        ("weakly positive", TrapezoidalFuzzyNumber(0.0, 1.0, 1.0, 2.0)),
        ("strongly positive", TrapezoidalFuzzyNumber(1.0, 2.0, BIG, BIG)),
    )

    def __init__(self, labels: Sequence[tuple[str, TrapezoidalFuzzyNumber]] | None = None):
        self.labels = list(labels) if labels is not None else list(self.DEFAULT)

    @classmethod
    def with_breakpoints(cls, centers: Sequence[float], names: Sequence[str] | None = None) -> "LinguisticLabelSet":
        """Ruspini partition with triangular labels at the given centers and shoulders at the ends."""
        centers = list(map(float, centers))
        if len(centers) < 2:
            raise ValueError("At least two centers are required.")
        names = list(names) if names is not None else [f"L{i + 1}" for i in range(len(centers))]
        labels = []
        for i, c in enumerate(centers):
            if i == 0:
                labels.append((names[i], TrapezoidalFuzzyNumber(-BIG, -BIG, c, centers[i + 1])))
            elif i == len(centers) - 1:
                labels.append((names[i], TrapezoidalFuzzyNumber(centers[i - 1], c, BIG, BIG)))
            else:
                labels.append((names[i], TrapezoidalFuzzyNumber(centers[i - 1], c, c, centers[i + 1])))
        return cls(labels)

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.labels]

    @staticmethod
    def _core_center(A: TrapezoidalFuzzyNumber) -> float:
        """Representative point of a label: the midpoint of its core, or the finite core end of a shoulder."""
        b, c = A.core
        b_open = (not np.isfinite(b)) or b <= -BIG
        c_open = (not np.isfinite(c)) or c >= BIG
        if b_open and c_open:
            return 0.0
        if b_open:
            return float(c)
        if c_open:
            return float(b)
        return 0.5 * (float(b) + float(c))

    @property
    def centers(self) -> np.ndarray:
        """Representative point of every label on the standardized axis, used to break ties between labels."""
        return np.array([self._core_center(A) for _, A in self.labels], dtype=float)

    def profile(self, F: TrapezoidalFuzzyNumber, scale: float = 1.0) -> dict[str, float]:
        """Equation 3 evaluated for every label on the standardized axis."""
        Fs = F.scaled(scale) if scale != 1.0 else F
        return {name: compatibility(Fs, A) for name, A in self.labels}

    def profile_type2(self, F: IntervalType2FuzzyNumber, scale: float = 1.0) -> dict[str, tuple[float, float]]:
        out = {}
        for name, A in self.labels:
            vals = [compatibility(m.scaled(scale), A) for m in F.members]
            out[name] = (float(min(vals)), float(max(vals)))
        return out

    def best_label(self, F: TrapezoidalFuzzyNumber, scale: float = 1.0) -> tuple[str, float]:
        """Label with the largest compatibility (Equation 3) and its degree.

        Equation 3 is a possibilistic measure, so every label whose core meets
        the core of F has compatibility one and a wide attribution ties for
        two or more labels. Ties are broken by the distance between the
        centroid of the scaled attribution and the representative point of
        the label (``centers``); a remaining tie goes to the first label in
        order.
        """
        prof = self.profile(F, scale)
        best = max(prof.values())
        cands = [n for n, v in prof.items() if v >= best - 1e-12]
        if len(cands) > 1:
            cen = (F.scaled(scale) if scale != 1.0 else F).centroid()
            centers = dict(zip(self.names, self.centers))
            name = min(cands, key=lambda n: abs(centers[n] - cen))
        else:
            name = cands[0]
        return name, prof[name]

    def best_label_array(self, Q: np.ndarray, scale: float = 1.0, profile: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Index and degree of the best label for every quantile vector in Q (..., 5).

        The tie breaking rule of ``best_label`` is applied to the whole array.
        ``profile`` may pass the output of ``profile_array`` for the same Q
        and scale so that the compatibilities are not recomputed. Returns two
        arrays of shape Q.shape[:-1].
        """
        Q = np.asarray(Q, dtype=float)
        prof = self.profile_array(Q, scale) if profile is None else np.asarray(profile, dtype=float)
        if prof.shape != (*Q.shape[:-1], len(self.labels)):
            raise ValueError("profile must have shape Q.shape[:-1] + (n_labels,).")
        best = prof.max(axis=-1, keepdims=True)
        tied = prof >= best - 1e-12
        dist = np.abs(centroid_array(Q / float(scale))[..., None] - self.centers)
        idx = np.argmin(np.where(tied, dist, np.inf), axis=-1)
        deg = np.take_along_axis(prof, idx[..., None], axis=-1)[..., 0]
        return idx, deg

    def profile_array(self, Q: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Compatibility of every quantile vector in Q (..., 5) with every label: shape (..., n_labels)."""
        Q = np.asarray(Q, dtype=float) / float(scale)
        flat = Q.reshape(-1, Q.shape[-1])
        out = np.zeros((flat.shape[0], len(self.labels)))
        for i, q in enumerate(flat):
            F = tfn_from_quantiles(q)
            for l, (_, A) in enumerate(self.labels):
                out[i, l] = compatibility(F, A)
        return out.reshape(*Q.shape[:-1], len(self.labels))


def standardization_scale(concept_values: np.ndarray, method: str | float = "mean_abs", reference_output: np.ndarray | None = None) -> float:
    """Scale that maps attributions to the standardized axis of the linguistic labels.

    'mean_abs' uses the mean absolute concept attribution of the reference
    pool member across instances and concepts (one unit is a typical
    contribution). 'output_sd' uses the standard deviation of the reference
    model output. A positive number is used as given; Python and numpy real
    scalars (for example float32 or int64) are accepted, booleans are not.
    """
    if isinstance(method, numbers.Real) and not isinstance(method, (bool, np.bool_)):
        if method <= 0:
            raise ValueError("A numeric scale must be positive.")
        return float(method)
    if method == "mean_abs":
        s = float(np.mean(np.abs(concept_values)))
    elif method == "output_sd":
        if reference_output is None:
            raise ValueError("reference_output is required for the 'output_sd' scale.")
        s = float(np.std(reference_output))
    else:
        raise ValueError("Unknown standardization method.")
    return s if s > 1e-12 else 1.0


def describe(F: TrapezoidalFuzzyNumber, label_set: LinguisticLabelSet, scale: float, sign_conf: float, disagreement: float, name: str = "concept") -> str:
    """One line linguistic description of a fuzzy attribution."""
    label, degree = label_set.best_label(F, scale)
    return (
        f"{name}: {label} (compatibility {degree:.2f}), centroid {F.centroid():+.3g}, "
        f"support [{F.a:+.3g}, {F.d:+.3g}], sign confidence {sign_conf:.2f}, model disagreement {disagreement:.2f}"
    )
