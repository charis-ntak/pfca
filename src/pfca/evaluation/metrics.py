"""Evaluation metrics of Table 1 of the study design guide.

Metrics are grouped by the claim they support: correctness (attribution
recovery error, concept recovery, rank recovery), faithfulness (deletion and
insertion curves, surrogate fidelity), stability (rank stability, perturbation
stability), calibration (interval coverage, sign confidence calibration) and
robustness (duplication invariance).
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from pfca.utils import fuzzy_partition_coefficient, harden

# ----------------------------------------------------------------------
# Correctness
# ----------------------------------------------------------------------


def match_concepts_to_factors(U_est: np.ndarray, U_true: np.ndarray, one_to_one: bool = False) -> np.ndarray:
    """Map every estimated concept to a true factor by maximum membership overlap.

    Returns an integer array ``assignment`` of length K_est with the factor
    index of every concept (-1 when unmatched in one to one mode). The overlap
    matrix is O = U_est^T U_true.
    """
    O = np.asarray(U_est).T @ np.asarray(U_true)
    K, F = O.shape
    if not one_to_one:
        return np.argmax(O, axis=1)
    rows, cols = linear_sum_assignment(-O)
    assignment = -np.ones(K, dtype=int)
    assignment[rows] = cols
    return assignment


def aggregate_by_assignment(values: np.ndarray, assignment: np.ndarray, n_factors: int) -> np.ndarray:
    """Sum estimated concept attributions mapped to the same factor: (n, K) -> (n, F)."""
    values = np.asarray(values, dtype=float)
    out = np.zeros((values.shape[0], n_factors))
    for k, f in enumerate(assignment):
        if f >= 0:
            out[:, f] += values[:, k]
    return out


def attribution_recovery_error(est: np.ndarray, true: np.ndarray, U_est: np.ndarray, U_true: np.ndarray, normalize: bool = True) -> float:
    """Mean absolute error between estimated and true concept attributions after matching.

    Estimated concepts are mapped to factors by maximum membership overlap and
    their attributions are summed per factor. With ``normalize`` the error is
    divided by the mean absolute true attribution so that it is comparable
    across cells of the design.
    """
    assignment = match_concepts_to_factors(U_est, U_true)
    F = np.asarray(U_true).shape[1]
    est_f = aggregate_by_assignment(est, assignment, F)
    err = float(np.mean(np.abs(est_f - true)))
    if normalize:
        scale = float(np.mean(np.abs(true)))
        err = err / scale if scale > 1e-12 else err
    return err


def feature_level_recovery_error(phi: np.ndarray, true: np.ndarray, U_true: np.ndarray, normalize: bool = True) -> float:
    """Recovery error of a feature level attribution, aggregated over the true blocks."""
    est = np.asarray(phi, dtype=float) @ np.asarray(U_true, dtype=float)
    err = float(np.mean(np.abs(est - true)))
    if normalize:
        scale = float(np.mean(np.abs(true)))
        err = err / scale if scale > 1e-12 else err
    return err


def concept_recovery(U_est: np.ndarray, U_true: np.ndarray) -> dict[str, float]:
    """Adjusted Rand index of the hardened partition against the truth, plus the fuzzy partition coefficient."""
    return {
        "ari": float(adjusted_rand_score(harden(U_true), harden(U_est))),
        "fpc": fuzzy_partition_coefficient(U_est),
    }


def rank_recovery(est_importance: np.ndarray, true_importance: np.ndarray, U_est: np.ndarray | None = None, U_true: np.ndarray | None = None) -> float:
    """Spearman correlation between estimated and true importance rankings.

    ``est_importance`` is either the estimated attribution matrix of shape
    (n, K), whose global importance is the mean absolute attribution of every
    concept as in ``SyntheticProblem.true_global_importance``, or a vector of
    K importances. When membership matrices are given the estimated concepts
    are first mapped to the true factors by maximum overlap. With the
    attribution matrix the attributions of the concepts mapped to the same
    factor are summed per instance before the absolute value is taken, which
    is the quantity the true importance is defined on and the one used by
    ``attribution_recovery_error``. With importances alone the importances are
    summed per factor instead, which is an approximation: because
    mean |a| + mean |b| >= mean |a + b|, a factor covered by several concepts
    is inflated whenever their attributions have opposite signs for some
    instances, so the attribution matrix should be passed when available.
    """
    est = np.asarray(est_importance, dtype=float)
    if est.ndim not in (1, 2):
        raise ValueError("est_importance must be a vector of importances or an (n, K) attribution matrix")
    if U_est is not None and U_true is not None:
        assignment = match_concepts_to_factors(U_est, U_true)
        F = np.asarray(U_true).shape[1]
        if est.ndim == 2:
            est = np.abs(aggregate_by_assignment(est, assignment, F)).mean(axis=0)
        else:
            est = aggregate_by_assignment(est[None, :], assignment, F)[0]
    elif est.ndim == 2:
        est = np.abs(est).mean(axis=0)
    if est.size < 2:
        return float("nan")
    r = spearmanr(est, true_importance).correlation
    return float(r) if np.isfinite(r) else 0.0


# ----------------------------------------------------------------------
# Faithfulness
# ----------------------------------------------------------------------


def _impute(
    X_query: np.ndarray,
    weights: np.ndarray,
    background: np.ndarray,
    mode: str,
    rng: np.random.Generator | None = None,
    k_neighbors: int = 10,
    replacement: np.ndarray | None = None,
) -> np.ndarray:
    """Replace features according to removal weights in [0, 1].

    ``weights`` has shape (n, d): the degree to which every feature of every
    row is removed. With 'conditional' imputation the replacement values are
    the mean of the ``k_neighbors`` nearest background rows, measured on the
    kept features (weighted by 1 - w) after division by the standard deviation
    of every background column. Columns that are constant in the background
    carry no information about the neighbors and are left out of the
    distance, so that a query value differing from the constant cannot swamp
    the kept features; when no kept feature is informative the background
    mean is used. With 'marginal' imputation the replacement rows are
    ``replacement`` (shape (n, d)) when given, so that a caller can reuse the
    same rows across calls, and otherwise one background row drawn at random
    with ``rng`` for every query row.
    """
    n, d = X_query.shape
    if mode == "marginal":
        if replacement is None:
            if rng is None:
                raise ValueError("marginal imputation requires rng or replacement rows")
            replacement = background[rng.integers(0, background.shape[0], size=n)]
        rep = np.asarray(replacement, dtype=float)
    elif mode == "conditional":
        rep = np.empty_like(X_query)
        sd = background.std(axis=0)
        informative = sd > 1e-12
        sd = np.where(informative, sd, 1.0)
        Bz = background / sd
        k = min(k_neighbors, background.shape[0])
        for i in range(n):
            keep_w = (1.0 - weights[i]) * informative
            if keep_w.sum() <= 1e-9:
                rep[i] = background.mean(axis=0)
                continue
            diff = (Bz - X_query[i] / sd) * np.sqrt(keep_w)
            dist = (diff**2).sum(axis=1)
            nn = np.argpartition(dist, k - 1)[:k]
            rep[i] = background[nn].mean(axis=0)
    else:
        raise ValueError("mode must be 'conditional' or 'marginal'")
    return (1.0 - weights) * X_query + weights * rep


def deletion_insertion_curves(
    model_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    background: np.ndarray,
    U: np.ndarray,
    order: np.ndarray,
    mode: str = "conditional",
    random_state: int | None = 0,
    k_neighbors: int = 10,
    marginal_draws: int = 10,
) -> dict[str, float]:
    """Area under the deletion and insertion curves when concepts are removed or inserted in order.

    ``order`` has shape (n, K) (per instance ordering of concepts, most
    important first) or (K,) for a global ordering. Removal of a concept
    removes its features to the degree of their membership. The removed values
    are imputed conditionally on the kept features as the mean of the
    ``k_neighbors`` nearest background rows, or marginally from
    ``marginal_draws`` background rows drawn once per instance and reused for
    the reference value and for every step of both curves, the model output
    being averaged over the draws, so that both curves are paths between the
    fully present and the fully removed instance. Every curve is the model
    output minus the output of the fully removed instance, multiplied by the
    sign of the total effect of the instance (output of the fully present
    minus output of the fully removed instance) and divided by the mean
    absolute total effect over instances. The scale is common to all
    instances, so that the average deletion curve runs from one to zero and
    the average insertion curve from zero to one, while an instance whose
    total effect is close to zero contributes a bounded value instead of
    dominating the area. Lower deletion area and higher insertion area
    indicate a more faithful ordering.
    """
    if mode not in ("conditional", "marginal"):
        raise ValueError("mode must be 'conditional' or 'marginal'")
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    background = np.asarray(background, dtype=float)
    U = np.asarray(U, dtype=float)
    n, d = X.shape
    K = U.shape[1]
    order = np.asarray(order)
    if order.ndim == 1:
        order = np.tile(order, (n, 1))
    if mode == "marginal":
        R = max(1, int(marginal_draws))
        replacements = background[rng.integers(0, background.shape[0], size=(R, n))]

    def output(weights: np.ndarray) -> np.ndarray:
        if mode == "marginal":
            Xs = np.concatenate([_impute(X, weights, background, mode, replacement=replacements[r]) for r in range(R)], axis=0)
            return np.asarray(model_fn(Xs), dtype=float).reshape(R, n).mean(axis=0)
        return np.asarray(model_fn(_impute(X, weights, background, mode, rng, k_neighbors=k_neighbors)), dtype=float)

    f_full = np.asarray(model_fn(X), dtype=float)
    f_none = output(np.ones((n, d)))
    total = f_full - f_none
    scale = float(np.mean(np.abs(total)))
    scale = scale if scale > 1e-12 else 1.0
    sign = np.where(total >= 0, 1.0, -1.0)

    def normalize(f: np.ndarray) -> np.ndarray:
        return (f - f_none) * sign / scale

    del_curve = np.zeros((n, K + 1))
    ins_curve = np.zeros((n, K + 1))
    del_curve[:, 0] = normalize(f_full)
    ins_curve[:, 0] = 0.0
    removed = np.zeros((n, K))
    for t in range(K):
        removed[np.arange(n), order[:, t]] = 1.0
        w_del = np.clip(removed @ U.T, 0.0, 1.0)
        w_ins = 1.0 - w_del
        del_curve[:, t + 1] = normalize(output(w_del))
        ins_curve[:, t + 1] = normalize(output(w_ins))
    auc_del = np.trapezoid(del_curve, dx=1.0 / K, axis=1) if hasattr(np, "trapezoid") else np.trapz(del_curve, dx=1.0 / K, axis=1)
    auc_ins = np.trapezoid(ins_curve, dx=1.0 / K, axis=1) if hasattr(np, "trapezoid") else np.trapz(ins_curve, dx=1.0 / K, axis=1)
    return {"deletion_auc": float(np.mean(auc_del)), "insertion_auc": float(np.mean(auc_ins)), "deletion_curve": del_curve.mean(axis=0), "insertion_curve": ins_curve.mean(axis=0), "deletion_auc_per_instance": np.asarray(auc_del, dtype=float), "insertion_auc_per_instance": np.asarray(auc_ins, dtype=float)}


def surrogate_fidelity(concept_attr: np.ndarray, output: np.ndarray, cv: int = 5, metric: str = "r2", random_state: int | None = 0) -> float:
    """Cross validated R squared (or log loss for probabilities) of a linear surrogate.

    ``concept_attr`` has shape (n, K) and ``output`` shape (n,), aligned row
    by row; a different number of rows raises ValueError.
    """
    Z = np.asarray(concept_attr, dtype=float)
    y = np.asarray(output, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[0] != y.shape[0]:
        raise ValueError("concept_attr and output must have the same number of rows")
    n = Z.shape[0]
    if Z.shape[1] == 0 or n < 4:
        return float("nan")
    kf = KFold(n_splits=min(cv, n), shuffle=True, random_state=random_state)
    pred = np.zeros(n)
    for tr, te in kf.split(Z):
        pred[te] = Ridge(alpha=1e-6).fit(Z[tr], y[tr]).predict(Z[te])
    if metric == "r2":
        var = np.var(y)
        return float(1.0 - np.mean((y - pred) ** 2) / var) if var > 1e-18 else float("nan")
    if metric == "log_loss":
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        yb = np.clip(y, 0.0, 1.0)
        return float(-np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p)))
    raise ValueError("metric must be 'r2' or 'log_loss'")


# ----------------------------------------------------------------------
# Stability
# ----------------------------------------------------------------------


def rank_stability(importance: np.ndarray, max_pairs: int = 2000, random_state: int | None = 0) -> float:
    """Mean pairwise Kendall tau of importance rankings across pool members (rows)."""
    I = np.asarray(importance, dtype=float)
    P, K = I.shape
    if P < 2 or K < 2:
        return float("nan")
    pairs = list(combinations(range(P), 2))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(random_state)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_pairs, replace=False)]
    taus = []
    for i, j in pairs:
        t = kendalltau(I[i], I[j]).correlation
        taus.append(1.0 if not np.isfinite(t) and np.allclose(I[i], I[j]) else (t if np.isfinite(t) else 0.0))
    return float(np.mean(taus))


def perturbation_stability(
    attribution_fn: Callable[[np.ndarray], np.ndarray],
    model_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    sigma: float = 0.05,
    n_perturbations: int = 10,
    feature_scale: np.ndarray | None = None,
    aggregate: str = "mean",
    random_state: int | None = 0,
) -> float:
    """Change in attribution under small input perturbations relative to the change in output.

    For every instance x and perturbation x' = x + sigma * s * eps the ratio
    ||a(x') - a(x)|| / ||a(x)|| divided by (|f(x') - f(x)| / sd(f) + 0.01) is
    computed, where s is the per feature scale; the mean (or maximum) over
    perturbations is then averaged over instances. Lower values are more
    stable.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    s = np.ones(d) if feature_scale is None else np.asarray(feature_scale, dtype=float)
    A0 = np.asarray(attribution_fn(X), dtype=float)
    f0 = np.asarray(model_fn(X), dtype=float)
    sd_f = float(np.std(f0)) + 1e-12
    ratios = np.zeros((n, n_perturbations))
    for p in range(n_perturbations):
        Xp = X + sigma * s * rng.normal(size=X.shape)
        Ap = np.asarray(attribution_fn(Xp), dtype=float)
        fp = np.asarray(model_fn(Xp), dtype=float)
        da = np.linalg.norm(Ap - A0, axis=1) / (np.linalg.norm(A0, axis=1) + 1e-12)
        df = np.abs(fp - f0) / sd_f
        ratios[:, p] = da / (df + 0.01)
    per_instance = ratios.mean(axis=1) if aggregate == "mean" else ratios.max(axis=1)
    return float(np.mean(per_instance))


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------


def interval_coverage(quantiles: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Proportion of (instance, concept) pairs whose true or replicated attribution lies in the support and in the core."""
    Q = np.asarray(quantiles, dtype=float)
    T = np.asarray(truth, dtype=float)
    in_support = (T >= Q[..., 0]) & (T <= Q[..., 4])
    in_core = (T >= Q[..., 1]) & (T <= Q[..., 3])
    return {"support_coverage": float(in_support.mean()), "core_coverage": float(in_core.mean())}


def sign_confidence_calibration(stated: np.ndarray, agreement: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability diagram of stated sign confidence against observed sign agreement.

    Returns the binned table and the expected calibration error (ECE), the
    weighted mean absolute gap between stated confidence and observed frequency.
    """
    s = np.asarray(stated, dtype=float).ravel()
    a = np.asarray(agreement, dtype=float).ravel()
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "count": 0, "mean_confidence": np.nan, "observed": np.nan})
            continue
        mc, ob = float(s[m].mean()), float(a[m].mean())
        rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "count": int(m.sum()), "mean_confidence": mc, "observed": ob})
        ece += m.sum() / s.size * abs(mc - ob)
    return {"table": pd.DataFrame(rows), "ece": float(ece)}


def observed_sign_agreement(median_sign: np.ndarray, replicate_values: np.ndarray) -> np.ndarray:
    """Whether the sign of a replicated attribution agrees with the stated median sign."""
    return (np.sign(replicate_values) == np.sign(median_sign)).astype(float)


# ----------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------


def duplication_invariance(before: np.ndarray, after: np.ndarray) -> float:
    """Relative change of an attribution vector when a duplicate feature is added.

    ``before`` and ``after`` are aligned arrays (for example the concept
    attributions of the concept that absorbs the copy, or the feature level
    attribution of the duplicated feature). Returns mean |after - before| /
    mean |before|.
    """
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    scale = float(np.mean(np.abs(b)))
    return float(np.mean(np.abs(a - b)) / scale) if scale > 1e-12 else float(np.mean(np.abs(a - b)))
