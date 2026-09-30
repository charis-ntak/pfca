"""Baseline explanation methods (Section 7.1 of the guide).

Feature level KernelSHAP and TreeSHAP, grouped Shapley values with crisp a
priori groups (exact enumeration of coalitions of groups or permutation
sampling), bootstrapped SHAP with percentile intervals, LIME (optional
dependency) and integrated gradients with finite difference gradients for
models without analytic gradients. Ablations of PFCA are obtained through the
parameters of :class:`pfca.PFCAExplainer` (see :func:`ablation_explainer`).
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

import numpy as np

from pfca.attribution import AttributionEngine, output_function, shapley_values
from pfca.utils import as_index_groups, make_rng


def feature_shap(model, X_train: np.ndarray, X_explain: np.ndarray, task: str, method: str = "auto", background_size: int = 100, random_state: int | None = 0, **kwargs) -> tuple[np.ndarray, float]:
    """Feature level Shapley values of a fitted model (KernelSHAP, TreeSHAP, exact or permutation)."""
    rng = make_rng(random_state)
    X_train = np.asarray(X_train, dtype=float)
    bg = X_train[rng.choice(X_train.shape[0], size=min(background_size, X_train.shape[0]), replace=False)]
    return shapley_values(model, np.asarray(X_explain, dtype=float), bg, task, explainer=method, seed=int(rng.integers(0, 2**31 - 1)), **kwargs)


def grouped_shapley(
    model_fn: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    background: np.ndarray,
    groups,
    feature_names: Sequence[str] | None = None,
    max_exact_groups: int = 12,
    n_permutations: int = 200,
    random_state: int | None = 0,
) -> tuple[np.ndarray, float]:
    """Shapley values of crisp groups of features treated as players.

    The value of a coalition S is the mean model output over the background
    sample with the features of the groups in S set to the explained instance
    (interventional expectation). With at most ``max_exact_groups`` groups all
    coalitions are enumerated; otherwise Shapley values are estimated by
    permutation sampling with antithetic permutations.

    Returns an array of shape (n_explain, n_groups) and the expected value.
    """
    X_explain = np.asarray(X_explain, dtype=float)
    background = np.asarray(background, dtype=float)
    d = X_explain.shape[1]
    idx_groups = as_index_groups(groups, d, feature_names)
    K = len(idx_groups)
    nb = background.shape[0]
    n = X_explain.shape[0]
    expected = float(np.mean(model_fn(background)))
    out = np.zeros((n, K))
    masks = np.zeros((K, d), dtype=bool)
    for k, g in enumerate(idx_groups):
        masks[k, g] = True

    def coalition_value(x, S_mask):
        feat = masks[S_mask].any(axis=0) if S_mask.any() else np.zeros(d, dtype=bool)
        Z = background.copy()
        Z[:, feat] = x[feat]
        return float(np.mean(model_fn(Z)))

    if K <= max_exact_groups:
        from math import factorial

        coalitions = []
        for r in range(K + 1):
            coalitions.extend(combinations(range(K), r))
        weights = {}
        for S in coalitions:
            s = len(S)
            weights[s] = factorial(s) * factorial(K - s - 1) / factorial(K) if s < K else 0.0
        for i in range(n):
            x = X_explain[i]
            values = {}
            Zs = []
            for S in coalitions:
                S_mask = np.zeros(K, dtype=bool)
                S_mask[list(S)] = True
                feat = masks[S_mask].any(axis=0) if S_mask.any() else np.zeros(d, dtype=bool)
                Z = background.copy()
                Z[:, feat] = x[feat]
                Zs.append(Z)
            preds = model_fn(np.vstack(Zs)).reshape(len(coalitions), nb).mean(axis=1)
            for S, v in zip(coalitions, preds):
                values[S] = float(v)
            for k in range(K):
                phi = 0.0
                for S in coalitions:
                    if k in S:
                        continue
                    S_with = tuple(sorted(S + (k,)))
                    phi += weights[len(S)] * (values[S_with] - values[S])
                out[i, k] = phi
    else:
        rng = make_rng(random_state)
        for i in range(n):
            x = X_explain[i]
            acc = np.zeros(K)
            for _ in range(n_permutations // 2):
                perm = rng.permutation(K)
                for order in (perm, perm[::-1]):
                    S_mask = np.zeros(K, dtype=bool)
                    prev = expected
                    for k in order:
                        S_mask[k] = True
                        cur = coalition_value(x, S_mask)
                        acc[k] += cur - prev
                        prev = cur
            out[i] = acc / (2 * (n_permutations // 2))
    return out, expected


def bootstrapped_shap(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_explain: np.ndarray,
    n_resamples: int = 50,
    background_size: int = 100,
    explainer: str = "auto",
    percentiles: tuple[float, float] = (5.0, 95.0),
    n_jobs: int = 1,
    random_state: int | None = 0,
) -> dict:
    """Feature level SHAP with bootstrap percentile intervals (single model class)."""
    engine = AttributionEngine(
        model_classes=[("model", model)],
        n_resamples=n_resamples,
        background_size=background_size,
        explainer=explainer,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    res = engine.fit_explain(X_train, y_train, X_explain)
    pool = res.values[:, 0]
    lo, hi = np.percentile(pool, percentiles, axis=0)
    return {
        "point": pool[0],
        "mean": pool.mean(axis=0),
        "median": np.median(pool, axis=0),
        "lower": lo,
        "upper": hi,
        "pool": pool,
        "expected": res.expected[:, 0],
        "engine": engine,
        "result": res,
    }


def lime_attribution(model, X_train: np.ndarray, X_explain: np.ndarray, task: str, num_samples: int = 2000, random_state: int | None = 0, feature_names: Sequence[str] | None = None) -> np.ndarray:
    """LIME feature weights for every explained instance (requires the optional lime package)."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError("LIME is optional; install it with 'pip install lime'.") from exc
    X_train = np.asarray(X_train, dtype=float)
    X_explain = np.asarray(X_explain, dtype=float)
    d = X_train.shape[1]
    mode = "classification" if task == "classification" else "regression"
    expl = LimeTabularExplainer(X_train, mode=mode, feature_names=feature_names, discretize_continuous=False, random_state=random_state)
    g = output_function(model, task)
    out = np.zeros((X_explain.shape[0], d))
    for i, x in enumerate(X_explain):
        if mode == "classification":
            e = expl.explain_instance(x, model.predict_proba, num_features=d, num_samples=num_samples)
            weights = dict(e.as_map()[1])
        else:
            e = expl.explain_instance(x, lambda Z: g(Z), num_features=d, num_samples=num_samples)
            weights = dict(e.as_map()[1] if 1 in e.as_map() else next(iter(e.as_map().values())))
        for j, w in weights.items():
            out[i, j] = w
    return out


def integrated_gradients(model_fn: Callable[[np.ndarray], np.ndarray], X_explain: np.ndarray, baseline: np.ndarray, steps: int = 50, eps: float = 1e-3) -> np.ndarray:
    """Integrated gradients with central finite difference gradients along the straight path.

    Applicable to any smooth model function; intended for the multilayer
    perceptron model class. ``baseline`` is a single row (for example the
    background mean).
    """
    X_explain = np.asarray(X_explain, dtype=float)
    baseline = np.asarray(baseline, dtype=float).ravel()
    n, d = X_explain.shape
    alphas = (np.arange(steps) + 0.5) / steps
    out = np.zeros((n, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = eps
        grads = np.zeros(n)
        for a in alphas:
            P = baseline + a * (X_explain - baseline)
            grads += (model_fn(P + e) - model_fn(P - e)) / (2 * eps)
        out[:, j] = (X_explain[:, j] - baseline[j]) * grads / steps
    return out


def ablation_explainer(kind: str, base_kwargs: dict, model_classes, apriori_groups=None):
    """Build a PFCA explainer for one of the ablations of Section 7.1.

    kind : {'full', 'crisp', 'single_model', 'fixed', 'apriori'}
    """
    from pfca.explainer import PFCAExplainer

    kw = dict(base_kwargs)
    kw["model_classes"] = model_classes
    if kind == "full":
        pass
    elif kind == "crisp":
        kw["crisp_concepts"] = True
    elif kind == "single_model":
        kw["model_classes"] = model_classes[:1]
    elif kind == "fixed":
        kw["solver"] = "fixed"
        grid = list(kw.get("n_concepts_grid", (2, 3, 4, 5, 6)))
        kw["fixed_configuration"] = {"n_concepts": grid[len(grid) // 2], "fuzzifier": 2.0, "sparsity": grid[len(grid) // 2], "alpha": 0.0}
    elif kind == "apriori":
        if apriori_groups is None:
            raise ValueError("apriori ablation requires apriori_groups")
        kw["apriori_groups"] = apriori_groups
        kw["partition_source"] = "apriori"
    else:
        raise ValueError(f"Unknown ablation '{kind}'")
    return PFCAExplainer(**kw)
