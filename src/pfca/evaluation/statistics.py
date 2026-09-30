"""Statistical analysis (Section 7.4): paired tests across datasets, Friedman and
Nemenyi procedures, effect sizes with confidence intervals, linear mixed models
for Phase A and ordinal or binomial mixed models for Phase D (statsmodels)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm_correction(pvalues) -> np.ndarray:
    """Holm step down adjusted p values."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj


def paired_effect_sizes(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, random_state: int | None = 0) -> dict:
    """Paired Cohen's d (of the differences) and Cliff's delta with bootstrap confidence intervals."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(random_state)

    def cohen(d):
        s = d.std(ddof=1)
        return float(d.mean() / s) if s > 1e-12 else 0.0

    def cliff(x, y):
        gt = (x[:, None] > y[None, :]).mean()
        lt = (x[:, None] < y[None, :]).mean()
        return float(gt - lt)

    n = diff.size
    boots_d, boots_c = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots_d.append(cohen(diff[idx]))
        boots_c.append(cliff(a[idx], b[idx]))
    return {
        "cohen_d": cohen(diff),
        "cohen_d_ci": tuple(np.percentile(boots_d, [2.5, 97.5])),
        "cliff_delta": cliff(a, b),
        "cliff_delta_ci": tuple(np.percentile(boots_c, [2.5, 97.5])),
        "mean_difference": float(diff.mean()),
        "mean_difference_ci": tuple(np.percentile([diff[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)], [2.5, 97.5])),
    }


def wilcoxon_holm(table: pd.DataFrame, reference: str, alternative: str = "two-sided") -> pd.DataFrame:
    """Wilcoxon signed rank tests of every method against a reference across datasets, Holm corrected.

    ``table`` has one row per dataset (or replication) and one column per
    method. Returns a data frame with the statistic, raw and adjusted p values
    and effect sizes for every comparison.
    """
    others = [c for c in table.columns if c != reference]
    rows = []
    for m in others:
        a, b = table[m].to_numpy(float), table[reference].to_numpy(float)
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if a.size < 2 or np.allclose(a, b):
            stat, p = np.nan, 1.0
        else:
            res = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
            stat, p = float(res.statistic), float(res.pvalue)
        es = paired_effect_sizes(a, b)
        rows.append({"method": m, "reference": reference, "n": int(a.size), "statistic": stat, "p_value": p, **{k: v for k, v in es.items()}})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm_correction(df["p_value"].to_numpy())
    return df


def friedman_nemenyi(table: pd.DataFrame, alpha: float = 0.05, higher_is_better: bool = False) -> dict:
    """Friedman test across datasets with Nemenyi post hoc critical difference.

    Returns the Friedman statistic and p value, the average ranks of the
    methods, the critical difference and a matrix of pairwise significance.
    """
    M = table.to_numpy(float)
    N, k = M.shape
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r if higher_is_better else r), 1, M)
    avg = ranks.mean(axis=0)
    if k < 3:
        chi2, p = np.nan, np.nan
    else:
        res = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
        chi2, p = float(res.statistic), float(res.pvalue)
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) if k >= 2 else np.nan
    cd = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))) if k >= 2 else np.nan
    diff = np.abs(avg[:, None] - avg[None, :])
    sig = pd.DataFrame(diff > cd, index=table.columns, columns=table.columns)
    return {
        "friedman_statistic": chi2,
        "p_value": p,
        "average_ranks": pd.Series(avg, index=table.columns).sort_values(),
        "critical_difference": cd,
        "significant": sig,
        "n_datasets": int(N),
    }


def mixed_model(df: pd.DataFrame, response: str, fixed: list[str], group: str, reml: bool = True):
    """Linear mixed model with the given fixed effects and a random intercept per group (statsmodels)."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("statsmodels is required for mixed models; install with 'pip install statsmodels'.") from exc
    formula = f"{response} ~ " + " + ".join(fixed)
    model = smf.mixedlm(formula, df, groups=df[group])
    return model.fit(reml=reml)


def ordinal_model(df: pd.DataFrame, response: str, predictors: list[str]):
    """Ordinal (proportional odds) regression for Likert ratings (statsmodels OrderedModel)."""
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
    except ImportError as exc:
        raise ImportError("statsmodels is required; install with 'pip install statsmodels'.") from exc
    X = pd.get_dummies(df[predictors], drop_first=True).astype(float)
    y = pd.Categorical(df[response], ordered=True)
    model = OrderedModel(y, X, distr="logit")
    return model.fit(method="bfgs", disp=False)


def binomial_mixed_model(df: pd.DataFrame, response: str, fixed: list[str], group: str):
    """Binomial mixed model with a random intercept per group (statsmodels Bayesian GLMM)."""
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as exc:
        raise ImportError("statsmodels is required; install with 'pip install statsmodels'.") from exc
    formula = f"{response} ~ " + " + ".join(fixed)
    model = BinomialBayesMixedGLM.from_formula(formula, {"g": f"0 + C({group})"}, df)
    return model.fit_vb()


def summarize_by(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean, standard deviation and count of metrics per group, with 95 percent normal confidence intervals."""
    g = df.groupby(by)[metrics]
    out = g.agg(["mean", "std", "count"])
    for m in metrics:
        se = out[(m, "std")] / np.sqrt(out[(m, "count")].clip(lower=1))
        out[(m, "ci_low")] = out[(m, "mean")] - 1.96 * se
        out[(m, "ci_high")] = out[(m, "mean")] + 1.96 * se
    return out.sort_index(axis=1)
