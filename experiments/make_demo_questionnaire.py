"""Simulate a questionnaire dataset with known subscales for testing Phase C.

The files produced by this script contain simulated responses only. No real
participants were involved, and the constructs, items, covariates and outcome
are artificial. The purpose is to exercise experiments/run_phase_c.py offline
and to document the input format expected by
pfca.evaluation.datasets.load_questionnaire. The real questionnaire dataset of
the study is not distributed with the repository (see data/README.md).

Generating process
------------------
1. Latent constructs z_1 to z_C are drawn from a multivariate normal
   distribution with unit variances and a common correlation between
   constructs.
2. Every construct is measured by a block of Likert items with values 1 to 5.
   The continuous response of an item in block c is lambda_c z_c plus
   sqrt(1 - lambda_c^2) e with independent standard normal e, so that the
   within subscale correlation equals lambda_c squared. The continuous
   response is discretized at fixed standard normal cut points. The loadings
   differ between constructs, so that the constructs are measured with
   different reliability and the concept level uncertainty of the
   attributions differs between them.
3. Two covariates are drawn, age in years and sex coded 0 and 1.
4. The outcome is a nonlinear function of the constructs (linear, tanh,
   square, cubic and sine links taken in turn, plus one product term between
   the first two constructs), small covariate effects and Gaussian noise. With
   --classification a binary outcome is drawn from a logistic model of the
   same signal.

The seed is fixed by --seed, printed, and stored in the specification file
together with every generator setting, so that the files can be regenerated.

Usage
-----
python experiments/make_demo_questionnaire.py --n 300 --seed 0
python experiments/make_demo_questionnaire.py --n 1000 --seed 1 --classification
python experiments/make_demo_questionnaire.py --n 500 --n-constructs 5 --items-per-construct 6 --loadings 0.9 0.85 0.8 0.75 0.7
python experiments/make_demo_questionnaire.py --n 300 --out-dir data --name demo_questionnaire
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

from pfca.evaluation.synthetic import LINKS  # noqa: E402

CONSTRUCTS = [
    ("anxiety", "anx"),
    ("rumination", "rum"),
    ("self_efficacy", "eff"),
    ("social_support", "sup"),
    ("sleep_quality", "slp"),
    ("workload", "wrk"),
]
LINK_CYCLE = ["linear", "tanh", "square", "cubic", "sin"]
LIKERT_CUTS = (-1.5, -0.5, 0.5, 1.5)


def construct_names(n_constructs: int) -> list[tuple[str, str]]:
    """Names and item prefixes of the simulated constructs."""
    out = list(CONSTRUCTS[:n_constructs])
    for c in range(len(out), n_constructs):
        out.append((f"construct_{c + 1}", f"c{c + 1}"))
    return out


def default_loadings(n_constructs: int) -> np.ndarray:
    """Loadings decreasing from 0.9 to 0.75, so that latent within subscale correlations range from 0.81 to 0.56.

    The correlations observed between the discretized Likert items are
    somewhat smaller than the latent values because of the coarse scale.
    """
    if n_constructs == 1:
        return np.array([0.85])
    return np.linspace(0.9, 0.75, n_constructs)


def simulate(
    n: int,
    seed: int,
    n_constructs: int = 4,
    items_per_construct: int = 5,
    loadings=None,
    construct_correlation: float = 0.3,
    signal_to_noise: float = 4.0,
    interaction: float = 0.5,
    classification: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Simulate the questionnaire and return the data frame and the specification dictionary."""
    if n_constructs < 1 or items_per_construct < 2:
        raise ValueError("At least one construct with at least two items is required.")
    rng = np.random.default_rng(seed)
    names = construct_names(n_constructs)
    lam = default_loadings(n_constructs) if loadings is None else np.asarray(loadings, dtype=float)
    if lam.shape != (n_constructs,):
        raise ValueError("One loading per construct is required.")
    if np.any(lam <= 0) or np.any(lam >= 1):
        raise ValueError("Loadings must lie strictly between 0 and 1.")
    # latent constructs with a common correlation
    R = np.full((n_constructs, n_constructs), float(construct_correlation))
    np.fill_diagonal(R, 1.0)
    Z = rng.multivariate_normal(np.zeros(n_constructs), R, size=n)
    # Likert items
    frame = {}
    subscales: dict[str, list[str]] = {}
    for c, (cname, prefix) in enumerate(names):
        cols = []
        for i in range(items_per_construct):
            latent = lam[c] * Z[:, c] + np.sqrt(1.0 - lam[c] ** 2) * rng.normal(size=n)
            item = 1 + np.searchsorted(np.asarray(LIKERT_CUTS), latent)
            col = f"{prefix}_{i + 1}"
            frame[col] = item.astype(int)
            cols.append(col)
        subscales[cname] = cols
    # covariates
    age = np.clip(np.round(rng.normal(40.0, 12.0, size=n)), 18, 70).astype(int)
    sex = rng.integers(0, 2, size=n)
    frame["age"] = age
    frame["sex"] = sex
    # outcome
    coefficients = np.linspace(1.5, 0.5, n_constructs) * np.array([1.0 if c % 2 == 0 else -1.0 for c in range(n_constructs)])
    links = [LINK_CYCLE[c % len(LINK_CYCLE)] for c in range(n_constructs)]
    signal = np.zeros(n)
    for c in range(n_constructs):
        signal += coefficients[c] * LINKS[links[c]](Z[:, c])
    if n_constructs >= 2 and interaction != 0.0:
        signal += interaction * Z[:, 0] * Z[:, 1]
    covariate_effects = {"age": 0.02, "sex": 0.3}
    signal += covariate_effects["age"] * (age - 40.0) + covariate_effects["sex"] * sex
    sd_signal = float(np.std(signal)) if np.std(signal) > 0 else 1.0
    noise_sd = sd_signal / np.sqrt(signal_to_noise)
    if classification:
        logit = 1.5 * (signal - np.median(signal)) / sd_signal
        p = 1.0 / (1.0 + np.exp(-logit))
        outcome = (rng.uniform(size=n) < p).astype(int)
        task = "classification"
    else:
        outcome = np.round(50.0 + 10.0 * (signal + rng.normal(size=n) * noise_sd) / sd_signal, 2)
        task = "regression"
    frame["outcome"] = outcome
    df = pd.DataFrame(frame)
    df.insert(0, "participant", [f"sim{k + 1:05d}" for k in range(n)])
    spec = {
        "name": "demo_questionnaire",
        "simulated": True,
        "description": "Simulated questionnaire with known subscales, generated by experiments/make_demo_questionnaire.py for testing. The rows are not real participants.",
        "outcome": "outcome",
        "task": task,
        "subscales": subscales,
        "covariates": ["age", "sex"],
        "likert_range": [1, 5],
        "generator": {
            "script": "experiments/make_demo_questionnaire.py",
            "seed": int(seed),
            "n": int(n),
            "n_constructs": int(n_constructs),
            "items_per_construct": int(items_per_construct),
            "loadings": [float(v) for v in lam],
            "within_subscale_correlation": [float(v**2) for v in lam],
            "construct_correlation": float(construct_correlation),
            "links": links,
            "coefficients": [float(v) for v in coefficients],
            "interaction": float(interaction) if n_constructs >= 2 else 0.0,
            "covariate_effects": covariate_effects,
            "signal_to_noise": float(signal_to_noise),
            "noise_sd": float(noise_sd),
            "likert_cut_points": list(LIKERT_CUTS),
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    if classification:
        spec["positive_class"] = 1
    return df, spec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600, help="number of simulated participants")
    ap.add_argument("--seed", type=int, default=0, help="random seed (fixed and stored in the specification file)")
    ap.add_argument("--n-constructs", type=int, default=4, help="number of latent constructs (subscales)")
    ap.add_argument("--items-per-construct", type=int, default=5, help="Likert items per subscale")
    ap.add_argument("--loadings", type=float, nargs="+", default=None, help="one loading per construct; default decreases from 0.9 to 0.75")
    ap.add_argument("--construct-correlation", type=float, default=0.3, help="correlation between latent constructs")
    ap.add_argument("--signal-to-noise", type=float, default=4.0, help="variance ratio of signal to noise for the regression outcome")
    ap.add_argument("--interaction", type=float, default=0.5, help="coefficient of the product term between the first two constructs")
    ap.add_argument("--classification", action="store_true", help="draw a binary outcome instead of a continuous one")
    ap.add_argument("--out-dir", default=str(ROOT / "data"), help="output directory (default: the data directory of the repository)")
    ap.add_argument("--name", default="demo_questionnaire", help="base name of the CSV and JSON files")
    args = ap.parse_args(argv)

    df, spec = simulate(
        n=args.n,
        seed=args.seed,
        n_constructs=args.n_constructs,
        items_per_construct=args.items_per_construct,
        loadings=args.loadings,
        construct_correlation=args.construct_correlation,
        signal_to_noise=args.signal_to_noise,
        interaction=args.interaction,
        classification=args.classification,
    )
    spec["name"] = args.name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.name}.csv"
    spec_path = out_dir / f"{args.name}_spec.json"
    df.to_csv(csv_path, index=False)
    with open(spec_path, "w", encoding="utf8") as fh:
        json.dump(spec, fh, indent=2)
    print(f"[demo questionnaire] seed {args.seed}, {args.n} simulated participants, {args.n_constructs} constructs x {args.items_per_construct} items, task {spec['task']}")
    print(f"[demo questionnaire] within subscale correlations: {', '.join(f'{v:.2f}' for v in spec['generator']['within_subscale_correlation'])}")
    print(f"[demo questionnaire] wrote {csv_path} and {spec_path}")


if __name__ == "__main__":
    main()
