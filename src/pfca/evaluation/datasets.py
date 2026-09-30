"""Dataset access for Phases B and C.

Public benchmark datasets are downloaded by name and version from OpenML at
run time. The list below is a proposal of tabular datasets with correlated
features and between 15 and 100 features; the loader verifies the dimension
range and skips datasets outside it. The psychological questionnaire of Phase
C is loaded from a local file together with a specification of its subscales.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARKS: list[dict] = [
    {"name": "spambase", "version": 1, "task": "classification"},
    {"name": "ionosphere", "version": 1, "task": "classification"},
    {"name": "sonar", "version": 1, "task": "classification"},
    {"name": "qsar-biodeg", "version": 1, "task": "classification"},
    {"name": "steel-plates-fault", "version": 1, "task": "classification"},
    {"name": "ozone-level-8hr", "version": 1, "task": "classification"},
    {"name": "climate-model-simulation-crashes", "version": 1, "task": "classification"},
    {"name": "pol", "version": 1, "task": "regression"},
    {"name": "ailerons", "version": 1, "task": "regression"},
    {"name": "elevators", "version": 1, "task": "regression"},
    {"name": "house_16H", "version": 1, "task": "regression"},
    {"name": "bank32nh", "version": 1, "task": "regression"},
    {"name": "puma32H", "version": 1, "task": "regression"},
    {"name": "cpu_act", "version": 1, "task": "regression"},
]


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    task: str
    feature_names: list[str]
    groups: dict[str, list[str]] | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def _prepare_frame(X: pd.DataFrame, y: pd.Series, task: str, max_rows: int | None, random_state: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = X.copy()
    for c in X.columns:
        if X[c].dtype.name in ("category", "object", "bool"):
            X[c] = pd.factorize(X[c])[0].astype(float)
    X = X.astype(float)
    X = X.loc[:, X.std(axis=0) > 0]
    if task == "classification":
        codes, uniques = pd.factorize(y)
        if len(uniques) > 2:
            top = pd.Series(codes).value_counts().index[0]
            codes = (codes == top).astype(int)
        y = codes.astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce").to_numpy(float)
    X = X.fillna(X.median())
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    if max_rows is not None and X.shape[0] > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=max_rows, replace=False)
        X, y = X.iloc[idx], y[idx]
    return X.to_numpy(float), np.asarray(y), [str(c) for c in X.columns]


def load_benchmark(name: str, version: int = 1, task: str | None = None, max_rows: int | None = 5000, random_state: int = 0, data_home: str | None = None) -> Dataset:
    """Download and prepare an OpenML dataset (categoricals are integer coded, constant features dropped)."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name=name, version=version, as_frame=True, parser="auto", data_home=data_home)
    if task is None:
        task = next((b["task"] for b in BENCHMARKS if b["name"] == name), None) or ("classification" if bunch.target.dtype.name in ("category", "object") else "regression")
    X, y, names = _prepare_frame(bunch.data, bunch.target, task, max_rows, random_state)
    meta = {"openml_id": bunch.details.get("id") if hasattr(bunch, "details") else None, "version": version, "n_rows": int(X.shape[0])}
    return Dataset(name, X, y, task, names, None, meta)


def load_benchmarks(min_features: int = 15, max_features: int = 100, names: list[str] | None = None, **kwargs) -> list[Dataset]:
    """Load the benchmark list, skipping datasets outside the dimension range or unavailable."""
    out = []
    for spec in BENCHMARKS:
        if names is not None and spec["name"] not in names:
            continue
        try:
            ds = load_benchmark(spec["name"], spec["version"], spec["task"], **kwargs)
        except Exception as exc:  # network or parsing failures are reported, not fatal
            print(f"[datasets] skipping {spec['name']}: {type(exc).__name__}: {exc}")
            continue
        if not (min_features <= ds.n_features <= max_features):
            print(f"[datasets] skipping {spec['name']}: {ds.n_features} features outside [{min_features}, {max_features}]")
            continue
        out.append(ds)
    return out


def load_questionnaire(csv_path: str, spec_path: str) -> Dataset:
    """Load a questionnaire dataset with known subscales (Phase C).

    The specification is a JSON file with keys ``outcome`` (column name),
    ``task`` ('regression' or 'classification'), ``subscales`` (mapping from
    subscale name to list of item columns) and optionally ``covariates`` (list
    of additional columns) and ``positive_class``.
    """
    spec = json.loads(Path(spec_path).read_text(encoding="utf8"))
    df = pd.read_csv(csv_path)
    items = [c for cols in spec["subscales"].values() for c in cols]
    covariates = list(spec.get("covariates", []))
    cols = items + covariates
    missing = [c for c in cols + [spec["outcome"]] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from the questionnaire file: {missing}")
    sub = df[cols + [spec["outcome"]]].dropna()
    X = sub[cols].astype(float)
    y = sub[spec["outcome"]]
    task = spec.get("task", "regression")
    if task == "classification":
        pos = spec.get("positive_class")
        y = (y == pos).astype(int).to_numpy() if pos is not None else pd.factorize(y)[0]
    else:
        y = y.astype(float).to_numpy()
    groups = {k: list(v) for k, v in spec["subscales"].items()}
    if covariates:
        groups["covariates"] = covariates
    return Dataset(spec.get("name", Path(csv_path).stem), X.to_numpy(float), np.asarray(y), task, cols, groups, {"n_rows": int(X.shape[0])})


def stratified_splits(y: np.ndarray, task: str, n_repeats: int = 10, proportions=(0.6, 0.2, 0.2), random_state: int = 0):
    """Repeated train, tuning and test splits with the proportions of Section 7.3."""
    from sklearn.model_selection import train_test_split

    n = len(y)
    for r in range(n_repeats):
        seed = random_state + r
        strat = y if task == "classification" else None
        idx_train, idx_rest = train_test_split(np.arange(n), train_size=proportions[0], random_state=seed, stratify=strat)
        rest_frac = proportions[1] / (proportions[1] + proportions[2])
        strat_rest = y[idx_rest] if task == "classification" else None
        idx_tune, idx_test = train_test_split(idx_rest, train_size=rest_frac, random_state=seed, stratify=strat_rest)
        yield r, idx_train, idx_tune, idx_test
