"""Dataset access for Phases B and C.

Public benchmark datasets come from two sources. OpenML datasets are
downloaded by name and version at run time, which requires network access.
Built in datasets of scikit-learn ship with the package and load offline, so
that the Phase B pipeline can be exercised without a network connection. The
list below is a proposal of tabular datasets with correlated features and
between 15 and 100 features; the loader verifies the dimension range and skips
datasets outside it. The psychological questionnaire of Phase C is loaded from
a local file together with a specification of its subscales.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Every entry has a "source" ("openml" or "sklearn"); OpenML entries also carry a version.
BENCHMARKS: list[dict] = [
    {"name": "breast_cancer", "source": "sklearn", "task": "classification"},
    {"name": "spambase", "source": "openml", "version": 1, "task": "classification"},
    {"name": "ionosphere", "source": "openml", "version": 1, "task": "classification"},
    {"name": "sonar", "source": "openml", "version": 1, "task": "classification"},
    {"name": "qsar-biodeg", "source": "openml", "version": 1, "task": "classification"},
    {"name": "steel-plates-fault", "source": "openml", "version": 1, "task": "classification"},
    {"name": "ozone-level-8hr", "source": "openml", "version": 1, "task": "classification"},
    {"name": "climate-model-simulation-crashes", "source": "openml", "version": 1, "task": "classification"},
    {"name": "pol", "source": "openml", "version": 1, "task": "regression"},
    {"name": "ailerons", "source": "openml", "version": 1, "task": "regression"},
    {"name": "elevators", "source": "openml", "version": 1, "task": "regression"},
    {"name": "house_16H", "source": "openml", "version": 1, "task": "regression"},
    {"name": "bank32nh", "source": "openml", "version": 1, "task": "regression"},
    {"name": "puma32H", "source": "openml", "version": 1, "task": "regression"},
    {"name": "cpu_act", "source": "openml", "version": 1, "task": "regression"},
]

# Built in scikit-learn datasets available offline, mapped to their loader functions.
SKLEARN_LOADERS: dict[str, str] = {
    "breast_cancer": "load_breast_cancer",
}
# Explained (positive) class of built in binary datasets; the clinically conventional class is used for breast_cancer
SKLEARN_POSITIVE_CLASS: dict[str, str] = {"breast_cancer": "malignant"}


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


def benchmark_spec(name: str) -> dict | None:
    """Return the BENCHMARKS entry of a dataset, or None when the name is not listed."""
    return next((dict(b) for b in BENCHMARKS if b["name"] == name), None)


def _prepare_frame(X: pd.DataFrame, y: pd.Series, task: str, max_rows: int | None, random_state: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = X.copy()
    for c in X.columns:
        if X[c].dtype.name in ("category", "object", "bool"):
            X[c] = pd.factorize(X[c])[0].astype(float)
    X = X.astype(float)
    X = X.loc[:, X.std(axis=0) > 0]
    if task == "classification":
        y_num = pd.to_numeric(pd.Series(np.asarray(y, dtype=object)), errors="coerce")
        values = set(y_num.dropna().unique().tolist())
        if y_num.notna().all() and len(values) == 2 and values <= {0, 1}:
            # labels already coded 0 and 1: keep the coding so that label 1 is the explained positive class
            y = y_num.to_numpy(float)
        else:
            codes, uniques = pd.factorize(y)
            if len(uniques) > 2:
                top = pd.Series(codes).value_counts().index[0]
                codes = (codes == top).astype(int)
            y = codes.astype(float)
    else:
        y = pd.to_numeric(y, errors="coerce").to_numpy(float)
    X = X.fillna(X.median())
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    if max_rows is not None and X.shape[0] > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=max_rows, replace=False)
        X, y = X.iloc[idx], y[idx]
    if task == "classification":
        y = y.astype(int)
    return X.to_numpy(float), np.asarray(y), [str(c) for c in X.columns]


def _load_sklearn_frame(name: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load a built in scikit-learn dataset as a data frame and a target series (offline)."""
    from sklearn import datasets as sk_datasets

    if name not in SKLEARN_LOADERS:
        raise ValueError(f"Unknown scikit-learn dataset '{name}'; available: {sorted(SKLEARN_LOADERS)}")
    bunch = getattr(sk_datasets, SKLEARN_LOADERS[name])(as_frame=True)
    target_names = [str(t) for t in getattr(bunch, "target_names", [])]
    meta = {"sklearn_loader": SKLEARN_LOADERS[name], "target_names": target_names}
    target = bunch.target
    if len(target_names) == 2:
        positive = SKLEARN_POSITIVE_CLASS.get(name, target_names[1])
        pos_idx = target_names.index(positive)
        target = (pd.Series(np.asarray(bunch.target)) == pos_idx).astype(int)
        target.name = getattr(bunch.target, "name", "target")
        meta["positive_class"] = positive
    return bunch.data, target, meta


def load_benchmark(
    name: str,
    version: int = 1,
    task: str | None = None,
    max_rows: int | None = 5000,
    random_state: int = 0,
    data_home: str | None = None,
    source: str | None = None,
) -> Dataset:
    """Load and prepare a benchmark dataset (categoricals are integer coded, constant features dropped).

    ``source`` is 'openml' (downloaded by name and version) or 'sklearn' (built
    in dataset, offline). When omitted it is taken from the BENCHMARKS entry
    of the name, and defaults to 'openml' for names that are not listed.
    """
    spec = benchmark_spec(name)
    if source is None:
        source = spec["source"] if spec is not None else "openml"
    if task is None and spec is not None:
        task = spec["task"]
    if source == "sklearn":
        data, target, meta = _load_sklearn_frame(name)
        if task is None:
            task = "classification" if pd.Series(target).nunique() <= 10 else "regression"
    elif source == "openml":
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(name=name, version=version, as_frame=True, parser="auto", data_home=data_home)
        data, target = bunch.data, bunch.target
        if task is None:
            task = "classification" if target.dtype.name in ("category", "object") else "regression"
        meta = {"openml_id": bunch.details.get("id") if hasattr(bunch, "details") else None, "version": version}
    else:
        raise ValueError("source must be 'openml' or 'sklearn'")
    X, y, names = _prepare_frame(data, target, task, max_rows, random_state)
    meta.update({"source": source, "n_rows": int(X.shape[0])})
    return Dataset(name, X, y, task, names, None, meta)


def load_benchmarks(min_features: int = 15, max_features: int = 100, names: list[str] | None = None, **kwargs) -> list[Dataset]:
    """Load the benchmark list, skipping datasets outside the dimension range or unavailable."""
    out = []
    for spec in BENCHMARKS:
        if names is not None and spec["name"] not in names:
            continue
        try:
            ds = load_benchmark(spec["name"], spec.get("version", 1), spec["task"], source=spec.get("source", "openml"), **kwargs)
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
