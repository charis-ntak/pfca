"""Shared utilities for the experiment scripts: configuration, logging of the
environment and seeds, incremental result files with resume support, and the
reference model tuning of Section 7.3."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    with open(path, "r", encoding="utf8") as fh:
        cfg = yaml.safe_load(fh) or {}
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def package_versions() -> dict:
    import sklearn
    import scipy
    import shap
    import pymoo

    import pfca

    versions = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "shap": shap.__version__, "pymoo": pymoo.__version__, "pfca": pfca.__version__}
    try:
        import statsmodels

        versions["statsmodels"] = statsmodels.__version__
    except ImportError:
        pass
    return versions


def prepare_run(results_dir: str | Path, name: str, config: dict) -> Path:
    """Create the results directory and write the frozen configuration and environment."""
    out = Path(results_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    seeds = {k: v for k, v in config.items() if "seed" in str(k).lower()}
    with open(out / "environment.json", "w", encoding="utf8") as fh:
        json.dump({"versions": package_versions(), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cwd": os.getcwd(), "argv": sys.argv, "seeds_from_config": seeds}, fh, indent=2)
    return out


def _serialize_value(v):
    """Make a metric value CSV friendly: nested tables and arrays are stored as JSON strings."""
    if isinstance(v, pd.DataFrame):
        return json.dumps(v.to_dict(orient="list"))
    if isinstance(v, np.ndarray):
        return json.dumps(v.tolist())
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    return v


class ResultWriter:
    """Append rows to a CSV file with resume support keyed on chosen columns.

    Rows may carry different column sets (for example a metric that exists only
    for some methods); the file is kept rectangular by aligning every batch to
    the union of the columns seen so far and rewriting the file when new
    columns appear. Nested tables (pandas data frames, arrays, lists, dicts)
    are stored as JSON strings so that figures can be regenerated from the
    saved files.
    """

    def __init__(self, path: str | Path, key_columns: list[str]):
        self.path = Path(path)
        self.key_columns = key_columns
        self._done: set[tuple] = set()
        self._columns: list[str] = []
        if self.path.exists():
            try:
                prev = pd.read_csv(self.path)
                self._columns = list(prev.columns)
                if all(c in prev.columns for c in key_columns):
                    self._done = set(map(tuple, prev[key_columns].astype(str).itertuples(index=False, name=None)))
            except Exception:
                pass

    def is_done(self, key: dict) -> bool:
        return tuple(str(key[c]) for c in self.key_columns) in self._done

    def append(self, rows: list[dict]) -> None:
        if not rows:
            return
        clean = [{k: _serialize_value(v) for k, v in r.items()} for r in rows]
        df = pd.DataFrame(clean)
        new_columns = [c for c in df.columns if c not in self._columns]
        if self.path.exists() and self._columns and new_columns:
            prev = pd.read_csv(self.path)
            merged = pd.concat([prev, df], ignore_index=True, sort=False)
            self._columns = list(merged.columns)
            merged.to_csv(self.path, index=False)
        else:
            if not self._columns:
                self._columns = list(df.columns)
            df = df.reindex(columns=self._columns)
            df.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)
        for r in rows:
            if all(c in r for c in self.key_columns):
                self._done.add(tuple(str(r[c]) for c in self.key_columns))


class PerInstanceWriter:
    """Long format store of per instance metric values (Section 7.3 asks for raw values per instance).

    Every call appends one row per instance and metric with the identifying
    columns of the cell, so that ``per_instance.csv`` can be aggregated in any
    way afterwards.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, key: dict, method: str, table: pd.DataFrame | None) -> None:
        if table is None or len(table) == 0:
            return
        long = table.reset_index().rename(columns={"index": "instance"}) if "instance" not in table.columns else table.copy()
        long = long.melt(id_vars=["instance"], var_name="metric", value_name="value")
        for k, v in key.items():
            long[k] = v
        long["method"] = method
        long.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)


def model_classes_from_names(names: list[str], task: str, seed: int):
    """Instantiate model classes by short name: gbm, rf, mlp, ridge, logistic."""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    reg = task == "regression"
    out = []
    for n in names:
        if n == "gbm":
            est = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed) if reg else GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=seed)
        elif n == "rf":
            est = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed) if reg else RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed)
        elif n == "mlp":
            core = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True, random_state=seed) if reg else MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True, random_state=seed)
            est = Pipeline([("scale", StandardScaler()), ("mlp", core)])
        elif n in ("ridge", "logistic"):
            est = Ridge(alpha=1.0) if reg else LogisticRegression(max_iter=1000)
        else:
            raise ValueError(f"Unknown model class '{n}'")
        out.append((n, est))
    return out


def tune_gradient_boosting(X_tr, y_tr, X_tune, y_tune, task: str, seed: int, grid: dict | None = None):
    """Select depth and number of trees of the reference gradient boosting model on the tuning set."""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import log_loss, mean_squared_error

    grid = grid or {"max_depth": [2, 3, 4], "n_estimators": [100, 300]}
    best, best_score, best_params = None, np.inf, None
    for depth in grid["max_depth"]:
        for n_est in grid["n_estimators"]:
            if task == "regression":
                m = GradientBoostingRegressor(max_depth=depth, n_estimators=n_est, learning_rate=0.05, random_state=seed).fit(X_tr, y_tr)
                score = mean_squared_error(y_tune, m.predict(X_tune))
            else:
                m = GradientBoostingClassifier(max_depth=depth, n_estimators=n_est, learning_rate=0.05, random_state=seed).fit(X_tr, y_tr)
                score = log_loss(y_tune, np.clip(m.predict_proba(X_tune)[:, 1], 1e-6, 1 - 1e-6))
            if score < best_score:
                best, best_score, best_params = m, score, {"max_depth": depth, "n_estimators": n_est}
    return best, best_params, float(best_score)


def flatten_row(r: dict) -> dict:
    """Prepare a metric dictionary for the flat CSV: nested tables become JSON strings.

    The per instance table (key ``per_instance``) is removed here because it is
    written separately by :class:`PerInstanceWriter`.
    """
    return {k: _serialize_value(v) for k, v in r.items() if k != "per_instance"}
