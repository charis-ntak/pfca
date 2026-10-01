"""Unit tests of the shared helpers of the experiment scripts (experiments/common.py)."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

import pfca

ROOT = Path(__file__).resolve().parents[1]


def _load_common():
    spec = importlib.util.spec_from_file_location("pfca_experiments_common", ROOT / "experiments" / "common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_common()


def test_load_config_applies_overrides_and_ignores_none(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\nb: [1, 2]\n", encoding="utf8")
    assert common.load_config(path) == {"a": 1, "b": [1, 2]}
    assert common.load_config(path, {"a": None, "b": [3], "c": "x"}) == {"a": 1, "b": [3], "c": "x"}
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf8")
    assert common.load_config(empty) == {}


def test_serialize_value_and_flatten_row():
    table = pd.DataFrame({"x": [1, 2]})
    assert json.loads(common._serialize_value(table)) == {"x": [1, 2]}
    assert json.loads(common._serialize_value(np.arange(3))) == [0, 1, 2]
    assert json.loads(common._serialize_value([1, "a"])) == [1, "a"]
    assert json.loads(common._serialize_value({"k": (1, 2)})) == {"k": [1, 2]}
    assert common._serialize_value(np.float64(1.5)) == 1.5 and type(common._serialize_value(np.float64(1.5))) is float
    assert common._serialize_value(np.int64(2)) == 2 and type(common._serialize_value(np.int64(2))) is int
    assert common._serialize_value(np.bool_(True)) is True
    assert common._serialize_value("s") == "s" and common._serialize_value(None) is None
    row = common.flatten_row({"method": "m", "value": np.float32(0.5), "table": table, "per_instance": table})
    assert row == {"method": "m", "value": 0.5, "table": json.dumps({"x": [1, 2]})}


def test_result_writer_appends_widens_and_resumes(tmp_path):
    path = tmp_path / "metrics.csv"
    writer = common.ResultWriter(path, ["cell", "method"])
    assert not writer.is_done({"cell": 1, "method": "a"})
    writer.append([])
    assert not path.exists()
    writer.append([{"cell": 1, "method": "a", "score": 0.5, "table": pd.DataFrame({"x": [1]})}, {"cell": 1, "method": "b", "score": np.float64(0.25)}])
    assert writer.is_done({"cell": 1, "method": "a"}) and writer.is_done({"cell": "1", "method": "b"})  # keys compare as strings
    df = pd.read_csv(path)
    assert list(df.columns) == ["cell", "method", "score", "table"] and len(df) == 2
    assert json.loads(df.loc[0, "table"]) == {"x": [1]} and pd.isna(df.loc[1, "table"])
    # a batch with a new column rewrites the file with the union of the columns
    writer.append([{"cell": 2, "method": "a", "extra": np.array([1, 2])}])
    df = pd.read_csv(path)
    assert list(df.columns) == ["cell", "method", "score", "table", "extra"] and len(df) == 3
    assert json.loads(df.loc[2, "extra"]) == [1, 2] and pd.isna(df.loc[2, "score"])
    # a batch without new columns is aligned to the existing ones and appended
    writer.append([{"method": "c", "cell": 2, "score": 1.0}])
    df = pd.read_csv(path)
    assert len(df) == 4 and df.loc[3, "cell"] == 2 and df.loc[3, "method"] == "c" and df.loc[3, "score"] == 1.0
    # a fresh writer resumes from the file
    resumed = common.ResultWriter(path, ["cell", "method"])
    assert resumed.is_done({"cell": 2, "method": "c"}) and not resumed.is_done({"cell": 3, "method": "c"})
    # rows lacking a key column are written but never counted as done
    resumed.append([{"cell": 3, "score": 2.0}])
    assert not resumed.is_done({"cell": 3, "method": "nan"})
    assert len(pd.read_csv(path)) == 5
    # a file without the key columns resumes nothing and an unreadable file is ignored
    other = tmp_path / "other.csv"
    other.write_text("x,y\n1,2\n", encoding="utf8")
    assert not common.ResultWriter(other, ["cell", "method"]).is_done({"cell": 1, "method": "a"})
    broken = tmp_path / "broken.csv"
    broken.write_text("", encoding="utf8")
    assert common.ResultWriter(broken, ["cell"])._done == set()


def test_per_instance_writer(tmp_path):
    path = tmp_path / "per_instance.csv"
    writer = common.PerInstanceWriter(path)
    writer.append({"cell": 1}, "m", None)
    writer.append({"cell": 1}, "m", pd.DataFrame())
    assert not path.exists()
    table = pd.DataFrame({"a": [0.1, 0.2], "b": [1.0, 2.0]})
    rows = [{"cell": 1, "method": "m", "score": 0.5, "per_instance": table}, {"cell": 1, "method": "n", "score": 0.1}, {"method": "o", "per_instance": table}]
    writer.append_from_rows(rows, ["cell"])
    assert all("per_instance" not in r for r in rows)  # the table is removed from the metric rows
    df = pd.read_csv(path)
    assert len(df) == 4 and list(df.columns) == ["instance", "metric", "value", "cell", "method"]
    assert set(df["metric"]) == {"a", "b"} and set(df["instance"]) == {0, 1} and (df["method"] == "m").all()
    # a table that already carries an instance column keeps it
    writer.append({"cell": 2}, "p", pd.DataFrame({"instance": [7, 8], "a": [1.0, 2.0]}))
    df = pd.read_csv(path)
    assert len(df) == 6 and set(df.loc[df["method"] == "p", "instance"]) == {7, 8}


def test_model_classes_from_names():
    reg = common.model_classes_from_names(["gbm", "rf", "mlp", "ridge"], "regression", 3)
    assert [n for n, _ in reg] == ["gbm", "rf", "mlp", "ridge"]
    assert isinstance(reg[0][1], GradientBoostingRegressor) and reg[0][1].random_state == 3
    assert isinstance(reg[1][1], RandomForestRegressor) and reg[1][1].random_state == 3
    assert isinstance(reg[2][1], Pipeline) and isinstance(reg[2][1].steps[-1][1], MLPRegressor)
    assert isinstance(reg[3][1], Ridge)
    clf = common.model_classes_from_names(["gbm", "rf", "mlp", "logistic"], "classification", 0)
    assert isinstance(clf[0][1], GradientBoostingClassifier) and isinstance(clf[1][1], RandomForestClassifier)
    assert isinstance(clf[2][1].steps[-1][1], MLPClassifier) and isinstance(clf[3][1], LogisticRegression)
    assert common.model_classes_from_names([], "regression", 0) == []
    with pytest.raises(ValueError, match="Unknown model class"):
        common.model_classes_from_names(["svm"], "regression", 0)


def test_tune_gradient_boosting(small_problem):
    p, _ = small_problem
    X, y, X_tune, y_tune = p.X[:150], p.y[:150], p.X[150:], p.y[150:]
    grid = {"max_depth": [1, 2], "n_estimators": [10, 20]}
    model, params, score = common.tune_gradient_boosting(X, y, X_tune, y_tune, "regression", 0, grid)
    assert isinstance(model, GradientBoostingRegressor)
    assert params["max_depth"] in (1, 2) and params["n_estimators"] in (10, 20)
    assert model.get_params()["max_depth"] == params["max_depth"] and model.get_params()["n_estimators"] == params["n_estimators"]
    assert score == pytest.approx(mean_squared_error(y_tune, model.predict(X_tune)))
    # the selected pair is the best of the grid on the tuning split
    for depth in grid["max_depth"]:
        for n_est in grid["n_estimators"]:
            m = GradientBoostingRegressor(max_depth=depth, n_estimators=n_est, learning_rate=0.05, random_state=0).fit(X, y)
            assert mean_squared_error(y_tune, m.predict(X_tune)) >= score - 1e-12
    y_cls, y_tune_cls = (y > np.median(y)).astype(int), (y_tune > np.median(y)).astype(int)
    clf, cparams, cscore = common.tune_gradient_boosting(X, y_cls, X_tune, y_tune_cls, "classification", 0, grid)
    assert isinstance(clf, GradientBoostingClassifier) and cscore > 0 and cparams["max_depth"] in (1, 2)
    # without a grid the default one of Section 7.3 is searched
    _, default_params, _ = common.tune_gradient_boosting(X[:60], y[:60], X_tune[:30], y_tune[:30], "regression", 0)
    assert default_params["max_depth"] in (2, 3, 4) and default_params["n_estimators"] in (100, 300)


def test_prepare_run_freezes_configuration_and_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_x.py", "--flag"])
    cfg = {"seed": 3, "split_seed": 4, "n": 2, "nested": {"a": [1, 2]}}
    out = common.prepare_run(tmp_path / "results", "run_x", cfg)
    assert out == tmp_path / "results" / "run_x" and out.is_dir()
    assert yaml.safe_load((out / "config.yaml").read_text(encoding="utf8")) == cfg
    env = json.loads((out / "environment.json").read_text(encoding="utf8"))
    assert env["argv"] == ["run_x.py", "--flag"] and env["seeds_from_config"] == {"seed": 3, "split_seed": 4}
    assert env["cwd"] == os.getcwd() and env["started"]
    versions = env["versions"]
    assert {"python", "platform", "numpy", "pandas", "scipy", "scikit-learn", "shap", "pymoo", "pfca"} <= set(versions)
    assert versions["pfca"] == pfca.__version__
    assert common.package_versions()["python"] == sys.version.split()[0]
    # a second run with the same name reuses the directory
    assert common.prepare_run(tmp_path / "results", "run_x", {"seed": 1}) == out
    assert yaml.safe_load((out / "config.yaml").read_text(encoding="utf8")) == {"seed": 1}
