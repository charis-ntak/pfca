"""Tests of the dataset access of Phases B and C (offline: the OpenML download is replaced by a stub)."""

import json

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
from sklearn.utils import Bunch

from pfca.evaluation import datasets


def test_benchmark_list_and_spec():
    names = [b["name"] for b in datasets.BENCHMARKS]
    assert len(names) == len(set(names))
    assert all(b["source"] in ("openml", "sklearn") and b["task"] in ("classification", "regression") for b in datasets.BENCHMARKS)
    assert all("version" in b for b in datasets.BENCHMARKS if b["source"] == "openml")
    spec = datasets.benchmark_spec("breast_cancer")
    assert spec == {"name": "breast_cancer", "source": "sklearn", "task": "classification"}
    spec["name"] = "changed"  # a copy is returned
    assert datasets.benchmark_spec("breast_cancer")["name"] == "breast_cancer"
    assert datasets.benchmark_spec("no_such_dataset") is None


def test_prepare_frame_codes_categoricals_drops_constants_and_imputes():
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "cat": pd.Categorical(["a", "b", "a", "b", "a", "b"]),
            "flag": [True, False, True, False, True, False],
            "const": [1.0] * 6,
            "obj": ["x", "y", "x", "y", "x", "y"],
        }
    )
    y = pd.Series(["pos", "neg", "pos", "neg", "pos", "neg"])
    Xn, yn, names = datasets._prepare_frame(X, y, "classification", None, 0)
    assert names == ["num", "cat", "flag", "obj"]
    assert Xn.shape == (6, 4) and Xn.dtype == float
    assert Xn[2, 0] == 4.0  # median of the observed values
    assert set(Xn[:, 1]) == {0.0, 1.0} and set(Xn[:, 2]) == {0.0, 1.0}
    assert yn.dtype.kind == "i" and yn.tolist() == [0, 1, 0, 1, 0, 1]


def test_prepare_frame_target_coding():
    X = pd.DataFrame({"a": np.arange(6, dtype=float), "b": np.arange(6, dtype=float) ** 2})
    # labels already coded 0 and 1 keep their coding
    _, y01, _ = datasets._prepare_frame(X, pd.Series([0, 1, 1, 0, 1, 0]), "classification", None, 0)
    assert y01.tolist() == [0, 1, 1, 0, 1, 0]
    # more than two classes: the most frequent class against the rest
    _, ymulti, _ = datasets._prepare_frame(X, pd.Series(["a", "b", "c", "a", "b", "a"]), "classification", None, 0)
    assert ymulti.tolist() == [1, 0, 0, 1, 0, 1]
    # regression targets are coerced and rows with a missing target are dropped
    Xr, yr, _ = datasets._prepare_frame(X, pd.Series(["1.5", "2.5", "bad", "4.5", "5.5", "6.5"]), "regression", None, 0)
    assert yr.tolist() == [1.5, 2.5, 4.5, 5.5, 6.5] and Xr.shape == (5, 2)
    # subsampling to max_rows is reproducible
    Xs, ys, _ = datasets._prepare_frame(X, pd.Series(np.arange(6, dtype=float)), "regression", 4, 3)
    Xs2, ys2, _ = datasets._prepare_frame(X, pd.Series(np.arange(6, dtype=float)), "regression", 4, 3)
    assert Xs.shape == (4, 2) and np.array_equal(ys, ys2) and np.array_equal(Xs, Xs2)
    assert np.array_equal(Xs[:, 0], ys)  # rows stay aligned with their targets


def test_load_builtin_breast_cancer_offline():
    ds = datasets.load_benchmark("breast_cancer")
    assert isinstance(ds, datasets.Dataset)
    assert ds.X.shape == (569, 30) and ds.n_features == 30 and ds.task == "classification"
    assert ds.y.dtype.kind == "i" and set(ds.y.tolist()) == {0, 1}
    assert int(ds.y.sum()) == 212  # malignant tumours are the explained positive class
    assert ds.meta["positive_class"] == "malignant" and ds.meta["source"] == "sklearn" and ds.meta["n_rows"] == 569
    assert ds.meta["sklearn_loader"] == "load_breast_cancer" and ds.meta["target_names"] == ["malignant", "benign"]
    assert ds.groups is None and len(ds.feature_names) == 30
    small = datasets.load_benchmark("breast_cancer", max_rows=100, random_state=1)
    assert small.X.shape == (100, 30) and small.meta["n_rows"] == 100
    inferred = datasets.load_benchmark("breast_cancer", task=None, source="sklearn")
    assert inferred.task == "classification"


def test_load_benchmark_rejects_unknown_sources_and_names():
    with pytest.raises(ValueError, match="source must be"):
        datasets.load_benchmark("breast_cancer", source="csv")
    with pytest.raises(ValueError, match="Unknown scikit-learn dataset"):
        datasets.load_benchmark("spambase", source="sklearn")


def test_load_openml_benchmark_with_a_stub(monkeypatch):
    calls = []
    rng = np.random.default_rng(0)

    def fake_fetch_openml(name, version, as_frame, parser, data_home):
        calls.append({"name": name, "version": version, "as_frame": as_frame, "data_home": data_home})
        data = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
        if name == "spambase":
            target = pd.Series(pd.Categorical(["spam", "ham"] * 20))
        elif name == "ionosphere":
            target = pd.Series(["g", "b"] * 20)  # plain strings: object dtype before pandas 3, string dtype after
        else:
            target = pd.Series(rng.normal(size=40))
        return Bunch(data=data, target=target, details={"id": 123 if name == "spambase" else 456})

    monkeypatch.setattr(sklearn.datasets, "fetch_openml", fake_fetch_openml)
    ds = datasets.load_benchmark("spambase", data_home="/tmp/openml-cache")
    assert calls[-1] == {"name": "spambase", "version": 1, "as_frame": True, "data_home": "/tmp/openml-cache"}
    assert ds.task == "classification" and set(ds.y.tolist()) == {0, 1} and ds.X.shape == (40, 3)
    assert ds.meta == {"openml_id": 123, "version": 1, "source": "openml", "n_rows": 40}
    # a string target is a classification task whatever its pandas dtype
    strings = datasets.load_benchmark("ionosphere", task=None)
    assert strings.task == "classification" and set(strings.y.tolist()) == {0, 1} and strings.y.dtype.kind == "i"
    # a name outside the list defaults to OpenML and the task is inferred from the target
    other = datasets.load_benchmark("unlisted", version=3)
    assert calls[-1]["name"] == "unlisted" and calls[-1]["version"] == 3
    assert other.task == "regression" and other.y.dtype == float and other.meta["openml_id"] == 456


def test_load_benchmarks_filters_names_and_skips_failures(monkeypatch, capsys):
    def fake_load(name, version, task, source, **kwargs):
        if name == "spambase":
            raise RuntimeError("no network")
        d = 5 if name == "sonar" else 20
        X = np.zeros((10, d))
        return datasets.Dataset(name, X, np.zeros(10, dtype=int), task, [f"x{j}" for j in range(d)], None, {"kwargs": kwargs})

    monkeypatch.setattr(datasets, "load_benchmark", fake_load)
    out = datasets.load_benchmarks(names=["breast_cancer", "spambase", "sonar"], max_rows=50)
    assert [d.name for d in out] == ["breast_cancer"]
    assert out[0].meta["kwargs"] == {"max_rows": 50}
    text = capsys.readouterr().out
    assert "skipping spambase: RuntimeError: no network" in text
    assert "skipping sonar: 5 features outside [15, 100]" in text
    assert datasets.load_benchmarks(names=[]) == []


def _write_questionnaire(tmp_path, task="regression", positive_class=None, covariates=True, name=True):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(1, 6, size=(30, 5)).astype(float), columns=["i1", "i2", "j1", "j2", "cov1"])
    if task == "regression":
        df["outcome"] = df[["i1", "i2"]].sum(axis=1) + rng.normal(size=30)
    else:
        df["outcome"] = np.where(df["i1"] > 3, "yes", "no")
    df.loc[0, "i1"] = np.nan  # one incomplete row is dropped
    csv = tmp_path / "q.csv"
    df.to_csv(csv, index=False)
    spec = {"outcome": "outcome", "task": task, "subscales": {"s1": ["i1", "i2"], "s2": ["j1", "j2"]}}
    if covariates:
        spec["covariates"] = ["cov1"]
    if positive_class is not None:
        spec["positive_class"] = positive_class
    if name:
        spec["name"] = "demo"
    spec_path = tmp_path / "q_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf8")
    return csv, spec_path, df


def test_load_questionnaire_regression(tmp_path):
    csv, spec_path, df = _write_questionnaire(tmp_path)
    ds = datasets.load_questionnaire(str(csv), str(spec_path))
    assert ds.name == "demo" and ds.task == "regression"
    assert ds.feature_names == ["i1", "i2", "j1", "j2", "cov1"]
    assert ds.X.shape == (29, 5) and ds.y.shape == (29,) and ds.y.dtype == float
    assert ds.groups == {"s1": ["i1", "i2"], "s2": ["j1", "j2"], "covariates": ["cov1"]}
    assert ds.meta == {"n_rows": 29}
    np.testing.assert_allclose(ds.y, df["outcome"].to_numpy()[1:])


def test_load_questionnaire_classification_and_defaults(tmp_path):
    csv, spec_path, df = _write_questionnaire(tmp_path, task="classification", positive_class="yes", covariates=False, name=False)
    ds = datasets.load_questionnaire(str(csv), str(spec_path))
    assert ds.name == "q" and ds.task == "classification"
    assert ds.feature_names == ["i1", "i2", "j1", "j2"] and "covariates" not in ds.groups
    assert ds.y.tolist() == (df["outcome"].to_numpy()[1:] == "yes").astype(int).tolist()
    # without a positive class the labels are factorized in order of appearance
    spec = json.loads(spec_path.read_text(encoding="utf8"))
    del spec["positive_class"]
    spec_path.write_text(json.dumps(spec), encoding="utf8")
    ds2 = datasets.load_questionnaire(str(csv), str(spec_path))
    assert set(ds2.y.tolist()) == {0, 1} and ds2.y[0] == 0


def test_load_questionnaire_reports_missing_columns(tmp_path):
    csv, spec_path, _ = _write_questionnaire(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf8"))
    spec["subscales"]["s3"] = ["k1"]
    spec["outcome"] = "score"
    spec_path.write_text(json.dumps(spec), encoding="utf8")
    with pytest.raises(ValueError, match=r"missing.*\['k1', 'score'\]"):
        datasets.load_questionnaire(str(csv), str(spec_path))


def test_stratified_splits():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=100) < 0.3).astype(int)
    splits = list(datasets.stratified_splits(y, "classification", n_repeats=3, random_state=5))
    assert [s[0] for s in splits] == [0, 1, 2]
    for _, tr, tu, te in splits:
        assert len(tr) == 60 and len(tu) == 20 and len(te) == 20
        assert set(tr) | set(tu) | set(te) == set(range(100))
        assert not (set(tr) & set(tu)) and not (set(tr) & set(te)) and not (set(tu) & set(te))
        # stratification keeps the class proportion in every part
        for part in (tr, tu, te):
            assert abs(y[part].mean() - y.mean()) < 0.06
    again = list(datasets.stratified_splits(y, "classification", n_repeats=3, random_state=5))
    assert all(np.array_equal(a[1], b[1]) and np.array_equal(a[3], b[3]) for a, b in zip(splits, again))
    assert not np.array_equal(splits[0][1], splits[1][1])
    yr = rng.normal(size=50)
    (r, tr, tu, te), = list(datasets.stratified_splits(yr, "regression", n_repeats=1, proportions=(0.5, 0.3, 0.2)))
    assert r == 0 and len(tr) == 25 and len(tu) == 15 and len(te) == 10


def test_pmlb_loader_reads_cached_file(tmp_path):
    """A cached PMLB file is read offline, the target column is split off and the spec names are resolved."""
    import gzip

    from pfca.evaluation import datasets

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(40, 16)), columns=[f"f{i}" for i in range(16)])
    frame["target"] = (frame["f0"] > 0).astype(int)
    home = tmp_path / "pmlb"
    home.mkdir()
    with gzip.open(home / "toy.tsv.gz", "wt") as fh:
        frame.to_csv(fh, sep="\t", index=False)
    ds = datasets.load_benchmark("toy", task="classification", source="pmlb", data_home=str(home))
    assert ds.n_features == 16 and ds.X.shape[0] == 40 and ds.task == "classification"
    assert set(np.unique(ds.y)) <= {0, 1} and ds.meta["source"] == "pmlb"
    spec = datasets.benchmark_spec("pol", source="pmlb")
    assert spec is not None and spec["pmlb_name"] == "201_pol"
    assert datasets.benchmark_spec("pol")["source"] == "openml"
    names = [b["name"] for b in datasets.PMLB_BENCHMARKS]
    assert len(names) == 15 and len(set(names)) == 15 and names[0] == "breast_cancer"
