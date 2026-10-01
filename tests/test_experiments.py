"""End to end checks of the experiment scripts.

Every script is run through its command line interface in a subprocess, the
way the README describes it, on the smallest configuration that exercises its
paths: two repetitions give the replicate of the calibration metrics, the
redundancy family runs the duplication experiment, two model classes give a
disagreement index. The runs write to a temporary directory and are shared by
the tests of the module through module scoped fixtures. The module is marked
``slow`` (about one minute in total) and is deselected with ``-m "not slow"``.
The shared helpers of ``experiments/common.py`` are tested in
``test_experiments_common.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

PFCA_SMALL = {"n_concepts_grid": [2, 3], "fuzzifier_grid": [2.0], "alpha_grid": [0.0, 0.5], "solver": "grid", "knee_method": "utopia"}

PHASE_A_CONFIG = {
    "families": ["additive", "redundancy"],
    "n_samples": [120],
    "n_features": [6],
    "rhos": [0.6],
    "n_seeds": 1,
    "n_explain": 8,
    "tune_fraction": 0.34,
    "model_classes": ["gbm"],
    "n_resamples": 2,
    "background_size": 20,
    "explainer": "auto",
    "methods": ["pfca", "pfca_fixed", "shap", "grouped_shap", "lime"],
    "pfca": {**PFCA_SMALL, "partition_source": "data", "fuzzy_shape": "trapezoidal", "disagreement_threshold": 0.5},
    "perturbation_instances": 2,
    "n_perturbations": 1,
    "imputation": "conditional",
    "grouped_background": None,
    "trace_memory": False,
    "n_jobs": 1,
}

PHASE_B_CONFIG = {
    "datasets": ["breast_cancer"],
    "min_features": 15,
    "max_features": 100,
    "max_rows": 120,
    "n_repeats": 3,
    "n_explain": 6,
    "seed": 0,
    "model_classes": ["gbm"],
    "n_resamples": 2,
    "background_size": 20,
    "explainer": "auto",
    "tuning_grid": {"max_depth": [2], "n_estimators": [30]},
    "methods": ["pfca", "shap", "bootstrapped_shap", "lime"],
    "pfca": dict(PFCA_SMALL),
    "perturbation_instances": 2,
    "n_perturbations": 1,
    "imputation": "conditional",
    "grouped_background": 20,
    "save_explanations": False,
    "n_jobs": 1,
}

PHASE_C_CONFIG = {
    "n_repeats": 2,
    "n_explain": 6,
    "model_classes": ["gbm"],
    "n_resamples": 2,
    "background_size": 20,
    "explainer": "auto",
    "methods": ["pfca", "pfca_apriori", "shap", "grouped_shap"],
    "pfca": {**PFCA_SMALL, "partition_source": "both"},
    "perturbation_instances": 2,
    "n_perturbations": 1,
    "imputation": "conditional",
    "grouped_background": None,
    "example_instances": [0],
    "split_scheme": "holdout",
    "seed": 0,
    "split_seed": 0,
    "n_jobs": 1,
}

PHASE_D_CONFIG = {
    "csv_path": "data/raw/questionnaire.csv",
    "spec_path": "data/raw/questionnaire_spec.json",
    "synthetic": {"family": "additive", "n_samples": 150, "n_features": 6, "rho": 0.6, "tune_fraction": 0.34},
    "n_instances": 2,
    "n_experts": 6,
    "n_candidates": 6,
    "model_classes": ["gbm", "rf"],
    "n_resamples": 2,
    "background_size": 20,
    "explainer": "auto",
    "pfca": {**PFCA_SMALL, "disagreement_threshold": 0.5},
    "reliable_sign_confidence": 0.9,
    "shap_top_features": 15,
    "likert_min": 1,
    "likert_max": 7,
    "ordinal_adjust_expert": True,
    "seed": 0,
    "n_jobs": 1,
    "simulation": {
        "expert_sd": 0.5,
        "rating_sd": 1.0,
        "rating_means": {
            "rating_confidence_calibration": {"shap": 3.5, "concept": 4.0, "fuzzy": 5.5},
            "rating_ease": {"shap": 4.5, "concept": 5.0, "fuzzy": 4.5},
            "rating_actionability": {"shap": 4.0, "concept": 4.5, "fuzzy": 5.0},
        },
        "accuracy": {"shap": 0.35, "concept": 0.45, "fuzzy": 0.85},
    },
}

SENSITIVITY_CONFIG = {
    "cells": [{"family": "additive", "n_samples": 120, "n_features": 6, "rho": 0.6}],
    "n_seeds": 1,
    "n_explain": 8,
    "tune_fraction": 0.34,
    "model_classes": ["gbm", "rf"],
    "background_size": 20,
    "explainer": "auto",
    "resamples": [2, 3],
    "model_class_counts": [1, 2],
    "shapes": ["trapezoidal", "triangular"],
    "percentiles": [{"support": [5, 95], "core": [25, 75]}, {"support": [10, 90], "core": [30, 70]}],
    "distances": ["correlation", "loading"],
    "solvers": ["grid", "nsga2"],
    "knee_methods": ["utopia", "hyperplane"],
    "label_sets": [{"name": "default"}, {"name": "three_labels", "centers": [-1.0, 0.0, 1.0], "names": ["negative", "negligible", "positive"]}],
    "defaults": {
        "n_resamples": 100,
        "n_model_classes": 3,
        "fuzzy_shape": "trapezoidal",
        "support_percentiles": [5, 95],
        "core_percentiles": [25, 75],
        "concept_distance": "correlation",
        "solver": "grid",
        "knee_method": "utopia",
    },
    "nsga_pop_size": 8,
    "nsga_generations": 2,
    "include_apriori": False,
    "pfca": {"n_concepts_grid": [2, 3], "fuzzifier_grid": [2.0], "alpha_grid": [0.0, 0.5], "disagreement_threshold": 0.5},
    "perturbation_instances": 0,
    "n_perturbations": 0,
    "imputation": "conditional",
    "n_jobs": 1,
}
SENSITIVITY_FACTORS = ["resamples", "model_classes", "solver", "knee", "labels"]
HAS_LIME = importlib.util.find_spec("lime") is not None


def expected_methods(methods: list[str]) -> list[str]:
    """Methods a script runs: LIME is skipped when the optional package is missing."""
    return [m for m in methods if m != "lime" or HAS_LIME]


def run_script(name: str, *args, expect_failure: bool = False) -> subprocess.CompletedProcess:
    """Run an experiment script from the repository root and fail the test on any error."""
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, str(EXPERIMENTS / name), *map(str, args)]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
    report = f"{' '.join(cmd)}\nexit code {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    if expect_failure:
        assert proc.returncode != 0, report
    else:
        assert proc.returncode == 0, report
        # the scripts record a failing method and continue with the others; here none may fail
        assert "Traceback" not in proc.stderr, report
    return proc


def write_config(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")
    return path


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("experiments")
    (base / "results").mkdir()
    (base / "configs").mkdir()
    return base


# ----------------------------------------------------------------------
# Phase A
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase_a_run(work):
    config = write_config(work / "configs" / "phase_a.yaml", PHASE_A_CONFIG)
    args = ("--config", config, "--results", work / "results", "--name", "phase_a", "--only-family", "redundancy", "--max-cells", "1")
    proc = run_script("run_phase_a.py", *args)
    return {"dir": work / "results" / "phase_a", "proc": proc, "args": args}


def test_phase_a_writes_the_metrics_of_every_method(phase_a_run):
    out = phase_a_run["dir"]
    assert "[phase A] 1 cells" in phase_a_run["proc"].stdout  # --only-family and --max-cells restrict the design
    assert "[phase A] finished" in phase_a_run["proc"].stdout
    df = pd.read_csv(out / "metrics.csv")
    assert df["method"].tolist() == expected_methods(PHASE_A_CONFIG["methods"])
    if not HAS_LIME:
        assert "skipping lime" in phase_a_run["proc"].stdout
    assert "error" not in df.columns and "duplication_error" not in df.columns
    pf = df[df["method"] == "pfca"]
    assert pf["n_front_distinct"].ge(1).all() and pf["n_feasible"].ge(pf["n_front"]).all()  # front diagnostics of Section 10
    assert "objective_corr_fidelity_complexity" in df.columns
    assert (df["family"] == "redundancy").all() and df["n_features"].eq(6).all() and df["n_features_total"].gt(6).all()
    assert np.isfinite(df["recovery_error"]).all()
    assert np.isfinite(df.loc[df["method"] != "lime", "rank_stability"]).all()  # LIME has no resample pool, so its rank stability is undefined
    assert np.isfinite(df["duplication_change"]).all()  # the redundancy family reruns every method without the duplicates
    assert df.loc[df["method"] == "pfca", "n_front"].ge(1).all()
    assert df.loc[df["method"] == "pfca_fixed", "n_front"].eq(1).all()
    assert np.isfinite(df["pool_wall_seconds"]).all() and df["reference_params"].notna().all()
    per = pd.read_csv(out / "per_instance.csv")
    assert {"instance", "metric", "value", "family", "n_samples", "n_features", "rho", "seed", "method"} <= set(per.columns)
    assert set(per["method"]) <= set(PHASE_A_CONFIG["methods"]) and per["instance"].max() < PHASE_A_CONFIG["n_explain"]
    frozen = read_yaml(out / "config.yaml")
    assert frozen["methods"] == expected_methods(PHASE_A_CONFIG["methods"]) and frozen["n_jobs"] == 1 and frozen["n_seeds"] == 1
    env = read_json(out / "environment.json")
    assert {"versions", "started", "cwd", "argv", "seeds_from_config"} <= set(env)
    assert env["versions"]["pfca"] and "--only-family" in env["argv"]


def test_phase_a_resumes_finished_cells(phase_a_run):
    before = (phase_a_run["dir"] / "metrics.csv").read_text(encoding="utf8")
    proc = run_script("run_phase_a.py", *phase_a_run["args"])
    assert "done in" not in proc.stdout and "[phase A] finished" in proc.stdout
    assert (phase_a_run["dir"] / "metrics.csv").read_text(encoding="utf8") == before


# ----------------------------------------------------------------------
# Phase B
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase_b_run(work):
    config = write_config(work / "configs" / "phase_b.yaml", PHASE_B_CONFIG)
    proc = run_script("run_phase_b.py", "--config", config, "--results", work / "results", "--name", "phase_b", "--datasets", "breast_cancer", "--max-repeats", "2")
    return {"dir": work / "results" / "phase_b", "proc": proc}


def test_phase_b_runs_the_built_in_dataset_offline(phase_b_run):
    out = phase_b_run["dir"]
    assert "breast_cancer: 120 rows, 30 features, classification" in phase_b_run["proc"].stdout
    assert "split scheme repeated_cv" in phase_b_run["proc"].stdout  # 120 rows are below the small sample threshold
    methods = expected_methods(PHASE_B_CONFIG["methods"])
    df = pd.read_csv(out / "metrics.csv")
    assert len(df) == 2 * len(methods) and sorted(df["repeat"].unique()) == [0, 1]
    assert set(df["method"]) == set(methods) and (df["dataset"] == "breast_cancer").all()
    assert "error" not in df.columns
    assert np.isfinite(df["deletion_auc"]).all() and np.isfinite(df["surrogate_r2"]).all()
    pfca_rows = df[df["method"] == "pfca"].set_index("repeat")
    assert np.isnan(pfca_rows.loc[0, "support_coverage_replicate"])  # no repetition holds the first explained fold out of training
    assert np.isfinite(pfca_rows.loc[1, "support_coverage_replicate"]) and np.isfinite(pfca_rows.loc[1, "sign_confidence_ece"])
    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 2 and (runs["n_train"] + runs["n_tune"] + runs["n_test"]).eq(120).all()
    assert runs["seed"].nunique() == 2 and (runs["split_scheme"] == "repeated_cv").all()
    # with two cross validation repetitions only the second finds a repetition that holds its explained fold out of training
    assert runs.set_index("repeat")["replicate_repeat"].tolist() == [1, 0] and runs.set_index("repeat").loc[1, "n_replicate_instances"] == 6
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["method"]) == set(methods) and summary["n_repeats"].eq(2).all()
    frozen = read_yaml(out / "config.yaml")
    assert frozen["n_repeats"] == 2 and frozen["datasets"] == ["breast_cancer"]  # --max-repeats caps the configuration


# ----------------------------------------------------------------------
# Phase C
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def questionnaire(work):
    data = work / "data"
    proc = run_script("make_demo_questionnaire.py", "--n", "120", "--seed", "0", "--n-constructs", "3", "--items-per-construct", "3", "--out-dir", data, "--name", "demo")
    return {"csv": data / "demo.csv", "spec": data / "demo_spec.json", "proc": proc}


def test_demo_questionnaire_files(questionnaire):
    df = pd.read_csv(questionnaire["csv"])
    spec = read_json(questionnaire["spec"])
    assert len(df) == 120
    assert len(spec["subscales"]) == 3 and all(len(items) == 3 for items in spec["subscales"].values())
    items = [c for cols in spec["subscales"].values() for c in cols]
    assert set(items) <= set(df.columns) and spec["outcome"] in df.columns
    assert all(c in df.columns for c in spec.get("covariates", []))
    assert spec.get("task", "regression") == "regression"
    assert "wrote" in questionnaire["proc"].stdout


@pytest.fixture(scope="module")
def phase_c_run(work, questionnaire):
    pytest.importorskip("matplotlib")
    cfg = {"csv_path": str(questionnaire["csv"]), "spec_path": str(questionnaire["spec"]), **PHASE_C_CONFIG}
    config = write_config(work / "configs" / "phase_c.yaml", cfg)
    args = ("--config", config, "--results", work / "results", "--name", "phase_c", "--repeats", "2")
    proc = run_script("run_phase_c.py", *args)
    return {"dir": work / "results" / "phase_c", "proc": proc, "args": args}


def test_phase_c_outputs(phase_c_run):
    out = phase_c_run["dir"]
    assert "task regression" in phase_c_run["proc"].stdout and "2 repetitions, split scheme holdout" in phase_c_run["proc"].stdout
    df = pd.read_csv(out / "metrics.csv")
    assert len(df) == 2 * len(PHASE_C_CONFIG["methods"]) and sorted(df["repeat"].unique()) == [0, 1]
    assert set(df["method"]) == set(PHASE_C_CONFIG["methods"]) and "error" not in df.columns
    assert (df.loc[df["method"] == "pfca_apriori", "partition"] == "apriori").all()
    assert np.isfinite(df["deletion_auc"]).all()
    summary = pd.read_csv(out / "summary_by_method.csv", header=[0, 1], index_col=0)  # metric by statistic columns, one row per method
    assert set(summary.index) == set(PHASE_C_CONFIG["methods"])
    assert ("deletion_auc", "mean") in summary.columns and summary[("deletion_auc", "count")].eq(2).all()
    for name in ["reliability.csv", "reliability_summary.csv", "reliability_repeat00.csv", "reliability_repeat01.csv", "per_instance.csv"]:
        assert (out / name).stat().st_size > 0, name
    for r in ("00", "01"):
        assert read_json(out / f"pfca_explanation_repeat{r}.json")
        assert (out / f"membership_repeat{r}.png").stat().st_size > 0
        assert (out / f"overlap_repeat{r}.png").stat().st_size > 0
        text = (out / f"example_repeat{r}_instance00.txt").read_text(encoding="utf8")
        assert text.strip()
        for suffix in ("fuzzy_attribution", "linguistic_bars", "shap_bars"):
            assert (out / f"example_repeat{r}_instance00_{suffix}.png").stat().st_size > 0


def test_phase_c_resumes_finished_repeats(phase_c_run):
    before = (phase_c_run["dir"] / "metrics.csv").read_text(encoding="utf8")
    proc = run_script("run_phase_c.py", *phase_c_run["args"])
    assert "done in" not in proc.stdout and "[phase C] finished" in proc.stdout
    assert (phase_c_run["dir"] / "metrics.csv").read_text(encoding="utf8") == before


def test_phase_c_reports_missing_questionnaire_files(work, phase_c_run):
    proc = run_script("run_phase_c.py", *phase_c_run["args"], "--csv", work / "data" / "missing.csv", expect_failure=True)
    assert "Questionnaire files not found" in proc.stderr


# ----------------------------------------------------------------------
# Phase D
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase_d_run(work):
    pytest.importorskip("matplotlib")
    pytest.importorskip("statsmodels")
    config = write_config(work / "configs" / "phase_d.yaml", PHASE_D_CONFIG)
    generate = run_script("run_phase_d.py", "generate", "--synthetic", "--config", config, "--results", work / "results", "--name", "phase_d")
    analyze = run_script("run_phase_d.py", "analyze", "--results", work / "results", "--name", "phase_d", "--simulate")
    return {"dir": work / "results" / "phase_d", "generate": generate, "analyze": analyze}


def test_phase_d_generates_materials_sheets_and_keys(phase_d_run):
    out = phase_d_run["dir"]
    n_inst, n_exp = PHASE_D_CONFIG["n_instances"], PHASE_D_CONFIG["n_experts"]
    assert f"wrote {4 * n_inst} material files, {3 * n_inst * n_exp} response rows for {n_exp} experts, key for {n_inst} predictions" in phase_d_run["generate"].stdout
    mapping = pd.read_csv(out / "private" / "format_mapping.csv")
    assert sorted(mapping["format_code"]) == ["F1", "F2", "F3"] and set(mapping["format"]) == {"shap", "concept", "fuzzy"}
    fuzzy_code = mapping.loc[mapping["format"] == "fuzzy", "format_code"].iloc[0]
    pngs = sorted((out / "materials").glob("instance_P*_F*.png"))
    txts = sorted((out / "materials").glob("instance_P*_F*.txt"))
    assert len(pngs) == 3 * n_inst and all(p.stat().st_size > 0 for p in pngs)
    assert len(txts) == n_inst and all(t.name.endswith(f"_{fuzzy_code}.txt") for t in txts)  # the text exists for the fuzzy format only
    assert (out / "materials" / "concept_dictionary.txt").read_text(encoding="utf8").strip()
    assert (out / "materials" / "instructions.txt").read_text(encoding="utf8").strip()
    sheet = pd.read_csv(out / "response_sheet.csv")
    assert len(sheet) == n_exp * n_inst * 3 and sheet["expert_id"].nunique() == n_exp
    pairs = sheet.drop_duplicates(["expert_id", "instance_id", "format_code"]).groupby("expert_id").size()
    assert (pairs == n_inst * 3).all()  # every expert sees every prediction in every format once
    assert len(list((out / "materials" / "sheets").glob("response_sheet_*.csv"))) == n_exp
    key = pd.read_csv(out / "private" / "answer_key.csv")
    assert len(key) == n_inst and {"instance_id", "candidate_index", "model_output", "q1_key", "q2_key", "q3_key"} <= set(key.columns)
    assert key["instance_id"].tolist() == [f"P{j + 1:02d}" for j in range(n_inst)]
    design = pd.read_csv(out / "private" / "design.csv")
    assert len(design) == n_exp
    blocks = design["format_sequence"].str.split("|", expand=True)
    assert blocks.shape[1] == 3
    for col in blocks.columns:  # balanced Latin square: every block position holds every format equally often
        assert sorted(blocks[col].value_counts().tolist()) == [n_exp // 3] * 3
    seeds = read_json(out / "private" / "seeds.json")
    assert seeds["seed"] == 0 and sorted(seeds["format_codes"].values()) == ["F1", "F2", "F3"]
    assert len(seeds["chosen_candidates"]) == n_inst and seeds["retained_concepts"]
    assert read_json(out / "private" / "explanation.json")
    instances = pd.read_csv(out / "private" / "instances.csv")
    assert len(instances) == n_inst and instances.shape[1] == 2 + PHASE_D_CONFIG["synthetic"]["n_features"]
    frozen = read_yaml(out / "config.yaml")
    assert frozen["use_synthetic"] is True and frozen["n_candidates"] == PHASE_D_CONFIG["n_candidates"]


def test_phase_d_simulated_analysis(phase_d_run):
    out = phase_d_run["dir"]
    n_rows = PHASE_D_CONFIG["n_experts"] * PHASE_D_CONFIG["n_instances"] * 3
    assert "[phase D] analyze finished" in phase_d_run["analyze"].stdout
    analysis = out / "analysis_simulated"
    for name in [
        "scored_responses.csv",
        "descriptives_ratings.csv",
        "descriptives_accuracy.csv",
        "friedman.csv",
        "nemenyi_pairs.csv",
        "wilcoxon_vs_fuzzy.csv",
        "ordinal_models.csv",
        "ordinal_models.txt",
        "accuracy_mixed_model.csv",
        "accuracy_mixed_model.txt",
        "ratings_by_format.png",
        "summary.txt",
    ]:
        assert (analysis / name).stat().st_size > 0, name
    assert "simulated" in (analysis / "summary.txt").read_text(encoding="utf8").lower()
    filled = pd.read_csv(out / "responses_simulated.csv")
    assert len(filled) == n_rows
    scored = pd.read_csv(analysis / "scored_responses.csv")
    assert len(scored) == n_rows and set(scored["format"]) == {"shap", "concept", "fuzzy"}
    desc = pd.read_csv(analysis / "descriptives_ratings.csv")
    assert set(desc["format"]) == {"shap", "concept", "fuzzy"}


# ----------------------------------------------------------------------
# Sensitivity analyses
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def sensitivity_run(work):
    config = write_config(work / "configs" / "sensitivity.yaml", SENSITIVITY_CONFIG)
    proc = run_script("run_sensitivity.py", "--config", config, "--results", work / "results", "--name", "sensitivity", "--factors", ",".join(SENSITIVITY_FACTORS))
    return {"dir": work / "results" / "sensitivity", "proc": proc, "config": config}


def test_sensitivity_outputs(sensitivity_run):
    out = sensitivity_run["dir"]
    assert "1 cells x 1 seeds x 10 settings" in sensitivity_run["proc"].stdout
    df = pd.read_csv(out / "metrics.csv")
    assert df["factor"].tolist() == [f for f in SENSITIVITY_FACTORS for _ in range(2)]
    assert df["setting"].tolist() == ["B2", "B3", "M1", "M2", "grid", "nsga2", "utopia", "hyperplane", "default", "three_labels"]
    labels = df[df["factor"] == "labels"].set_index("setting")
    assert labels.loc["default", "label_agreement_with_default"] == 1.0 and labels.loc["default", "n_labels"] == 5
    assert labels.loc["three_labels", "n_labels"] == 3 and np.isnan(labels.loc["three_labels", "label_agreement_with_default"])
    assert np.isfinite(df["label_sign_agreement_with_default"]).all() and df["label_entropy"].between(0, 1).all()
    assert labels.loc["default", "recovery_error"] == labels.loc["three_labels", "recovery_error"]  # the selection does not depend on the labels
    assert "error" not in df.columns
    assert np.isfinite(df["recovery_error"]).all() and np.isfinite(df["rank_stability"]).all()
    assert np.isfinite(df.loc[df["setting"] == "nsga2", "front_recovery"]).all()  # the front comparison exists on the NSGA II row only
    assert df.loc[df["setting"] != "nsga2", "front_recovery"].isna().all()
    assert np.isfinite(df.loc[df["setting"] == "B2", "quantile_mad_to_largest_b"]).all()
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 10 and summary["n_seeds"].eq(1).all()
    runs = pd.read_csv(out / "runs.csv")
    assert len(runs) == 1
    frozen = read_yaml(out / "config.yaml")
    assert frozen["factors_run"] == SENSITIVITY_FACTORS
    assert frozen["resolved_model_classes"] == ["gbm", "rf"]
    assert frozen["resolved_default"]["n_resamples"] == 3 and frozen["resolved_default"]["n_model_classes"] == 2  # defaults fall back to what is listed


def test_sensitivity_rejects_unknown_factors(work, sensitivity_run):
    proc = run_script("run_sensitivity.py", "--config", sensitivity_run["config"], "--results", work / "results", "--name", "sensitivity_bad", "--factors", "resamples,bogus", expect_failure=True)
    assert "Unknown factors ['bogus']" in proc.stderr


# ----------------------------------------------------------------------
# Example front and figures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def example_front_run(work):
    pytest.importorskip("matplotlib")
    fig_dir = work / "out" / "figures"
    proc = run_script("example_front.py", "--n-samples", "100", "--n-features", "9", "--n-resamples", "2", "--n-explain", "5", "--background-size", "20", "--n-jobs", "1", "--out", fig_dir, "--results", work / "results")
    return {"figures": fig_dir, "dir": work / "results" / "example_front", "proc": proc}


def test_example_front_outputs(example_front_run):
    fig, out = example_front_run["figures"], example_front_run["dir"]
    assert (fig / "example_front.png").stat().st_size > 0 and (fig / "example_front.pdf").stat().st_size > 0
    table = pd.read_csv(out / "configurations.csv")
    assert table["knee"].sum() == 1 and table["pareto"].sum() >= 1 and table.loc[table["knee"], "pareto"].all()
    points = pd.read_csv(out / "front_points.csv")
    assert len(points) == 3
    assert read_json(out / "explanation_knee.json") and read_json(out / "seeds.json")
    assert (out / "explanations.txt").read_text(encoding="utf8").strip()
    for name in ("sparsest", "knee", "most faithful"):
        assert f"[example front] {name}:" in example_front_run["proc"].stdout


@pytest.fixture(scope="module")
def figures_run(work, phase_a_run, phase_b_run, phase_c_run, sensitivity_run, example_front_run):
    proc = run_script("make_figures.py", "--results", work / "results", "--phase-a", "phase_a", "--phase-b", "phase_b", "--phase-c", "phase_c", "--sensitivity", "sensitivity", "--out", work / "out")
    return {"out": work / "out", "dir": work / "results" / "make_figures", "proc": proc}


def test_make_figures_writes_every_item(figures_run):
    out = figures_run["out"]
    items = pd.read_csv(figures_run["dir"] / "items.csv")
    assert len(items) == 14
    assert items["status"].eq("written").all(), items.loc[items["status"] != "written", ["item", "title", "detail"]].to_string()
    assert "14 of 14 items written" in figures_run["proc"].stdout
    for name in [
        "pipeline_schematic.png",
        "pipeline_schematic.pdf",
        "phase_a_recovery_stability.png",
        "phase_a_duplication.png",
        "reliability_sign_confidence.png",
        "interval_coverage.png",
        "example_front.png",
        "phase_c_example_linguistic_bars.png",
        "phase_c_subscale_overlap.png",
    ]:
        assert (out / "figures" / name).stat().st_size > 0, name
    for name in [
        "properties.csv",
        "properties.md",
        "phase_a_recovery_stability.csv",
        "phase_a_duplication.csv",
        "phase_b_metrics.csv",
        "phase_b_friedman.csv",
        "phase_b_friedman_pairwise.csv",
        "phase_b_wilcoxon_holm.csv",
        "phase_b_table.md",
        "reliability_bins.csv",
        "interval_coverage.csv",
        "runtime_memory.csv",
        "sensitivity_summary.csv",
        "phase_a_mixed_models.csv",
        "phase_a_mixed_models.md",
        "success_criteria.csv",
        "success_criteria.md",
        "pareto_front_diagnostics.csv",
        "pareto_front_diagnostics.md",
    ]:
        assert (out / "tables" / name).stat().st_size > 0, name
    properties = pd.read_csv(out / "tables" / "properties.csv")
    assert len(properties) >= 6
    mixed = pd.read_csv(out / "tables" / "phase_a_mixed_models.csv")
    assert (mixed["model"] == "ordinary least squares (single seed, no random effect)").all()  # the tiny run has one seed
    criteria = pd.read_csv(out / "tables" / "success_criteria.csv")
    assert sorted(criteria["criterion"].unique()) == [1, 2, 3] and criteria["status"].isin(["met", "not met"]).any()
    front = pd.read_csv(out / "tables" / "pareto_front_diagnostics.csv")
    assert "all families" in set(front["group"]) and {"grid", "nsga2"} <= set(front["group"])
