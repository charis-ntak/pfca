# Results directory

The experiment scripts under `experiments/` write their outputs here, one folder per run named by the `--name` option (`results/<name>/`). Everything in this directory except this file is ignored by git: result files are rebuilt by the scripts and, for Phase C, may contain model outputs for individual rows of a dataset that cannot be redistributed.

## Files common to every run

| File | Written by | Contents |
|---|---|---|
| `config.yaml` | `experiments/common.prepare_run` | Frozen copy of the configuration after command line overrides |
| `environment.json` | `experiments/common.prepare_run` | Python version, platform, versions of numpy, pandas, scipy, scikit-learn, shap, pymoo, statsmodels and pfca, start time, working directory and the full command line |
| `metrics.csv` | `experiments/common.ResultWriter` | One row per experimental unit and method with the metrics of Table 1 of the guide that apply to the phase; nested values (tables, arrays) are stored as JSON strings |

`metrics.csv` is appended to after every unit, so an interrupted run is resumed by repeating the same command: units whose key columns are already present are skipped.

## Run folders of the study

| Run | Script | Unit of `metrics.csv` | Other files |
|---|---|---|---|
| `phase_a` | `run_phase_a.py` | family, sample size, dimension, correlation, seed, method | duplication change of the redundancy family is a column of `metrics.csv` |
| `phase_b` | `run_phase_b.py` | dataset, repetition, method | `runs.csv` (seeds, split scheme and sizes, replicate repetition, tuned reference parameters, test performance, explained indices), `summary.csv` |
| `phase_c` | `run_phase_c.py` | repetition, method | `summary_by_method.csv`, `reliability*.csv`, `pfca_explanation_repeatRR.json`, `membership_repeatRR.png`, `overlap_repeatRR.png`, `example_repeatRR_instanceII.txt` and `.png` |
| `phase_d` | `run_phase_d.py generate` | not applicable | `materials/` (figures and text shown to the experts, response sheets), `response_sheet.csv`, `private/` (format mapping, answer key, seeds, explanation) |
| `phase_d/analysis` | `run_phase_d.py analyze` | expert, prediction, format | descriptive statistics, Friedman and Nemenyi, Wilcoxon with Holm correction, ordinal and binomial mixed models, `summary.txt`; a simulated analysis is written to `analysis_simulated/` |
| `sensitivity` | `run_sensitivity.py` | cell, seed, factor, setting | `runs.csv`, `summary.csv` |
| `example_front` | `example_front.py` | not applicable | `configurations.csv` (every evaluated configuration with objectives, front and knee flags), `front_points.csv`, `explanations.txt`, `explanation_knee.json`, `seeds.json` |
| `make_figures` | `make_figures.py` | not applicable | `items.csv`, the list of produced and skipped items of Table 2 |

The smoke configurations write to `phase_a_smoke`, `phase_b_smoke`, `phase_c_smoke`, `phase_d_smoke` and `sensitivity_smoke`; their numbers only show that the pipeline runs and carry no evidence.

## Seeds

Every run records the seeds it used: the base seed in `config.yaml`, the derived seeds in `runs.csv` (Phases B and C, sensitivity), in `private/seeds.json` (Phase D) and, for Phase A, the seed of the cell, from which the seeds of the resamples and model refits are derived deterministically (`AttributionEngine.seeds_`). Resample 0 of every pool is the original training data.

## From results to figures and tables

`experiments/make_figures.py --results results --out .` reads the metric files of the run folders and writes the figures to `figures/` and the tables to `tables/`; the run names are given by `--phase-a`, `--phase-b`, `--phase-c` and `--sensitivity`. Figures and tables are never edited by hand (Section 8.2 of the guide).

## Before sharing a run folder

Phase C and Phase D folders contain per instance model outputs and attributions of a questionnaire dataset. Check the data sharing agreement described in `data/README.md` before publishing any file from them.
