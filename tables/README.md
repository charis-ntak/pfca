# Tables directory

The tables of the paper are written here by `experiments/make_figures.py` from the saved metric files under `results/`. Everything in this directory except this file is rebuilt by the script and never edited by hand (Section 8.2 of the study design guide); generated tables are not meant to be versioned.

```
python3 experiments/make_figures.py --results results --out .
```

`make_figures.py` writes into `<out>/tables`; the run names it reads are given by `--phase-a`, `--phase-b`, `--phase-c` and `--sensitivity` (defaults `phase_a`, `phase_b`, `phase_c` and `sensitivity`). An item whose inputs are missing is skipped with a printed note and recorded in `results/make_figures/items.csv`.

## Files (items of Table 2 of the guide)

| File | Item | Content | Source |
|---|---|---|---|
| `properties.csv`, `properties.md` | 1 | Properties of Section 5 satisfied by PFCA, grouped SHAP, bootstrapped SHAP and feature level SHAP | the propositions and the tests of `tests/test_properties.py` |
| `phase_a_recovery_stability.csv` | 3 | Mean and confidence interval of the recovery error and the rank stability per family, correlation and method | `results/phase_a/metrics.csv` |
| `phase_a_duplication.csv` | 4 | Change of the attribution when a duplicate feature is added, per method | `results/phase_a/metrics.csv` |
| `phase_b_metrics.csv`, `phase_b_friedman.csv`, `phase_b_friedman_pairwise.csv`, `phase_b_wilcoxon_holm.csv`, `phase_b_table.md` | 5 | Faithfulness and stability per dataset and method, Friedman average ranks with the Nemenyi critical difference, pairwise decisions, and Wilcoxon signed rank tests with Holm correction against feature level SHAP | `results/phase_b/metrics.csv` |
| `reliability_bins.csv`, `interval_coverage.csv` | 6 | Binned reliability of the sign confidence and the support and core coverage against the nominal levels | `results/phase_a`, `results/phase_b` |
| `runtime_memory.csv`, `runtime_memory.md` | 8 | Wall clock time and peak memory against the dimension d (Phase A) and against the number of resamples B and model classes M (sensitivity) | `results/phase_a`, `results/sensitivity` |
| `sensitivity_summary.csv`, `sensitivity_summary.md` | 9 | Effect of every factor of Section 7.5 on the metrics and on the chosen configuration, including the label set factor | `results/sensitivity/metrics.csv` |
| `phase_a_mixed_models.csv`, `phase_a_mixed_models.md` | 11 | Linear mixed models of the Phase A metrics per family: method, sample size, dimension and correlation as fixed effects, random intercept per seed (Section 7.4) | `results/phase_a/metrics.csv` |
| `success_criteria.csv`, `success_criteria.md` | 12 | The pre registered success criteria of Section 7.6 evaluated against feature level SHAP, with a verdict per criterion and overall | `results/phase_a`, `results/phase_b` |
| `pareto_front_diagnostics.csv`, `pareto_front_diagnostics.md` | 13 | Size of the Pareto front, fraction of runs in which it collapses to one explanation and correlation between the objectives (Section 10) | `results/phase_a`, `results/sensitivity` |

Tables that support the text of the phases (Phase C concept recovery and per construct reliability, Phase D descriptive statistics, Friedman and Nemenyi tests, Wilcoxon tests and model estimates) are written by the phase scripts inside their run folders under `results/`.

## Format

Tables are written as CSV files with one row per unit and, where a manuscript ready version is useful, also as Markdown. File names are lower case with underscores and carry the phase and the content.
