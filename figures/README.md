# Figures directory

The figures of the paper are written here by `experiments/make_figures.py` from the saved metric files under `results/`, and by `experiments/example_front.py` for the example Pareto front. Everything in this directory except this file is ignored by git, because every figure is rebuilt by the scripts and never edited by hand (Section 8.2 of the study design guide).

```
python3 experiments/example_front.py --out figures
python3 experiments/make_figures.py --results results --out .
```

`make_figures.py` writes into `<out>/figures`; the run names it reads are given by `--phase-a`, `--phase-b`, `--phase-c` and `--sensitivity` (defaults `phase_a`, `phase_b`, `phase_c` and `sensitivity`). An item whose inputs are missing is skipped with a printed note and recorded in `results/make_figures/items.csv`.

## Files (items of Table 2 of the guide)

Every figure of `make_figures.py` is saved as PNG and PDF.

| File | Item | Content | Source |
|---|---|---|---|
| `pipeline_schematic` | 2 | Schematic of the pipeline from the model to the fuzzy linguistic explanation | drawn from the method definition, no data |
| `phase_a_recovery_stability` | 3 | Attribution recovery error and rank stability against within block correlation, by method, faceted by family, with 95 percent confidence bands over seeds | `results/phase_a/metrics.csv` |
| `phase_a_duplication` | 4 | Duplication experiment of the redundancy family: credit splitting of feature level SHAP and invariance of PFCA | `results/phase_a/metrics.csv` |
| `reliability_sign_confidence` | 6 | Reliability diagram of the stated sign confidence against the observed sign agreement | `results/phase_a`, `results/phase_b` |
| `interval_coverage` | 6 | Support and core coverage against the nominal levels, with the five percentage point tolerance band of Section 7.6 | `results/phase_a`, `results/phase_b` |
| `example_front` | 7 | Example Pareto front with the knee point and the linguistic explanation of one instance at the sparsest, the knee and the most faithful front member | `experiments/example_front.py` (checked, not redrawn, by `make_figures.py`) |
| `phase_c_example_linguistic_bars.png`, `phase_c_subscale_overlap.png` | 10 | Fuzzy linguistic explanation of one instance of the psychological dataset and the overlap of the selected concepts with the subscales | copied from `results/phase_c/` |

Per run figures (membership heat maps, subscale overlaps, example explanations of Phase C, the expert study materials and rating plots of Phase D) are written by the phase scripts inside their run folders under `results/`.

## Conventions

Figures are produced with matplotlib on the Agg backend through the helpers of `pfca.plotting`: a fixed categorical palette with method colors assigned in a fixed order, serif fonts, recessive axes and grid, and direct labels where useful. File names are lower case with underscores and carry the phase and the content.
