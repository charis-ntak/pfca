# PFCA: Pareto optimal fuzzy concept attribution

Charis Ntakolia (GitHub: charis-ntak), Department of Aeronautical Studies, Hellenic Air Force Academy, cntakolia@hafa.gr

PFCA is a post hoc explainability method for tabular black box models. It is built on three ideas that are treated separately in the literature.

1. Fuzzy concepts from correlated features. Features are clustered by fuzzy c means on their correlation profiles, and the Shapley credit of a prediction is assigned to concepts before it is distributed to features according to their membership. Credit is no longer split arbitrarily between correlated features, and an exact duplicate of a feature leaves the attribution of its concept unchanged.
2. Fuzzy attributions with linguistic labels. The attribution is estimated over bootstrap resamples and several model classes, and the resulting distribution is summarized by a trapezoidal fuzzy number with a sign confidence, a disagreement index between model classes and linguistic labels such as weakly positive or negligible. When the model classes disagree the attribution is upgraded to an interval type 2 fuzzy set.
3. Pareto selection of the explanation. The number of concepts, the fuzzifier, the sparsity and the alpha level of the cut are chosen from a Pareto front that trades off fidelity loss, complexity and instability. A knee point is reported by default, the whole front is kept, and every concept receives a selection membership equal to how often the front retains it.

The package follows the study design guide written for the method, `docs/study_design.md`. The definitions chosen where the guide leaves freedom are recorded in `docs/implementation_notes.md`.

## Installation

The package needs Python 3.10 or later. From the repository root:

```
pip install -e ".[all]"
```

The runtime dependencies are numpy, scipy, pandas, scikit-learn, shap, pymoo, joblib and pyyaml (`requirements.txt`). The `all` extra adds statsmodels (statistical analysis), matplotlib (figures) and pytest (tests). The exact versions used for the study are frozen in `requirements-lock.txt`, and a container image is described by the `Dockerfile`.

## Quick start

The example fits PFCA on a synthetic regression problem with three latent factors, each measured by three correlated features, and explains 20 fresh instances. The pool is small so that the example runs in well under a minute; the study uses 100 resamples.

```python
from pfca import PFCAExplainer
from pfca.evaluation.synthetic import make_synthetic

# regression problem with three latent factors, each measured by three correlated features
problem = make_synthetic("additive", n_samples=300, n_features=9, rho=0.8, n_factors=3, random_state=0)
X_test, _ = problem.sample(20, random_state=1)

explainer = PFCAExplainer(
    n_resamples=5,               # bootstrap resamples B; resample 0 is the original data
    background_size=50,          # background sample of the Shapley computation
    n_concepts_grid=(2, 3, 4),   # candidate numbers of concepts K
    fuzzifier_grid=(2.0,),       # candidate fuzzifier exponents
    alpha_grid=(0.0, 0.5, 1.0),  # alpha levels of the cut
    random_state=0,
)
explanation = explainer.fit(problem.X, problem.y, feature_names=problem.feature_names).explain(X_test)

print(explanation.describe(0))                 # linguistic explanation of the first instance
print(explanation.global_frame().to_string())  # global fuzzy importance ranking
print(explanation.front().to_string())         # Pareto front of explanation configurations
```

`describe(0)` prints, for the instance, the selected configuration and one line per retained concept with its linguistic label and compatibility, the centroid, support and core of the fuzzy attribution, the sign confidence, the disagreement index and the selection membership. `global_frame()` is a table with one row per concept, ranked by the possibility degree of exceeding the other concepts. `front()` is the table of Pareto optimal configurations with the three objectives and the retained concepts; the knee point is flagged in the column `knee`.

The default model classes are gradient boosting, random forest and a multilayer perceptron; the first is the reference model. Pass `model_classes=[("gbm", GradientBoostingRegressor()), ...]` to change them, `apriori_groups` to add a known grouping as a candidate partition, `solver="nsga2"` for large decision spaces, and `n_jobs` to parallelize the pool over resamples and model classes. `PFCAExplainer` accepts data frames and classifiers, in which case the probability of the positive class is explained. The other outputs are available as attributes and methods of the explanation: `partition.U` and `partition.to_frame()` (membership matrix), `instance_frame(i)`, `fuzzy_attribution(i, k)`, `possibility`, `ranking`, `feature_attribution()`, `selection.table`, and `save(path)` for a JSON file. The functions of `pfca.plotting` draw the fuzzy attributions, the linguistic bars, the Pareto front and the membership heat map.

## Outputs of the method

PFCA returns the outputs listed in Section 3.6 of the guide.

1. A fuzzy concept partition with the membership matrix of features in concepts (`explanation.partition`).
2. For each instance, a fuzzy attribution per concept with sign confidence, disagreement index and linguistic labels (`quantiles`, `sign_confidence`, `disagreement`, `linguistic`, `instance_frame`, `describe`).
3. A global fuzzy importance ranking with pairwise possibility degrees (`global_quantiles`, `possibility`, `ranking`, `dominance`, `global_frame`).
4. A Pareto front of explanation configurations and a knee point default (`front()`, `selection.table`, `configuration`), together with the selection membership of every feature and concept.
5. A defuzzified attribution vector at the feature level obtained by redistributing the concept centroids through the membership matrix (`feature_attribution()`), which reduces to ordinary SHAP when every concept contains one feature with full membership.

## Equations

Concept level attribution of resample b and model class m (Equation 1):

$$
\phi_k^{(b,m)} = \sum_{j=1}^{d} u_{jk}\, \phi_j^{(b,m)}
$$

Trapezoidal fuzzy attribution from the empirical distribution over resamples and model classes (Equation 2):

$$
\Phi_k = \left( q_{0.05},\; q_{0.25},\; q_{0.75},\; q_{0.95} \right)
$$

Compatibility of a fuzzy attribution with a linguistic set A (Equation 3):

$$
\operatorname{comp}(\Phi_k, A) = \sup_{z} \min\left( \mu_{\Phi_k}(z),\; \mu_A(z) \right)
$$

Multiobjective selection of the explanation configuration theta (Equation 4):

$$
\min_{\theta} \left( f_1(\theta),\; f_2(\theta),\; f_3(\theta) \right) = \left( \text{fidelity loss},\; \text{complexity},\; \text{instability} \right)
$$

## Package layout

The package uses a src layout and is organized in the five modules named in Section 8.1 of the guide, plus the explainer that ties them together.

| Path | Contents |
|---|---|
| `src/pfca/attribution.py` | `AttributionEngine` (pool of B resamples by M model classes, refits and feature level Shapley values), `AttributionResult`, `shapley_values`, `aggregate_to_concepts` (Equation 1), `redistribute_to_features`, `default_model_classes` |
| `src/pfca/concepts.py` | Correlation profile embedding, `fuzzy_c_means`, loading based and a priori memberships, consensus clustering, `ConceptPartition`, `ConceptFormer` |
| `src/pfca/fuzzification.py` | `TrapezoidalFuzzyNumber`, `fuzzify` and `fuzzify_array` (Equation 2), sign confidence, disagreement index, `IntervalType2FuzzyNumber`, `LinguisticLabelSet` and `compatibility` (Equation 3), possibility degree and fuzzy ranking, standardization scale |
| `src/pfca/selection.py` | Alpha cut retention, the objectives of Equation 4 (`fidelity_loss`, `complexity`, `instability`), `pareto_mask`, `knee_point`, selection membership, `ParetoSelector` with grid and NSGA II solvers |
| `src/pfca/explainer.py` | `PFCAExplainer` (scikit-learn style `fit` and `explain`) and `PFCAExplanation` (outputs, tables, text, JSON) |
| `src/pfca/plotting.py` | Figures: fuzzy attributions, linguistic bars, SHAP bars, Pareto front, membership heat map, reliability diagram (matplotlib, Agg backend) |
| `src/pfca/utils.py` | Random state helpers, membership matrix checks, row entropy, concept fuzziness, partition coefficient |
| `src/pfca/evaluation/synthetic.py` | Synthetic generators of Phase A (additive, interaction and redundancy families) with closed form true attributions, and the factorial design |
| `src/pfca/evaluation/metrics.py` | The metrics of Table 1: recovery error, concept and rank recovery, deletion and insertion curves, surrogate fidelity, rank and perturbation stability, interval coverage, sign confidence calibration, duplication invariance |
| `src/pfca/evaluation/baselines.py` | Feature level SHAP, grouped Shapley values with crisp groups, bootstrapped SHAP, LIME (optional), integrated gradients, ablation builders |
| `src/pfca/evaluation/protocol.py` | The shared procedure of Section 7.3: every method computed from one attribution pool (PFCA and its ablations, feature level SHAP, bootstrapped SHAP, grouped SHAP, integrated gradients and LIME), the Pareto front diagnostics, and `evaluate` for the metrics |
| `src/pfca/evaluation/statistics.py` | Wilcoxon tests with Holm correction, Friedman and Nemenyi, effect sizes with bootstrap intervals, linear mixed, ordinal and binomial mixed models |
| `src/pfca/evaluation/datasets.py` | Benchmark loaders of Phase B (OpenML by name and version, built in scikit-learn datasets offline), questionnaire loader of Phase C, repeated 60/20/20 splits and repeated five fold cross validation for small samples |
| `src/pfca/evaluation/budget.py` | Wall clock time and peak memory measurement |
| `experiments/` | `common.py` (configuration, environment logging, resumable result files, reference model tuning), the phase scripts, `make_demo_questionnaire.py`, `example_front.py`, `make_figures.py` and the YAML configurations under `configs/` |
| `tests/` | Unit tests of every module, the numerical checks of the properties of Section 5, and end to end runs of the experiment scripts on tiny configurations |
| `docs/` | The study design guide, its Markdown conversion and the implementation notes |
| `data/`, `results/`, `figures/`, `tables/` | Inputs of Phase C and outputs of the scripts; contents other than the README files are not versioned |

## Reproducing the study

Every script has a module docstring with usage examples and an argparse interface (`--help`). Each run writes to `results/<name>/` a frozen copy of its configuration (`config.yaml`), the package versions and command line (`environment.json`) and its metric files, and every run can be resumed because finished cells are skipped. Commands are given from the repository root. The full configurations use `n_jobs: 4`; pass `--n-jobs` to change the number of workers.

### Phase A, synthetic ground truth

```
python3 experiments/run_phase_a.py --config experiments/configs/phase_a.yaml --results results
```

The design of Section 6.1 (three families, sample sizes 200, 500 and 2000, dimensions 10, 30 and 100, within block correlations 0.3, 0.6 and 0.9, 50 seeds) is crossed in `phase_a_grid`; every method of the configuration is computed from one attribution pool per cell, and the metrics of Table 1 are appended to `metrics.csv` per cell and method. Options: `--seeds` overrides the number of seeds, `--only-family` restricts the run to one family and `--max-cells` stops after a number of cells.

### Phase B, public benchmarks

```
python3 experiments/run_phase_b.py --config experiments/configs/phase_b.yaml --results results
```

The benchmark list of `pfca.evaluation.datasets.BENCHMARKS` is downloaded from OpenML by name and version at run time (network access required); datasets outside 15 to 100 features are skipped. Ten repetitions of the 60/20/20 split are run per dataset, the reference gradient boosting model is tuned on the tuning split, and `metrics.csv`, `runs.csv` and `summary.csv` are written. Datasets with fewer than `small_sample_rows` rows (500) use repeated five fold cross validation instead of the 60/20/20 splits (`split_scheme: auto`, Section 7.3). The LIME baseline needs the optional `lime` package (`pip install lime`) and is skipped with a message otherwise; integrated gradients need the `mlp` model class. Options: `--datasets breast_cancer,spambase` restricts the list and `--max-repeats` caps the repetitions. The built in `breast_cancer` dataset loads offline.

### Phase C, psychological questionnaire

```
python3 experiments/run_phase_c.py --config experiments/configs/phase_c.yaml --results results
```

The real questionnaire is not distributed; place its CSV file and JSON subscale specification under `data/raw/` in the format described in `data/README.md` and point `csv_path` and `spec_path` of the configuration (or `--csv` and `--spec`) to them. The a priori subscales enter as a candidate partition, as the partition of the `pfca_apriori` ablation and of grouped SHAP, and as the external criterion of concept recovery. Besides `metrics.csv` and `summary_by_method.csv` the run writes per construct reliability tables, the knee explanation as JSON, membership and subscale overlap heat maps, and the linguistic explanation and figures of example instances. A simulated questionnaire for testing is produced by `python3 experiments/make_demo_questionnaire.py --n 300 --seed 0`.

### Phase D, expert judgment

```
python3 experiments/run_phase_d.py generate --config experiments/configs/phase_d.yaml --results results --name phase_d
python3 experiments/run_phase_d.py analyze --results results --name phase_d --responses results/phase_d/responses.csv
```

`generate` fits PFCA on the questionnaire of Phase C (or on a synthetic problem with `--synthetic`), renders every presented prediction as a feature level SHAP bar plot, a plain concept bar plot and the fuzzy linguistic explanation behind neutral format codes, writes the counterbalanced response sheets (balanced Latin square over the three formats) and stores the answer key and the format mapping in a private folder. `analyze` reads the filled response sheets and writes descriptive statistics, Friedman and Nemenyi tests, Wilcoxon tests with Holm correction against the fuzzy format, ordinal models of the ratings and a binomial mixed model of factual accuracy. `analyze --simulate` fills the blank sheets with random responses to test the analysis path; its outputs are marked as simulated.

### Sensitivity analyses

```
python3 experiments/run_sensitivity.py --config experiments/configs/sensitivity.yaml --results results
```

One attribution pool per cell and seed is fitted at the largest number of resamples with every model class, and the factors of Section 7.5 are varied one at a time around the default setting: number of resamples (20, 50, 100, 200), number of model classes (1 to 4), fuzzy number shape and percentile levels, correlation against loading distance, grid against NSGA II solver (with the fraction of the grid front recovered and the hypervolumes), the knee criterion against alternative representative selections, and the default linguistic label set against the alternative label sets of the configuration (Section 10 of the guide). `--factors resamples,solver` restricts the factors and `--seeds` the number of seeds.

### Figures and tables

```
python3 experiments/example_front.py --out figures
python3 experiments/make_figures.py --results results --out .
```

`make_figures.py` builds the items of Table 2 of the guide from the saved metric files of the runs above and writes the figures (PNG and PDF) to `figures/` and the tables (CSV and Markdown) to `tables/` under `--out`; an item whose inputs are missing is skipped with a printed note, so the script can be run at any stage of the study, and the list of produced and skipped items is logged in `results/make_figures/items.csv`. The run names are given by `--phase-a`, `--phase-b`, `--phase-c` and `--sensitivity` (defaults `phase_a`, `phase_b`, `phase_c`, `sensitivity`), and `--stability-column` selects the stability metric of the Phase A figure. `example_front.py` fits PFCA on one synthetic problem and draws the example Pareto front with the knee point and three explanations taken from the front (item 7); `make_figures.py` checks that this figure exists. Beyond Table 2 the script writes the linear mixed models of the Phase A metrics (Section 7.4), the evaluation of the pre registered success criteria (Section 7.6) and the Pareto front diagnostics of Section 10 (items 11 to 13). Figures and tables are never edited by hand. The lists of files are given in `figures/README.md` and `tables/README.md`.

## Smoke configurations

Every phase has a small configuration that exercises the complete pipeline offline in a few minutes, with at most two workers. They are also collected in the `smoke` target of the `Makefile`.

| Phase | Command |
|---|---|
| A | `python3 experiments/run_phase_a.py --config experiments/configs/phase_a_smoke.yaml --results results --name phase_a_smoke --n-jobs 2` |
| B | `python3 experiments/run_phase_b.py --config experiments/configs/phase_b_smoke.yaml --results results --name phase_b_smoke` (built in breast cancer dataset, no download) |
| C | `python3 experiments/make_demo_questionnaire.py --n 300 --seed 0` followed by `python3 experiments/run_phase_c.py --config experiments/configs/phase_c_smoke.yaml --results results --name phase_c_smoke` |
| D | `python3 experiments/run_phase_d.py generate --synthetic --n-instances 3 --n-experts 6 --results results --name phase_d_smoke --n-resamples 3` followed by `python3 experiments/run_phase_d.py analyze --results results --name phase_d_smoke --simulate` |
| Sensitivity | `python3 experiments/run_sensitivity.py --config experiments/configs/sensitivity_smoke.yaml --results results --name sensitivity_smoke` |
| Figures | `python3 experiments/example_front.py --n-resamples 4 --n-explain 30 --out figures` followed by `python3 experiments/make_figures.py --results results --phase-a phase_a_smoke --phase-b phase_b_smoke --phase-c phase_c_smoke --sensitivity sensitivity_smoke --out .` |

The smoke runs write to `results/<phase>_smoke/` and their numbers carry no evidence; they only check that every script runs end to end.

## Reproducibility

The practices of Section 8.2 of the guide are implemented as follows.

- Seeds. Every script takes a seed from its configuration or command line, derives the seeds of the resamples, the model refits, the background samples, the splits and the explained instances from it, and logs them: `AttributionResult.seeds` holds one seed per pool member, `environment.json` records the seed entries of the configuration, and the metric and run files (`metrics.csv`, `runs.csv` where a script writes one, `private/seeds.json` of Phase D) record the derived seeds actually used. Resample 0 of every pool is the original training data.
- Per instance, per replication and per configuration storage. Metrics are appended to `metrics.csv` with one row per cell or split repetition and method, the configuration tables of PFCA contain every evaluated configuration with its three objectives, and the per instance explanations of Phases C and D are saved as JSON.
- Raw values. Besides the per cell means in `metrics.csv`, the Phase A, B and C scripts write `per_instance.csv` (long format: cell keys, method, instance, metric, value) with the recovery error, the deletion and insertion areas, the model output, the coverage indicators and the sign agreement of every explained instance. Wall times are measured with memory tracing off; set `trace_memory: true` in a configuration for a dedicated cost run that also records peak memory.
- Environment. `experiments/common.prepare_run` writes `environment.json` with the Python version, the platform, the versions of numpy, pandas, scipy, scikit-learn, shap, pymoo, statsmodels and pfca, the start time, the working directory and the full command line, next to a frozen copy of the configuration.
- Lock file and container. `requirements-lock.txt` freezes the versions of the scientific stack used for the study (`make lock` regenerates it from the current environment), and the `Dockerfile` builds an image from `python:3.11-slim` that installs the package with all extras and runs the test suite as a build check. The GitHub Actions workflow in `.github/workflows/tests.yml` runs the tests on Python 3.10, 3.11 and 3.12.
- Data. Benchmark datasets are downloaded by identifier at run time; the questionnaire of Phase C is described with its ethics approval and data availability statements in `data/README.md`.
- Figures and tables are produced by `experiments/make_figures.py` and `experiments/example_front.py` from the saved metric files and from seeded runs, and every produced or skipped item is logged in `results/make_figures/items.csv`.

## Pre registered success criteria

Section 7.6 of the guide freezes the analysis plan and the success criteria before Phase B and Phase C are run. Relative to feature level SHAP, the method is considered successful if it

1. achieves lower attribution recovery error and higher rank stability in the redundancy family of Phase A with at least a medium effect size,
2. matches or exceeds faithfulness on the Phase B datasets, and
3. produces calibrated intervals with coverage within five percentage points of the nominal level.

Failure on any criterion is reported rather than concealed: `make_figures.py` evaluates the three criteria from the saved metric files and writes `tables/success_criteria.csv` and `tables/success_criteria.md` with a verdict per criterion and overall.

## Testing

```
python3 -m pytest
```

The suite covers every module and contains numerical checks of the six properties of Section 5 of the guide (`tests/test_properties.py`, with supporting tests in `tests/test_selection.py` and `tests/test_fuzzification.py`); the mapping from property to test is given in `docs/implementation_notes.md`. The unit tests run offline in well under a minute. `tests/test_experiments.py` additionally runs every script under `experiments/` end to end on tiny configurations through its command line interface (Phases A to D, the sensitivity analysis, the example front and the figures), which takes about one more minute; these tests carry the marker `slow` and are skipped with `python3 -m pytest -m "not slow"`.

## Manuscript

The manuscript drafts are not versioned. The directory `paper/` is ignored by git, together with every `.docx` file, and is kept in the local working copy of the author. It holds the manuscript source, the verified bibliography, the scripts which turn the result files of the reduced runs (`experiments/configs/*_reduced.yaml`) into the tables and figures of the manuscript, and the built document. The reduced runs use two seeds, 20 resamples and two model classes, the breast cancer dataset for Phase B and the simulated questionnaire for Phase C; the full study configurations are the files without the `_reduced` suffix.

## Citation

If you use this software, please cite it with the metadata in `CITATION.cff`:

Ntakolia, C. (2026). pfca: Pareto optimal fuzzy concept attribution (Version 0.1.0) [Computer software]. Hellenic Air Force Academy. https://github.com/charis-ntak/pfca

## License

MIT License, see `LICENSE`.
