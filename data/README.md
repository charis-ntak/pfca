# Data directory

This directory holds the input files of Phase C of the study (Section 6.3 of the study design guide, `docs/_source_guide.txt`): a psychological questionnaire with known subscales in which the items of a subscale are strongly correlated and a clinical or academic outcome is predicted. The files are read by `pfca.evaluation.datasets.load_questionnaire` and analysed by `experiments/run_phase_c.py`.

## The real dataset is not distributed

The questionnaire dataset analysed in the paper contains individual level responses that were collected under an ethics approval which does not allow public redistribution. The dataset is therefore not part of this repository, and no row of it appears in any file of the repository, in the test suite or in the result files. Place the two files of the real dataset under `data/raw/`, which is ignored by git, and point the keys `csv_path` and `spec_path` of `experiments/configs/phase_c.yaml` (or the options `--csv` and `--spec` of the runner) to them. The results, figures and per instance explanation files that the runner writes contain model outputs and attributions for individual rows; check the data sharing agreement before publishing them.

## Simulated demonstration files

`experiments/make_demo_questionnaire.py` writes `data/demo_questionnaire.csv` and `data/demo_questionnaire_spec.json`. These files are simulated. The rows are not real participants, and the constructs, items, covariates and outcome are artificial. They exist so that the Phase C pipeline can be run offline and so that the input format is documented by a working example. The generator draws latent constructs, measures every construct by Likert items with values 1 to 5 and a construct specific loading, adds age and sex as covariates and computes the outcome as a nonlinear function of the constructs plus noise. Every generator setting and the seed are stored under the key `generator` of the specification file, and the key `simulated` is set to true. The smoke configuration `experiments/configs/phase_c_smoke.yaml` reads these files.

```
python experiments/make_demo_questionnaire.py --n 300 --seed 0
python experiments/make_demo_questionnaire.py --n 1000 --seed 1 --classification
python experiments/run_phase_c.py --config experiments/configs/phase_c_smoke.yaml --results results --name phase_c_smoke
```

## Expected CSV format

The CSV file has one row per participant and a header row with the column names. The loader keeps only the columns named in the specification, so additional columns such as an identifier are allowed and ignored.

1. One column per questionnaire item. Item responses are numeric codes, for example 1 to 5 for a five point Likert scale. Reverse keyed items should be recoded before saving, so that all items of a subscale point in the same direction; this is recommended for the readability of the explanations, not required by the software.
2. Zero or more covariate columns. Covariates must be numeric. Binary covariates are coded 0 and 1 (for example sex), and categorical covariates with more than two levels must be coded numerically or expanded into indicator columns before saving.
3. One outcome column. For a regression task the outcome is numeric. For a classification task the column holds two labels, numeric or text, and the label of the positive class is named in the specification.
4. Missing values are empty cells. The loader drops every row that has a missing value in any item, covariate or outcome column (complete case analysis), so imputation, if wanted, has to be done before saving.
5. Column names should consist of letters, digits and underscores, because they appear in file names, figure labels and JSON keys of the outputs.

Example of the first rows of a file with two subscales of three items, two covariates and a continuous outcome:

```
participant,anx_1,anx_2,anx_3,rum_1,rum_2,rum_3,age,sex,outcome
p0001,4,5,4,2,2,3,34,1,61.2
p0002,1,2,1,3,3,4,51,0,44.7
```

## Expected JSON specification

The specification is a JSON object with the following keys.

| Key | Required | Meaning |
|---|---|---|
| `name` | no | Name of the dataset used in the result files; the file stem is used when absent |
| `outcome` | yes | Name of the outcome column |
| `task` | no | `regression` (default) or `classification` |
| `subscales` | yes | Object mapping every subscale name to the list of its item columns |
| `covariates` | no | List of covariate columns; they form one additional a priori group named `covariates` |
| `positive_class` | for classification | Value of the outcome column that defines the positive class; the outcome is coded 1 for this value and 0 otherwise |

Any other key is ignored by the loader and can be used for documentation, for example `description`, `simulated`, `likert_range` or `generator`.

```json
{
  "name": "questionnaire",
  "outcome": "outcome",
  "task": "regression",
  "subscales": {
    "anxiety": ["anx_1", "anx_2", "anx_3"],
    "rumination": ["rum_1", "rum_2", "rum_3"]
  },
  "covariates": ["age", "sex"]
}
```

## How the specification is used in Phase C

The feature matrix consists of the item columns in the order of the subscales followed by the covariate columns. The subscales, together with the covariate group, define the a priori partition of the features. `experiments/run_phase_c.py` uses this partition in two ways. It enters PFCA as a candidate partition next to the data driven fuzzy c means partitions (`apriori_groups` with `partition_source` set to `both`), it is the only partition of the `pfca_apriori` ablation, and it provides the crisp groups of grouped SHAP. It also serves as the external criterion: every estimated partition is compared with it by the adjusted Rand index and the fuzzy partition coefficient, and the per construct reliability table reports the membership overlap of every concept with every subscale.

## Ethics approval statement

Placeholder, to be completed before submission (Section 8.2 of the study design guide).

The study was reviewed and approved by [name of the ethics committee or institutional review board] of [institution] under protocol number [number] on [date]. All participants gave [written or electronic] informed consent before completing the questionnaire. Data were [pseudonymised or anonymised] before analysis, and no direct identifier was available to the analysts.

## Data availability statement

Placeholder, to be completed before submission (Section 8.2 of the study design guide).

The questionnaire data analysed in Phase C are not publicly available because [reason, for example the consent form did not include public sharing of individual level data]. Anonymised data are available from [the corresponding author or the data custodian] on reasonable request and subject to [a data sharing agreement or approval by the ethics committee]. The simulated demonstration data, the code that generates them and all analysis code are available in this repository. The public benchmark datasets of Phase B are downloaded by identifier at run time from [OpenML or the UCI repository].

## Checklist before running Phase C on the real data

1. The CSV and JSON files are under `data/raw/` and the paths in `experiments/configs/phase_c.yaml` point to them.
2. Every item column of every subscale and every covariate column exists in the CSV file and is numeric.
3. Reverse keyed items are recoded and the two ethics and data availability statements above are filled in.
4. The number of complete rows after dropping missing values is large enough for repeated 60/20/20 splits with 200 explained test instances, or `n_explain` in the configuration is reduced.
5. The run directory under `results/` is not committed and is checked against the data sharing agreement before any file from it is shared.
