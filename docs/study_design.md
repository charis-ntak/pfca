# Pareto optimal fuzzy concept attribution: study design, methodological steps and evaluation protocol

Charis Ntakolia

Department of Aeronautical Studies, Sector of Materials Engineering, Machining Technology and Production Management, Hellenic Air Force Academy, Dekeleia Base, 13672, Acharnes, Attica, Greece

cntakolia@hafa.gr

Implementation guide, version 1, 30 September 2026

## 1. Purpose and scope

This document specifies a novel post hoc explainability method, referred to as Pareto optimal fuzzy concept attribution (PFCA), together with the study design and evaluation protocol required to establish it as a publishable alternative to SHAP. The method combines three ideas that have been treated separately in the literature: attribution to groups of correlated features rather than to individual features, the representation of attribution uncertainty as fuzzy membership, and the selection of explanations as a multiobjective trade off between fidelity, simplicity and stability. The guide is written so that each step can be implemented and checked independently. Sections 3 and 4 define the method and the algorithm, Section 5 lists the properties that must be proved, Sections 6 and 7 describe the study design and the evaluation protocol, and Sections 8 to 11 cover implementation, reporting, risks and timeline.

## 2. Problem statement and gaps addressed

Shapley additive explanations assign to each input feature a single real number that represents its contribution to a prediction. Three limitations motivate the proposed method. First, when features are correlated, credit is split between them in a manner that depends on the estimation scheme rather than on the data generating process, so that the resulting ranking is unstable and, for practitioners, misleading. Second, the attribution is a point estimate that carries no information about its reliability, although it varies with the training sample, the model class, the background distribution and random seeds. Third, the attribution is fixed by the additive axioms and cannot be tuned toward the properties that a given user needs, such as sparsity or robustness.

PFCA addresses the three limitations as follows. Features are aggregated into fuzzy concepts, so that credit is assigned to a concept and only then distributed to features according to their membership, which removes the arbitrary split between correlated features. Attributions are estimated repeatedly and represented as fuzzy numbers with linguistic labels, so that reliability is part of the explanation. Finally, the level of aggregation and the level of sparsity are selected from a Pareto front that trades off fidelity, simplicity and stability, so that the explanation reported is optimal in a stated sense and the trade off is transparent.

## 3. Method definition

### 3.1 Notation

Let f be a trained black box model mapping an input vector x in R to the power d to a prediction. Let X denote the training data with n rows and let x* denote an instance to be explained. A concept partition C is a set of K fuzzy concepts, where each feature j has a membership degree in concept k denoted by u subscript jk in the closed interval from 0 to 1, with memberships summing to one across concepts for each feature. A bootstrap resample is indexed by b from 1 to B, and a model class by m from 1 to M.

### 3.2 Concept formation

Concepts are formed by fuzzy c means clustering applied to the features rather than to the instances, using a distance based on the absolute correlation between features, or, when a latent structure is known, from a factor analytic loading matrix normalized row wise. The number of concepts K and the fuzzifier exponent are not fixed in advance but are treated as decision variables of the multiobjective problem in Section 3.5. When domain knowledge is available, as in psychological questionnaires with known subscales, the a priori grouping is included as one candidate partition and compared with the data driven partitions.

### 3.3 Concept level attribution

For each bootstrap resample b and model class m, the model is refitted and a feature level attribution vector is obtained by an exact or approximate Shapley computation with a background sample drawn from the resample. The concept level attribution is then defined by aggregation through the membership matrix, as in Equation 1.

$$
\phi_k^{(b,m)} = \sum_{j=1}^{d} u_{jk}\, \phi_j^{(b,m)} \qquad (1)
$$

Because memberships sum to one across concepts for every feature, the concept attributions sum to the same total as the feature attributions, so the efficiency property of SHAP is preserved at the concept level. This is the first property to be verified formally in Section 5.

### 3.4 Fuzzification of attributions

The set of values obtained across resamples and model classes for concept k forms an empirical distribution. The distribution is summarized by a trapezoidal fuzzy number whose support is given by the 5th and 95th percentiles and whose core is given by the 25th and 75th percentiles, as in Equation 2. A triangular alternative uses the median as the single core value.

$$
\Phi_k = \left( q_{0.05},\; q_{0.25},\; q_{0.75},\; q_{0.95} \right) \qquad (2)
$$

Two derived quantities are reported with each fuzzy attribution. The sign confidence is the proportion of resamples in which the attribution has the same sign as the median. The disagreement index is the ratio of the between model class variance to the total variance, and when it exceeds a stated threshold the attribution is upgraded to an interval type 2 fuzzy set whose footprint of uncertainty is the union of the type 1 fuzzy numbers obtained for each model class. Linguistic labels are obtained by evaluating the membership of the fuzzy attribution in five fixed fuzzy sets over the standardized attribution axis, namely strongly negative, weakly negative, negligible, weakly positive and strongly positive, using the fuzzy compatibility measure in Equation 3, where A is the linguistic set and the operator denotes the minimum.

$$
\operatorname{comp}(\Phi_k, A) = \sup_{z} \min\left( \mu_{\Phi_k}(z),\; \mu_A(z) \right) \qquad (3)
$$

Global importance is defined as the fuzzy number obtained from the mean absolute concept attribution across instances, and the fuzzy ranking between two concepts is expressed as the possibility degree that one fuzzy number exceeds the other, computed with the standard possibility measure for trapezoidal numbers.

### 3.5 Multiobjective selection of the explanation

An explanation configuration is a vector of decision variables consisting of the number of concepts K, the fuzzifier exponent, the sparsity level given by the number of concepts retained after alpha cut, and the alpha level of the cut. For every candidate configuration three objectives are evaluated, as in Equation 4. Fidelity loss is the error of a surrogate that reproduces the model output from the retained concept attributions, evaluated on held out data. Complexity is the number of retained concepts weighted by the entropy of their memberships, so that crisp concepts are preferred. Instability is one minus the average rank correlation of concept importance across bootstrap resamples.

$$
\min_{\theta} \left( f_1(\theta),\; f_2(\theta),\; f_3(\theta) \right) = \left( \text{fidelity loss},\; \text{complexity},\; \text{instability} \right) \qquad (4)
$$

The problem is solved with NSGA II when the decision space is large, or by exhaustive evaluation on a grid when K and the alpha level take a small number of values, which is expected in most applications. The output is a Pareto front of explanation configurations. A default representative solution is chosen by the knee point criterion, and the full front is retained for reporting so that a user may select a sparser or a more faithful explanation. The membership of a concept in the final explanation is then defined as the proportion of Pareto optimal configurations in which the concept is retained, which provides a second, selection based, layer of fuzziness that complements the estimation based layer of Section 3.4.

### 3.6 Outputs of the method

- A fuzzy concept partition with the membership matrix of features in concepts.
- For each instance, a fuzzy attribution per concept with sign confidence, disagreement index and linguistic labels.
- A global fuzzy importance ranking with pairwise possibility degrees.
- A Pareto front of explanation configurations and a knee point default.
- For compatibility with existing practice, a defuzzified attribution vector at the feature level obtained by redistributing the concept centroid values through the membership matrix, which reduces to ordinary SHAP when every concept contains one feature with full membership.

## 4. Algorithm

The steps below are listed in execution order. Steps 2 to 4 are the computationally expensive part and are parallelized over resamples and model classes.

1. Fit the reference model f on the training data and fix the set of model classes M that will be used to quantify model dependence, for example gradient boosting, random forest and a multilayer perceptron.
2. Draw B bootstrap resamples of the training data. For each resample and each model class, refit the model and compute feature level Shapley values for the instances to be explained, using a background sample of fixed size drawn from the resample.
3. Compute the feature correlation matrix on the full training data and run fuzzy c means on the features for every candidate value of K and of the fuzzifier exponent, producing a set of candidate membership matrices.
4. For every candidate membership matrix, aggregate the feature level attributions into concept level attributions by Equation 1, for every resample and model class.
5. Summarize the concept attributions into trapezoidal fuzzy numbers by Equation 2 and compute sign confidence, disagreement index and linguistic memberships by Equation 3.
6. For every candidate configuration, apply the alpha cut, fit the fidelity surrogate on the retained concepts, and evaluate the three objectives of Equation 4 on held out data.
7. Compute the Pareto front of configurations, identify the knee point, and compute the selection based membership of each concept.
8. Produce the outputs of Section 3.6 and the defuzzified feature level vector for comparison with baselines.

## 5. Theoretical properties to establish

A method paper in explainability is expected to state which axioms hold, which are relaxed, and why. The following properties are to be proved or demonstrated with counterexamples.

1. Efficiency at the concept level: the sum of concept attributions equals the difference between the prediction and the expected prediction, for every membership matrix whose rows sum to one.
2. Reduction to SHAP: when K equals d and the membership matrix is the identity, and when B and M equal one, the method returns the SHAP values exactly.
3. Symmetry at the concept level: two concepts with identical aggregated contributions in every coalition receive identical fuzzy attributions.
4. Invariance to feature duplication: adding an exact copy of a feature does not change the concept attribution of the concept that absorbs the copy, whereas it halves the feature level SHAP value. This is the central argument for concept attribution and should be given as a proposition with proof.
5. Consistency of the fuzzy summary: as B tends to infinity the trapezoidal fuzzy number converges to the corresponding quantiles of the sampling distribution of the concept attribution.
6. Monotonicity of the Pareto front: increasing the alpha level cannot decrease complexity, and the selection based membership is well defined and lies in the unit interval.

## 6. Study design

The study is organized in four phases. Phase A establishes the method on synthetic problems with known ground truth. Phase B evaluates it on public benchmark datasets against the baselines. Phase C applies it to a real psychological dataset in which correlated items and known subscales make the concept level attribution substantively meaningful. Phase D collects a small expert judgment study on the interpretability of the linguistic outputs. Phases A and B are required for a methods venue, Phase C is required for an applied venue, and Phase D is optional but strengthens the interpretability claim.

### 6.1 Phase A, synthetic ground truth

Data generating processes are constructed so that the true concept structure and the true attributions are known. Three families are used. In the additive family the response is a sum of functions of latent factors, each factor being measured by a block of correlated observed features, so that the true concept attribution is the contribution of each factor. In the interaction family a product term between two factors is added. In the redundancy family exact and near duplicates of features are introduced, which produces the credit splitting failure of feature level attribution. Sample sizes of 200, 500 and 2000, dimensions of 10, 30 and 100 features, and within block correlations of 0.3, 0.6 and 0.9 are crossed, and each cell is replicated 50 times with different seeds.

### 6.2 Phase B, public benchmarks

Tabular datasets with correlated features and moderate dimension are selected, for example datasets from the UCI repository and OpenML with between 15 and 100 features, covering regression and binary classification. At least six datasets are used so that paired statistical tests across datasets are meaningful.

### 6.3 Phase C, psychological application

A questionnaire dataset with known subscales is used, in which items within a subscale are strongly correlated and a clinical or academic outcome is predicted. The a priori subscale structure serves both as a candidate partition and as an external criterion for the data driven concepts. The substantive question is whether the concept level fuzzy explanation identifies the same constructs as the theory, and whether it reveals reliability differences between constructs that feature level SHAP conceals.

### 6.4 Phase D, expert judgment

A small number of domain experts are shown explanations of the same predictions in three formats, namely a SHAP bar plot, a plain concept level bar plot and the fuzzy linguistic explanation, and are asked to rate confidence calibration, ease of understanding and actionability on Likert scales, and to answer factual questions whose correct answers depend on the reliability information. The design is within subject with counterbalanced order.

## 7. Evaluation protocol

### 7.1 Baselines

The following baselines are implemented with their reference software and default settings, and are tuned only where the reference paper prescribes tuning.

- KernelSHAP and TreeSHAP at the feature level.
- Grouped or partition SHAP with crisp a priori groups, which isolates the effect of fuzziness.
- Bootstrapped SHAP with confidence intervals, which isolates the effect of the concept level and the Pareto selection.
- LIME and integrated gradients where applicable, as widely used references.
- Ablations of the proposed method: crisp concepts instead of fuzzy, a single model class, a single fixed configuration instead of the Pareto front, and the a priori partition instead of the data driven one.

### 7.2 Metrics

Metrics are grouped by the claim they support. Table 1 lists them with their definitions and the phases in which they apply.

Table 1. Evaluation metrics, definitions and applicable phases

| Claim | Metric | Definition | Phases |
|---|---|---|---|
| Correctness | Attribution recovery error | Mean absolute error between estimated and true concept attributions, after matching concepts to factors by maximum membership overlap | A |
| Correctness | Concept recovery | Adjusted Rand index between the hardened concept partition and the true factor structure, plus fuzzy partition coefficient | A, C |
| Correctness | Rank recovery | Spearman correlation between estimated and true importance ranking | A |
| Faithfulness | Deletion and insertion curves | Area under the curve of model output when concepts are removed or inserted in order of importance, using conditional imputation | A, B, C |
| Faithfulness | Surrogate fidelity | R squared or log loss of the surrogate fitted on retained concepts, on held out data | A, B, C |
| Stability | Rank stability | Mean pairwise Kendall tau of importance rankings across bootstrap resamples | A, B, C |
| Stability | Perturbation stability | Change in attribution under small input perturbations relative to change in output | A, B |
| Calibration | Interval coverage | Proportion of held out replications in which the true or replicated attribution falls inside the fuzzy support and inside the core | A, B |
| Calibration | Sign confidence calibration | Reliability diagram of stated sign confidence against observed sign agreement on a fresh sample | A, B |
| Robustness | Duplication invariance | Change in attribution when a duplicate feature is added | A |
| Cost | Runtime and memory | Wall clock time and peak memory as a function of d, B and M | A, B |
| Interpretability | Expert ratings and accuracy | Likert ratings and proportion of correct factual answers | D |

### 7.3 Experimental procedure

1. For each dataset or synthetic cell, split into training, tuning and test sets with proportions 60, 20 and 20, or use repeated cross validation when the sample is small.
2. Fit the reference model on the training set with hyperparameters selected on the tuning set, and fix the model for all methods so that differences are due to the explanation method only.
3. Compute explanations for a fixed set of 200 test instances with every method, using the same background sample and the same random seeds.
4. Compute all metrics of Table 1 on the test set, and store raw values per instance and per replication.
5. Repeat over the 50 seeds of Phase A or over 10 repetitions of the split in Phases B and C.

### 7.4 Statistical analysis

Within Phase A, differences between methods are analyzed with linear mixed models in which method, sample size, dimension and correlation are fixed effects and the replication seed is a random effect. Across the datasets of Phase B, paired comparisons use the Wilcoxon signed rank test with Holm correction, and the Friedman test with the Nemenyi post hoc procedure when more than two methods are compared, following the standard practice for comparing algorithms across datasets. Effect sizes are reported with confidence intervals. For Phase D, ratings are analyzed with ordinal mixed models and factual accuracy with a generalized linear mixed model.

### 7.5 Sensitivity analyses

- Number of resamples B in the set 20, 50, 100, 200, to determine the smallest B at which fuzzy summaries stabilize.
- Number of model classes M, from one to four.
- Choice of fuzzy number shape, triangular against trapezoidal, and of the percentile levels defining support and core.
- Choice of the fuzzy c means distance, correlation based against loading based.
- Choice of the Pareto solver, exhaustive grid against NSGA II, to confirm that the front is recovered.
- Choice of the knee point criterion against alternative representative selections.

### 7.6 Pre registration and success criteria

Before running Phase B and Phase C, the analysis plan and the success criteria are frozen. The method is considered successful if, relative to feature level SHAP, it achieves lower attribution recovery error and higher rank stability in the redundancy family of Phase A with at least a medium effect size, matches or exceeds faithfulness on the Phase B datasets, and produces calibrated intervals with coverage within five percentage points of the nominal level. Failure on any criterion is reported rather than concealed.

## 8. Implementation plan

### 8.1 Software and structure

The method is implemented in Python as a package with a scikit learn compatible interface. The core dependencies are shap for attribution, scikit fuzzy or a custom implementation for fuzzy c means, pymoo for NSGA II, and numpy and pandas for data handling. The package is organized in five modules, namely attribution, concepts, fuzzification, selection and evaluation, each with unit tests. The evaluation module contains the synthetic generators of Phase A so that the whole study is reproducible from a single script.

### 8.2 Reproducibility

- All random seeds are fixed and logged, and every metric is saved per instance, per replication and per configuration in a tabular file.
- Environment and package versions are frozen in a lock file, and a container image is provided.
- The public benchmark datasets are downloaded by identifier at run time, and the psychological dataset is described with its ethics approval and data availability statement.
- Figures and tables are generated by scripts from the saved metric files, never edited by hand.

### 8.3 Computational budget

The cost is dominated by B times M model refits and Shapley computations. With B equal to 100, M equal to 3, 200 explained instances and TreeSHAP, a dataset with 50 features runs in the order of minutes on a workstation, while KernelSHAP for non tree models increases the cost by one to two orders of magnitude and should be restricted to a subset of experiments. Parallelization over resamples is embarrassingly parallel and is implemented with joblib.

## 9. Reporting

The paper follows the structure of abstract, introduction, methodology, results, discussion and conclusions. Table 2 lists the minimum set of tables and figures, so that each claim of Section 7 is supported by at least one item.

Table 2. Planned tables and figures of the paper

| Item | Content | Supports |
|---|---|---|
| Table | Properties satisfied by PFCA, grouped SHAP, bootstrapped SHAP and feature level SHAP | Section 5 |
| Figure | Schematic of the pipeline from model to fuzzy linguistic explanation | Section 3 |
| Figure | Attribution recovery error and rank stability against within block correlation, by method, Phase A | Correctness, stability |
| Figure | Duplication experiment showing credit splitting in SHAP and invariance of PFCA | Robustness |
| Table | Faithfulness and stability on Phase B datasets with Friedman ranks | Faithfulness, stability |
| Figure | Reliability diagrams for interval coverage and sign confidence | Calibration |
| Figure | Example Pareto front with knee point and three explanations taken from the front | Selection |
| Figure | Fuzzy linguistic explanation of one instance in the psychological dataset, with subscale overlap | Phase C |
| Table | Runtime and memory against d, B and M | Cost |
| Table | Sensitivity analysis summary | Section 7.5 |

## 10. Risks and mitigations

- Risk: reviewers argue that the method is a combination of known components. Mitigation: prove the duplication invariance and the reduction to SHAP, and show empirically that no single component achieves the combined gains, using the ablations of Section 7.1.
- Risk: fuzzy c means on features produces unstable partitions. Mitigation: use consensus clustering over resamples and report the fuzzy partition coefficient, and include the a priori partition as a candidate.
- Risk: the Pareto front collapses to one solution. Mitigation: verify on Phase A that the objectives conflict, and if they do not, reduce the objective set and report this as a finding.
- Risk: computational cost prevents KernelSHAP baselines on large problems. Mitigation: restrict non tree models to the smaller datasets and state this limitation.
- Risk: the linguistic labels are perceived as arbitrary. Mitigation: define the label sets on a standardized axis, test alternative label sets in the sensitivity analysis, and validate them in Phase D.

## 11. Timeline

Table 3. Indicative timeline by phase

| Month | Activity | Deliverable |
|---|---|---|
| 1 | Formal definition, proofs of Section 5, package skeleton | Method section draft, unit tests |
| 2 | Synthetic generators and Phase A experiments | Phase A results and figures |
| 3 | Baselines and Phase B experiments, sensitivity analyses | Phase B tables |
| 4 | Phase C application and Phase D expert study | Applied results |
| 5 | Writing, internal review, code release | Manuscript and repository |
