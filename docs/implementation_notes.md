# Implementation notes

Charis Ntakolia, Hellenic Air Force Academy

The study design guide (`docs/study_design.md`) defines the method and leaves a number of details to the implementation. This document records the precise definitions chosen in the code of the `pfca` package, so that every number in the result files can be traced to a stated rule. Section numbers and equation numbers refer to the guide. Module paths refer to `src/pfca/`.

## 1. Attribution pool (Section 3.3, Steps 1 and 2 of Section 4)

Module: `attribution.py`, class `AttributionEngine`.

### 1.1 Resamples and seeds

The pool has B resamples and M model classes. Resample 0 is always the original training data: `resample_indices_[0]` is `numpy.arange(n)`. Resamples 1 to B minus 1 are bootstrap draws of n row indices with replacement from a `numpy.random.Generator` seeded with `random_state`. With B equal to 1 the pool therefore contains only the fit on the original data, which is what the reduction to SHAP (Property 2) requires.

One integer seed per pool member is drawn from the same generator (`spawn_seeds`, B times M values reshaped to a B by M array and stored in `seeds_`). When a model class is refitted on resample b, every parameter named `random_state` or ending in `__random_state` (pipelines) is set to `seeds_[b, m]`. The Shapley computation of that member uses `seeds_[b, m] + 1` for the background draw and for the permutation explainer. The seeds are returned in `AttributionResult.seeds` for logging.

The reference model is the pool member with b equal to 0 and m equal to 0, that is the first model class fitted on the original data. `AttributionResult.reference_output` is its output on the explained instances, and it is the quantity that the fidelity surrogate reproduces.

### 1.2 Explained function

Regressors are explained through `predict`. Classifiers are explained through the predicted probability of the target class (`predict_proba`, column of `target_class`, default 1; the last column is used when the label is absent). The task is detected from the target when `task` is `'auto'`: string, object or boolean targets and numeric targets with at most two distinct values in {0, 1} are classification, everything else is regression.

### 1.3 Interventional background convention

Every explainer uses the interventional (marginal) expectation with respect to a background sample, never a conditional or tree path dependent expectation. The background sample of pool member (b, m) is drawn without replacement from the rows of resample b, with `background_size` rows (default 100); when `background_size` is `None` or not smaller than the resample, the whole resample is the background. The expected value returned with every member is the mean of the explained output over that background sample, so that for every member and every explained instance the feature level Shapley values sum to the output minus the expected value (checked in `tests/test_attribution.py::test_engine_shapes_and_efficiency`).

### 1.4 Which shap explainers are used

The `explainer` parameter resolves as follows when set to `'auto'` (`choose_explainer`).

| Condition | Explainer | Settings |
|---|---|---|
| Tree ensemble or tree of scikit-learn, XGBoost or LightGBM, not wrapped in a pipeline | `shap.TreeExplainer` | `data=background`, `feature_perturbation='interventional'`, `model_output='probability'` for classification, `check_additivity=False`; for multi output values the column of the target class is taken |
| Linear model of scikit-learn (`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `BayesianRidge`), not in a pipeline | `shap.LinearExplainer(model, background)` | interventional by construction |
| Any other model with at most 10 features | `shap.ExactExplainer` | masker `shap.maskers.Independent(background, max_samples=len(background))`, exact enumeration of coalitions |
| Any other model with more than 10 features | `shap.PermutationExplainer` | same masker, `seed`, `max_evals = max(2 d + 1, 500)` unless `permutation_max_evals` is given |

The values `'tree'`, `'linear'`, `'exact'`, `'permutation'` and `'kernel'` force one explainer. `'kernel'` uses `shap.KernelExplainer(g, background)` with `nsamples` from `kernel_nsamples`. The multilayer perceptron of the default model classes is a pipeline with a standard scaler, so it is explained with the exact explainer for d at most 10 and with the permutation explainer otherwise.

### 1.5 Default model classes

`default_model_classes` returns, in this order, gradient boosting (200 trees, depth 3), random forest (200 trees, minimum leaf 2) and a multilayer perceptron with layers (64, 32), early stopping and at most 800 iterations, preceded by a standard scaler. The first entry is the reference model class. The experiment scripts replace the first entry by the gradient boosting model tuned on the tuning set (`experiments/common.tune_gradient_boosting`, depth in {2, 3, 4} and 100 or 300 trees with learning rate 0.05).

### 1.6 Equation 1

`aggregate_to_concepts(values, U)` computes `values @ U` along the last axis, that is phi_k equals the sum over j of u_jk phi_j, for pools of any leading shape. Because the rows of U sum to one the total is preserved for every member of the pool (`tests/test_attribution.py::test_aggregate_preserves_total`).

## 2. Concept formation (Section 3.2, Step 3)

Module: `concepts.py`.

### 2.1 Correlation profile embedding

The feature by feature correlation matrix R is computed on the full training data with Pearson (`numpy.corrcoef`) or Spearman (`scipy.stats.spearmanr`) correlation. Correlations that are undefined because a feature is constant are set to zero and the diagonal is set to one. Feature j is embedded as row j of the matrix of absolute correlations, that is the vector of |R_jl| over all l, including the entry 1 for l equal to j. Fuzzy c means is run on these d rows with the squared Euclidean distance, so that two features are close when their patterns of absolute correlation with all features are similar. This is the `'correlation'` distance of the guide.

The `'loading'` distance fits `sklearn.decomposition.FactorAnalysis` with K components on the training data and takes the absolute loadings of every feature, normalized row wise; a feature whose loadings are all zero receives the uniform membership 1 / K. This candidate has no fuzzifier.

### 2.2 Fuzzy c means

`fuzzy_c_means(Z, K, m)` requires K smaller than the number of rows and a fuzzifier m larger than one. Memberships are initialized from a flat Dirichlet distribution; centers are the membership weighted means with weights u^m; squared distances are floored at 1e-12; the update is u_ik proportional to D_ik to the power minus 1 / (m minus 1), normalized over k. The iteration stops when the largest membership change is below 1e-7 or the relative change of the objective is below 1e-7, with at most 300 iterations. Five random initializations are run and the solution with the smallest objective is kept.

### 2.3 Ordering of concepts

The columns of U are ordered deterministically after clustering: every feature is assigned to its concept of maximum membership (hardening), and the concepts are sorted by the smallest index of the features assigned to them; a concept to which no feature is assigned is placed after the others, in its original order. With this rule the concept that contains feature 0 is always C1, and two runs with the same seed give identical column orders (`tests/test_concepts.py::test_fcm_deterministic_with_seed`).

### 2.4 Candidate partitions

`ConceptFormer` builds one candidate per pair (K, m) of `n_concepts_grid` and `fuzzifier_grid`, named `fcm_K{K}_m{m}`, for the K values with 1 at most K smaller than d; values outside that range are skipped. With the loading distance one candidate per K is built, named `loading_K{K}`. When `apriori_groups` is given a crisp candidate named `apriori` is added; features that belong to no group are collected in one additional concept. With `partition_source='apriori'` only the a priori candidate is used. If no grid value is valid the explainer falls back to K equal to d minus 1, and with a single feature to the identity partition. Every candidate uses the same seed derived from `random_state`. With `consensus=True` fuzzy c means is run on `n_consensus` (default 20) bootstrap resamples of the instances, the co-membership matrix U U^T is averaged over the resamples, and fuzzy c means is run once more on the rows of the averaged matrix. With `crisp_concepts=True` (ablation) every candidate is hardened to a 0/1 membership matrix.

### 2.5 Fuzziness of a concept

`utils.row_entropy` is the entropy of the membership distribution of a feature across concepts, minus the sum over k of u_jk log u_jk, divided by log K so that it lies in [0, 1] (zero when K equals 1). `utils.concept_fuzziness` is the membership weighted mean of the row entropies of a concept, F_k equals the sum over j of u_jk H_j divided by the sum over j of u_jk. A crisp concept has fuzziness zero and a concept of the uniform partition has fuzziness one. The fuzzy partition coefficient of Bezdek, the sum of squared memberships divided by d, is reported as `fpc`.

## 3. Fuzzification (Section 3.4, Step 5)

Module: `fuzzification.py`.

### 3.1 Five quantile storage of fuzzy numbers

For a partition with K concepts the pool of concept attributions has shape (B times M, n_explain, K). Every fuzzy attribution is stored as a vector of five quantiles of the pool over its first axis, at the percentile levels (support low, core low, 50, core high, support high), by default (5, 25, 50, 75, 95), computed with `numpy.percentile` and linear interpolation. The trapezoid of Equation 2 is (q_0, q_1, q_3, q_4) and the median q_2 is kept as the point summary of the attribution. With `fuzzy_shape='triangular'` the entries q_1 and q_3 are replaced by the median. `PartitionPool.quantiles` has shape (n_explain, K, 5). The alpha cut of a stored fuzzy number is [q_0 + alpha (q_1 minus q_0), q_4 minus alpha (q_4 minus q_3)] (`alpha_cut_array`), and its centroid is (d^2 + c^2 + c d minus a^2 minus b^2 minus a b) / (3 (d + c minus a minus b)) with (a, b, c, d) equal to (q_0, q_1, q_3, q_4), or the midpoint of the support when the denominator vanishes (`centroid_array`).

### 3.2 Global importance

The global importance of concept k for pool member p is the mean over the explained instances of |phi_k^(p)|. The global fuzzy importance is the five quantile summary of these B times M values, stored in `PartitionPool.global_quantiles` of shape (K, 5). The possibility degree that A is at least B is 1 when the core of A starts at or above the core of B (a_3 at least b_2), 0 when the support of A ends at or below the support of B (a_4 at most b_1), and (a_4 minus b_1) / ((a_4 minus a_3) + (b_2 minus b_1)) otherwise. The dominance of a concept is the minimum possibility of exceeding every other concept; concepts are ranked by dominance, with the centroid as tie breaker (`fuzzy_ranking`).

### 3.3 Sign confidence and disagreement index

The sign confidence of an attribution is the proportion of pool members whose value has the sign of the pool median; when the median is exactly zero it is the proportion of members that are exactly zero. The disagreement index is the variance over model classes of the class means (mean over resamples) divided by the variance of the whole B times M pool; it is zero when the total variance is at most 1e-18 and always zero when M equals 1, and it is clipped to [0, 1].

### 3.4 Type 2 upgrade rule

An attribution of instance i and concept k is upgraded to an interval type 2 fuzzy number when its disagreement index exceeds `disagreement_threshold` (default 0.5). The type 2 number has one trapezoidal member per model class, each obtained by fuzzifying the B values of that class with the same shape and percentile levels; its upper membership is the pointwise maximum and its lower membership the pointwise minimum of the members, so the footprint of uncertainty is the union of the type 1 numbers. Its compatibility with a linguistic set is the interval between the minimum and the maximum member compatibility, and the label reported in `describe` is the one with the largest upper compatibility. The five quantile summary of the whole pool is kept for the retention rule and the objectives, so that the selection problem does not depend on the threshold.

### 3.5 Standardization scale of the linguistic axis and the default label set

Linguistic labels are evaluated on the attribution divided by a scale. With `linguistic_scale='mean_abs'` (default) the scale is the mean absolute concept attribution of the reference pool member (b equal to 0, m equal to 0) over all explained instances and all concepts of the selected partition, so that one unit of the axis is a typical contribution of the reference model. With `'output_sd'` the scale is the standard deviation of the reference output over the explained instances, and a positive number is used as given. A scale smaller than 1e-12 is replaced by one.

The default label set is a Ruspini partition of the standardized axis with five trapezoids, written as (a, b, c, d):

| Label | Trapezoid |
|---|---|
| strongly negative | (minus infinity, minus infinity, minus 2, minus 1) |
| weakly negative | (minus 2, minus 1, minus 1, 0) |
| negligible | (minus 1, 0, 0, 1) |
| weakly positive | (0, 1, 1, 2) |
| strongly positive | (1, 2, plus infinity, plus infinity) |

Infinity is represented by 1e12. The memberships of the five labels sum to one at every point of the axis (`tests/test_fuzzification.py::test_linguistic_labels_ruspini`). Equation 3 is evaluated in closed form: the compatibility is one when the cores intersect and otherwise the height of the crossing of the two facing linear edges, (l_d minus r_a) / ((l_d minus l_c) + (r_b minus r_a)), where (l_c, l_d) is the descending edge of the left number and (r_a, r_b) the ascending edge of the right number, and zero when the supports do not overlap. The label of an attribution is the one with the largest compatibility, the first in the order above in case of a tie. Alternative label sets for the sensitivity analysis are built with `LinguisticLabelSet.with_breakpoints(centers, names)`.

## 4. Multiobjective selection (Section 3.5, Steps 6 and 7)

Module: `selection.py`.

### 4.1 Alpha cut retention rule

For a candidate partition, an alpha level and a sparsity level s, `retained_concepts` proceeds as follows.

1. Activity. Concept k is active for instance i when the alpha cut of its fuzzy attribution excludes zero, that is when the lower endpoint is positive or the upper endpoint is negative. The activity of concept k is the fraction of explained instances for which it is active.
2. Passing. A concept passes the cut when it is active for at least one instance (activity larger than zero).
3. Ranking. Passing concepts are ranked by the lower endpoint of the alpha cut of their fuzzy global importance, in decreasing order; ties are broken by decreasing activity.
4. Sparsity cap. The first s concepts of that ranking are retained. When fewer than s concepts pass, all passing concepts are retained.

Increasing alpha shrinks every alpha cut, so the set of passing concepts is nested in alpha; this is the mechanism behind the monotonicity of Property 6. A configuration that retains no concept is infeasible: its objectives are set to infinity for the Pareto computation and it is excluded from the front. If no configuration is feasible a `RuntimeError` asks for a lower alpha grid or more resamples.

The decision grid of the exhaustive solver is, for every candidate partition, the product of `alpha_grid` (default 0, 0.25, 0.5, 0.75, 1) and `sparsity_grid` (default 1 to K). The NSGA II solver (pymoo, mixed variables: partition source, integer K, fuzzifier from the grid, integer sparsity, real alpha in [0, 1]) evaluates the same function and caches every distinct (partition, effective sparsity, rounded alpha) triple; the returned table contains every evaluated configuration. The `'fixed'` solver evaluates one configuration (ablation).

### 4.2 Objectives of Equation 4

Fidelity loss f_1 is one minus the cross validated R squared of a surrogate that predicts the reference model output on the explained instances from the medians of the retained concept attributions: the sum over folds of the squared out of fold errors divided by n times the variance of the output, with `KFold(min(cv, n), shuffle=True, random_state)` (default five folds) and a ridge regression with penalty 1e-6 (`surrogate='linear'`) or a gradient boosting regressor with 100 trees of depth 2 (`surrogate='tree'`). The loss is one when nothing is retained or when fewer than four instances are explained, and zero when the output is constant. The held out data of the guide are therefore the out of fold instances of the explained set.

Complexity f_2 is the sum over retained concepts of (1 + F_k), the count of retained concepts weighted by one plus the concept fuzziness of Section 2.5. For a crisp partition it equals the number of retained concepts, and every concept of the uniform partition counts two.

Instability f_3 is one minus the mean pairwise Spearman correlation, over all pairs of pool members, of the concept importance vectors restricted to the retained concepts, where the importance of concept k for member p is the mean absolute attribution over the explained instances. Ranks are computed with average ties; when more than 2000 pairs exist a random subset of 2000 pairs is used with the given seed; a pair in which one member has constant ranks contributes one when the two rank vectors are equal and zero otherwise. The result is clipped to [0, 2] and rounded to 12 decimals. Single concept convention: with fewer than two retained concepts, or fewer than two pool members, the ranking is trivially stable and the instability is zero.

### 4.3 Pareto front and knee criteria

`pareto_mask` marks a configuration as non dominated when no other configuration is at least as good in every objective and strictly better in one. The knee point is chosen among the front members after min max normalization of each objective across the front (a constant objective is left unscaled).

- `'utopia'` (default): the member with the smallest Euclidean distance to the ideal point, the origin of the normalized objective space.
- `'hyperplane'`: the extreme points of the front are the members that minimize each normalized objective; a hyperplane is fitted through them (null vector of the extreme points augmented with a column of ones, oriented so that the normal has a positive sum) and the knee is the member with the largest distance below that hyperplane. When the extreme points are fewer than the number of objectives the utopia rule is used instead.

A front with a single member returns that member. The sensitivity script adds two representative selections that are not knee criteria: the sparsest front member whose fidelity loss is within 10 percent of the best fidelity loss on the front, and the member whose complexity is closest to the median complexity of the front.

### 4.4 Selection membership at feature level and its concept level aggregation

The guide defines the selection based membership of a concept as the proportion of Pareto optimal configurations in which it is retained. Concepts of different candidate partitions are not the same objects, so the proportion is computed at feature level and mapped back. For every Pareto optimal configuration with partition U and retained set R, feature j is retained with degree equal to the sum over k in R of u_jk, which is one for a feature that belongs crisply to a retained concept and zero for a feature of a discarded concept. The feature selection membership is the mean of that degree over the front members and lies in [0, 1] (`selection_membership`). For the reported partition the selection membership of concept k is the membership weighted average of the feature selection memberships, the sum over j of u_jk s_j divided by the sum over j of u_jk (`concept_selection_membership`). With crisp partitions and a single candidate this reduces to the proportion of the guide.

## 5. Outputs (Section 3.6, Step 8)

Module: `explainer.py`, class `PFCAExplanation`.

### 5.1 Defuzzified feature vector

The defuzzified concept attribution of instance i and concept k is the centroid of its trapezoid (`concept_centroids`). The defuzzified feature vector redistributes the concept centroids through the membership matrix in proportion to membership: psi_j equals the sum over k of c_k u_jk divided by the sum over j' of u_j'k (`redistribute_to_features`). The total is preserved, and with the identity membership matrix the feature vector equals the concept vector, so that with B and M equal to one the method returns the SHAP values of the reference model (`tests/test_properties.py::test_reduction_to_shap`). With `retained_only=True` the centroids of discarded concepts are set to zero before redistribution.

### 5.2 Other reported quantities

`instance_frame` lists, per concept, the centroid, median, support, core, sign confidence, disagreement index, type 2 flag, best label with its compatibility and the selection membership. `global_frame` lists the rank, the centroid and support of the global fuzzy importance, the dominance, the activity, the fuzziness and the selection membership. `front()` returns the Pareto optimal rows of the configuration table, and `selection.table` the full table with the objectives, the retained sets, feasibility, Pareto and knee flags. `explanation_at(explanation, index)` rebuilds the explanation for another evaluated configuration, for example another front member. `to_dict` and `save` serialize everything except the raw pool.

## 6. Synthetic generators (Section 6.1)

Module: `evaluation/synthetic.py`.

### 6.1 Process

Latent factors z_1 to z_F are independent standard normal. The d original features are split into F contiguous blocks of sizes as equal as possible, the first blocks receiving the extra features. Every feature of block k is x_j equal to sqrt(rho) z_k plus sqrt(1 minus rho) e_j with independent standard normal e_j, so that features are standard normal and the within block correlation equals rho. The default number of factors is round(sqrt(d)) with a minimum of two. The signal is the sum over k of beta_k g_k(z_k), with coefficients spaced linearly from 2 to 0.5 and random signs, and link functions taken in turn from linear, sin(1.5 z), tanh(1.5 z), z^2 minus 1 and 0.3 z^3 (all linear when `nonlinear=False`). The noise is Gaussian with standard deviation chosen so that the variance of the signal is `signal_to_noise` (default 4) times the noise variance.

### 6.2 Posterior mean of the factor given its block

Given the n_k original features of block k, the factor is normal with mean c_k times the sum of the block features, where c_k equals sqrt(rho) / (1 + (n_k minus 1) rho), and variance tau_k^2 equal to 1 minus n_k rho / (1 + (n_k minus 1) rho). Duplicate features never enter the posterior, so the truth of the redundancy family is that of the original features.

### 6.3 Gauss Hermite quadrature

The true regression function E[y | x] is the sum over k of beta_k E[g_k(z_k) | x]. For a nonlinear link the conditional expectation is evaluated by probabilists' Gauss Hermite quadrature with 40 nodes (`numpy.polynomial.hermite_e.hermegauss`): the nodes are scaled by tau_k and shifted by the posterior mean, and the weights are divided by sqrt(2 pi). The population mean of a link, used to center the attributions, is obtained with the same rule at mean zero and variance one. For the linear link the posterior mean is used directly.

### 6.4 True concept attributions and the interaction split

Because the regression function is additive across independent blocks, the Shapley value of block k with a marginal baseline is the centered block contribution beta_k (E[g_k(z_k) | x] minus E[g_k(z_k)]). In the interaction family the term gamma z_1 z_2 (gamma equal to 1.5) is added to the signal; its conditional expectation is gamma times the product of the two posterior means, and it is split equally between the first two factors, one half each. The true global importance is the mean absolute true attribution over the explained instances.

### 6.5 Redundancy family

By default F duplicates are appended after the d original features. Duplicate i copies the first feature of block i modulo F; duplicates with even index are exact copies and duplicates with odd index add Gaussian noise with standard deviation `duplicate_noise` (default 0.1). Duplicates are named `dup{i}_of_x{source}` and are added to the block of their source, so that the true membership matrix assigns them to the concept of the source. `SyntheticProblem.sample` draws fresh data from the same process with the same coefficients, links and duplicates, for the tuning, explained and replicate samples.

## 7. Evaluation conventions used by the experiment scripts

The attribution pool of a problem is fitted once and shared by PFCA, its ablations, feature level SHAP (pool member b equal to 0, m equal to 0) and bootstrapped SHAP (resamples of the reference model class), so that differences between methods are due to the explanation method only (`evaluation/protocol.py`). The ablations are crisp concepts (`pfca_crisp`), the reference model class only (`pfca_single_model`, subset of the pool), a single fixed configuration (`pfca_fixed`: the middle value of the concept grid, fuzzifier 2, all concepts retained, alpha 0) and the a priori partition only (`pfca_apriori`). Deletion and insertion curves remove the features of a concept to the degree of their membership and impute them from the k nearest background rows on the kept features (conditional imputation, k equal to 10). Calibration metrics compare the fuzzy supports and cores with a replicate attribution computed from a model refitted on a fresh sample (Phase A) or on the training split of the next repetition (Phases B and C). Runtime and memory are measured with `tracemalloc` and the process resident set size (`evaluation/budget.py`).

## 8. Properties of Section 5 and the tests that check them

| Property | Statement checked | Test |
|---|---|---|
| 1. Efficiency at the concept level | For random row stochastic U with K in {1, 2, 5}, concept attributions plus the expected value equal the model output for every pool member | `tests/test_properties.py::test_efficiency_at_concept_level` |
| 2. Reduction to SHAP | Identity a priori partition, B equal to 1 and M equal to 1: the defuzzified feature vector and the concept centroids equal the TreeSHAP values of the reference model to 1e-8; all quantiles coincide, sign confidence is one and disagreement is zero | `tests/test_properties.py::test_reduction_to_shap` |
| 3. Symmetry at the concept level | Two blocks with identical coalition contributions of a linear model receive identical grouped Shapley values and identical aggregated concept attributions | `tests/test_properties.py::test_symmetry_at_concept_level` |
| 4. Invariance to feature duplication | An exact copy halves the feature level Shapley value of the copied feature while the concept absorbing the copy keeps its attribution (exact, linear model); on a fitted gradient boosting model the concept level change is smaller than the feature level change | `tests/test_properties.py::test_duplication_invariance`, `tests/test_properties.py::test_duplication_invariance_fitted_model` |
| 5. Consistency of the fuzzy summary | The maximum deviation of the trapezoid parameters from the normal quantiles decreases from B equal to 20 to 200 to 20000 and is below 0.05 at the largest B; the trapezoid equals the empirical percentiles | `tests/test_fuzzification.py::test_consistency_of_fuzzy_summary`, `tests/test_fuzzification.py::test_fuzzify_matches_percentiles`, `tests/test_fuzzification.py::test_fuzzify_array_consistency` |
| 6. Monotonicity of the Pareto front | Complexity is non decreasing in alpha at fixed sparsity in a fitted explanation; the set passing the alpha cut is nested and complexity is non decreasing on a synthetic pool; a fuzzy number that excludes zero at one alpha excludes it at every larger alpha | `tests/test_properties.py::test_monotonicity_of_front_and_selection_membership`, `tests/test_selection.py::test_retention_monotone_in_alpha`, `tests/test_fuzzification.py::test_excludes_zero_monotone_in_alpha` |
| 6. Selection membership well defined in [0, 1] | Feature and concept selection memberships lie in the unit interval, with the exact values of a hand computed front | `tests/test_properties.py::test_monotonicity_of_front_and_selection_membership`, `tests/test_selection.py::test_selection_membership_bounds`, `tests/test_selection.py::test_selector_grid` |

Supporting checks of the definitions above: the closed forms of Equation 3 and of the possibility degree against brute force evaluation (`tests/test_fuzzification.py::test_compatibility_closed_form`, `test_possibility_closed_form`), the Ruspini property of the default label set and the standardization scale (`test_linguistic_labels_ruspini`), the union rule of the type 2 number (`test_type2_union`), sign confidence and disagreement index (`test_sign_confidence_and_disagreement`), the complexity weight (`tests/test_selection.py::test_complexity_crisp_equals_count`), the single concept convention of the instability (`test_instability`), the fidelity loss (`test_fidelity_loss`), the Pareto mask and both knee criteria (`test_pareto_mask_and_knee`) and the two solvers (`test_selector_grid`, `test_selector_nsga2`).
