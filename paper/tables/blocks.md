::table tab:phasea | Correctness, faithfulness and stability per family and method in Phase A, mean (SD) over cells and seeds
Family | Method | n | recovery error | rank recovery | rank stability | deletion AUC | insertion AUC | surrogate R2
additive | PFCA | 24 | 0.816 (0.311) | 0.329 (0.519) | 0.868 (0.337) | 0.396 (0.122) | 0.709 (0.065) | 0.922 (0.049)
additive | SHAP | 24 | 0.430 (0.126) | 0.921 (0.141) | 0.514 (0.140) | 0.312 (0.113) | 0.975 (0.040) | 0.970 (0.050)
additive | Bootstrapped SHAP | 24 | 0.401 (0.120) | 0.938 (0.135) | 0.514 (0.140) | 0.307 (0.098) | 0.964 (0.039) | 0.932 (0.050)
additive | Grouped SHAP | 24 | 0.430 (0.126) | 0.921 (0.141) | 0.868 (0.140) | 0.236 (0.050) | 0.816 (0.048) | 0.805 (0.135)
additive | LIME | 24 | 1.383 (0.116) | 0.842 (0.167) |  | 0.405 (0.115) | 0.848 (0.055) | 0.127 (0.405)
additive | PFCA, crisp | 24 | 0.587 (0.181) | 0.854 (0.188) | 0.988 (0.051) | 0.303 (0.057) | 0.727 (0.050) | 0.926 (0.048)
additive | PFCA, single class | 24 | 0.784 (0.275) | 0.416 (0.470) | 0.898 (0.291) | 0.383 (0.153) | 0.728 (0.063) | 0.930 (0.063)
additive | PFCA, fixed | 24 | 0.446 (0.200) | 0.885 (0.156) | 0.647 (0.315) | 0.444 (0.148) | 0.829 (0.096) | 0.944 (0.037)
additive | PFCA, a priori | 24 | 0.389 (0.116) | 0.900 (0.156) | 0.856 (0.141) | 0.248 (0.051) | 0.805 (0.043) | 0.815 (0.134)
interaction | PFCA | 24 | 0.842 (0.307) | 0.327 (0.565) | 0.850 (0.333) | 0.387 (0.128) | 0.677 (0.070) | 0.885 (0.100)
interaction | SHAP | 24 | 0.510 (0.139) | 0.954 (0.102) | 0.466 (0.141) | 0.307 (0.098) | 0.949 (0.034) | 0.969 (0.067)
interaction | Bootstrapped SHAP | 24 | 0.471 (0.127) | 0.946 (0.114) | 0.466 (0.141) | 0.304 (0.096) | 0.940 (0.041) | 0.895 (0.094)
interaction | Grouped SHAP | 24 | 0.509 (0.139) | 0.963 (0.088) | 0.874 (0.139) | 0.231 (0.079) | 0.779 (0.047) | 0.806 (0.178)
interaction | LIME | 24 | 1.372 (0.116) | 0.867 (0.158) |  | 0.412 (0.092) | 0.818 (0.050) | 0.138 (0.317)
interaction | PFCA, crisp | 24 | 0.623 (0.199) | 0.877 (0.149) | 0.985 (0.043) | 0.284 (0.074) | 0.698 (0.068) | 0.891 (0.074)
interaction | PFCA, single class | 24 | 0.800 (0.273) | 0.403 (0.498) | 0.805 (0.340) | 0.380 (0.155) | 0.693 (0.076) | 0.887 (0.105)
interaction | PFCA, fixed | 24 | 0.497 (0.194) | 0.910 (0.126) | 0.647 (0.314) | 0.410 (0.152) | 0.794 (0.096) | 0.915 (0.055)
interaction | PFCA, a priori | 24 | 0.450 (0.123) | 0.917 (0.131) | 0.870 (0.147) | 0.248 (0.070) | 0.766 (0.045) | 0.786 (0.158)
redundancy | PFCA | 24 | 0.760 (0.313) | 0.483 (0.495) | 0.950 (0.205) | 0.378 (0.099) | 0.719 (0.051) | 0.910 (0.067)
redundancy | SHAP | 24 | 0.435 (0.137) | 0.929 (0.133) | 0.481 (0.117) | 0.341 (0.113) | 0.980 (0.042) | 0.967 (0.082)
redundancy | Bootstrapped SHAP | 24 | 0.400 (0.122) | 0.933 (0.134) | 0.481 (0.117) | 0.319 (0.102) | 0.971 (0.051) | 0.917 (0.085)
redundancy | Grouped SHAP | 24 | 0.435 (0.137) | 0.929 (0.133) | 0.875 (0.138) | 0.239 (0.054) | 0.809 (0.044) | 0.800 (0.165)
redundancy | LIME | 24 | 1.389 (0.127) | 0.842 (0.167) |  | 0.446 (0.111) | 0.853 (0.052) | -0.092 (0.494)
redundancy | PFCA, crisp | 24 | 0.600 (0.189) | 0.851 (0.184) | 1.000 (0.000) | 0.307 (0.066) | 0.725 (0.057) | 0.911 (0.052)
redundancy | PFCA, single class | 24 | 0.745 (0.293) | 0.545 (0.458) | 0.866 (0.337) | 0.397 (0.127) | 0.718 (0.055) | 0.931 (0.054)
redundancy | PFCA, fixed | 24 | 0.459 (0.206) | 0.843 (0.185) | 0.639 (0.304) | 0.391 (0.155) | 0.831 (0.095) | 0.940 (0.039)
redundancy | PFCA, a priori | 24 | 0.388 (0.118) | 0.921 (0.132) | 0.863 (0.139) | 0.250 (0.053) | 0.799 (0.046) | 0.815 (0.130)
Note: n is the number of cells times seeds. Lower is better for the recovery error and the deletion AUC, higher is better for the other metrics. Empty cells mark metrics that do not apply to a method.
::end

::table tab:coverage | Interval coverage and sign confidence calibration in Phase A, mean (SD) over all cells and seeds
Method | support, truth | core, truth | support, replicate | core, replicate | sign confidence ECE
PFCA | 0.593 (0.173) | 0.284 (0.114) | 0.703 (0.064) | 0.349 (0.054) | 0.085 (0.031)
Bootstrapped SHAP |  |  | 0.667 (0.045) | 0.327 (0.039) | 0.101 (0.038)
PFCA, crisp | 0.715 (0.110) | 0.351 (0.085) | 0.723 (0.083) | 0.390 (0.135) | 0.078 (0.033)
PFCA, single class | 0.602 (0.165) | 0.300 (0.114) | 0.686 (0.070) | 0.340 (0.054) | 0.087 (0.036)
PFCA, fixed | 0.669 (0.173) | 0.316 (0.118) | 0.692 (0.059) | 0.336 (0.055) | 0.091 (0.038)
PFCA, a priori | 0.733 (0.091) | 0.357 (0.079) | 0.699 (0.055) | 0.343 (0.051) | 0.084 (0.040)
Note: Nominal coverage is 0.90 for the support and 0.50 for the core. ECE is the expected calibration error of the sign confidence against the sign agreement with the replicate attribution.
::end

::table tab:duplication | Relative change of the attribution when the duplicate features are added, redundancy family of Phase A, mean (SD)
Method | duplication change | n
PFCA | 0.266 (0.419) | 24
SHAP | 0.524 (0.127) | 24
Bootstrapped SHAP | 0.531 (0.038) | 24
Grouped SHAP | 0.047 (0.023) | 24
LIME | 0.522 (0.129) | 24
PFCA, crisp | 0.071 (0.132) | 24
PFCA, single class | 0.187 (0.230) | 24
PFCA, fixed | 0.285 (0.283) | 24
PFCA, a priori | 0.017 (0.006) | 24
Note: The change is computed between the explanation of the problem without the duplicates and the explanation of the same instances with the duplicates, on the attribution of the item that contains the duplicated feature.
::end

::table tab:mixed | Linear mixed model contrasts of PFCA and of grouped SHAP against feature level SHAP in Phase A, per family and metric
Family | Metric | Method | Estimate | SE | 95% CI | p
additive | recovery error | Grouped SHAP | +0.000 | 0.040 | [-0.079, 0.079] | 0.996
additive | recovery error | PFCA | +0.386 | 0.040 | [0.307, 0.465] | <0.001
additive | rank stability | Grouped SHAP | +0.355 | 0.049 | [0.258, 0.451] | <0.001
additive | rank stability | PFCA | +0.354 | 0.049 | [0.258, 0.451] | <0.001
additive | deletion auc | Grouped SHAP | -0.076 | 0.029 | [-0.133, -0.019] | 0.009
additive | deletion auc | PFCA | +0.084 | 0.029 | [0.027, 0.141] | 0.004
interaction | recovery error | Grouped SHAP | -0.001 | 0.041 | [-0.082, 0.080] | 0.979
interaction | recovery error | PFCA | +0.332 | 0.041 | [0.251, 0.413] | <0.001
interaction | rank stability | Grouped SHAP | +0.408 | 0.050 | [0.310, 0.507] | <0.001
interaction | rank stability | PFCA | +0.384 | 0.050 | [0.285, 0.483] | <0.001
interaction | deletion auc | Grouped SHAP | -0.076 | 0.031 | [-0.136, -0.016] | 0.013
interaction | deletion auc | PFCA | +0.080 | 0.031 | [0.020, 0.140] | 0.009
redundancy | recovery error | Grouped SHAP | +0.000 | 0.041 | [-0.080, 0.080] | 0.996
redundancy | recovery error | PFCA | +0.325 | 0.041 | [0.245, 0.405] | <0.001
redundancy | rank stability | Grouped SHAP | +0.394 | 0.048 | [0.301, 0.487] | <0.001
redundancy | rank stability | PFCA | +0.469 | 0.048 | [0.375, 0.562] | <0.001
redundancy | deletion auc | Grouped SHAP | -0.102 | 0.028 | [-0.157, -0.048] | <0.001
redundancy | deletion auc | PFCA | +0.037 | 0.028 | [-0.018, 0.092] | 0.184
Note: Model: linear mixed model, random intercept per seed; sample size, dimension and correlation as categorical fixed effects. A negative estimate means a smaller value than SHAP. The contrasts of the other methods are given in the supplementary file phase_a_mixed_models.csv.
::end

::table tab:phaseb | Faithfulness, stability, calibration and cost on the breast cancer dataset, mean (SD) over 5 repetitions of the split
Method | deletion AUC | insertion AUC | surrogate R2 | rank stability | perturbation stability | support coverage | core coverage | time (s)
PFCA | 0.639 (0.050) | 0.765 (0.037) | 0.970 (0.010) | 1.000 (0.000) | 1.532 (0.389) | 0.810 (0.112) | 0.444 (0.104) | 10.082 (0.639)
SHAP | 0.814 (0.105) | 0.992 (0.006) | 1.000 (0.000) | 0.507 (0.033) | 2.223 (0.606) |  |  | 0.463 (0.069)
Bootstrapped SHAP | 0.819 (0.099) | 0.990 (0.008) | 0.976 (0.023) | 0.507 (0.033) | 3.055 (0.261) | 0.850 (0.031) | 0.566 (0.048) | 9.265 (1.371)
LIME | 0.902 (0.041) | 0.976 (0.006) | -0.168 (0.089) |  | 0.391 (0.384) |  |  | 1.050 (0.112)
PFCA, crisp | 0.656 (0.059) | 0.781 (0.077) | 0.979 (0.012) | 0.987 (0.029) | 1.657 (0.411) | 0.780 (0.130) | 0.439 (0.172) | 10.113 (0.778)
PFCA, single class | 0.645 (0.045) | 0.769 (0.042) | 0.975 (0.013) | 1.000 (0.000) | 1.464 (0.644) | 0.842 (0.111) | 0.509 (0.159) | 8.798 (0.406)
PFCA, fixed | 0.809 (0.039) | 0.919 (0.024) | 0.978 (0.013) | 0.862 (0.061) | 1.979 (0.266) | 0.738 (0.141) | 0.325 (0.110) | 5.433 (0.259)
Note: Surrogate R2 is computed on the retained items at the sparsity of the PFCA knee. Time is the wall clock time of the explanation step after the shared pool.
::end

::table tab:phasec | Concept recovery, faithfulness, stability and calibration on the simulated questionnaire, mean (SD) over 5 repetitions
Method | ARI vs subscales | deletion AUC | insertion AUC | surrogate R2 | rank stability | support coverage | type 2 attributions
PFCA | 0.347 (0.130) | 0.346 (0.065) | 0.626 (0.038) | 0.890 (0.092) | 0.933 (0.149) | 0.838 (0.058) | 5.6 (2.6)
SHAP | 0.000 (0.000) | 0.207 (0.037) | 0.953 (0.021) | 0.989 (0.017) | 0.517 (0.022) |  | 
Bootstrapped SHAP | 0.000 (0.000) | 0.206 (0.025) | 0.954 (0.024) | 0.948 (0.030) | 0.517 (0.022) | 0.810 (0.036) | 
Grouped SHAP | 1.000 (0.000) | 0.177 (0.045) | 0.773 (0.013) | 0.694 (0.111) | 0.848 (0.035) |  | 
LIME | 0.000 (0.000) | 0.373 (0.046) | 0.759 (0.022) | -0.019 (0.069) |  |  | 
PFCA, crisp | 0.285 (0.085) | 0.328 (0.027) | 0.612 (0.021) | 0.853 (0.124) | 1.000 (0.000) | 0.835 (0.059) | 5.0 (1.0)
PFCA, single class | 0.347 (0.130) | 0.343 (0.075) | 0.627 (0.050) | 0.901 (0.096) | 0.955 (0.100) | 0.808 (0.046) | 0.0 (0.0)
PFCA, a priori | 1.000 (0.000) | 0.188 (0.045) | 0.765 (0.009) | 0.694 (0.076) | 0.843 (0.018) | 0.806 (0.065) | 21.2 (7.8)
Note: ARI vs subscales is the adjusted Rand index between the hardened partition of the method and the a priori subscales; it equals one by construction for the a priori partition and the grouped baseline.
::end

::table tab:reliability | Reliability of the attribution per construct on the simulated questionnaire, mean over repetitions
Construct | PFCA relative support width | PFCA sign confidence | PFCA disagreement | PFCA fraction type 2 | PFCA overlap with subscale | Bootstrapped SHAP relative support width | Bootstrapped SHAP sign confidence
anxiety | 1.278 | 0.903 | 0.110 | 0.013 | 0.554 | 0.825 | 0.931
rumination | 1.355 | 0.907 | 0.115 | 0.020 | 0.865 | 1.106 | 0.921
self efficacy | 2.067 | 0.869 | 0.146 | 0.055 | 0.357 | 1.417 | 0.905
social support | 1.945 | 0.887 | 0.190 | 0.040 | 0.413 | 2.550 | 0.790
Note: For PFCA every retained concept is matched to the subscale with which it shares most membership mass (overlap). The relative support width is the width of the 5th to 95th percentile interval divided by the absolute median attribution; for bootstrapped SHAP the feature level values of the items of a subscale are summed before the width is computed.
::end

::table tab:sensitivity | Sensitivity of the explanation to the design choices of the method, mean (SD) over cells and seeds
Factor | Setting | n | recovery error | rank stability | support coverage | distinct front size | relative quantile MAD to largest B | front recovery | label sign agreement
resamples | B10 | 4 | 1.158 (0.094) | 1.000 (0.000) | 0.379 (0.024) | 13.500 (5.196) | 0.136 (0.011) |  | 1.000 (0.000)
resamples | B20 | 4 | 1.154 (0.091) | 1.000 (0.000) | 0.431 (0.020) | 18.500 (7.594) | 0.080 (0.005) |  | 1.000 (0.000)
resamples | B50 | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) | 0.000 (0.000) |  | 1.000 (0.000)
model_classes | M1 | 4 | 1.096 (0.116) | 1.000 (0.000) | 0.429 (0.051) | 13.250 (3.500) |  |  | 1.000 (0.000)
model_classes | M2 | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) |  |  | 1.000 (0.000)
model_classes | M3 | 4 | 1.095 (0.189) | 0.817 (0.215) | 0.467 (0.071) | 20.750 (5.377) |  |  | 1.000 (0.000)
shape | trapezoidal | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) |  |  | 1.000 (0.000)
shape | triangular | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) |  |  | 1.000 (0.000)
percentiles | s5-95_c25-75 | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) |  |  | 1.000 (0.000)
percentiles | s2.5-97.5_c25-75 | 4 | 1.157 (0.088) | 1.000 (0.000) | 0.512 (0.025) | 17.250 (6.850) |  |  | 1.000 (0.000)
percentiles | s10-90_c30-70 | 4 | 1.153 (0.089) | 1.000 (0.000) | 0.355 (0.027) | 17.250 (6.850) |  |  | 1.000 (0.000)
distance | correlation | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) |  |  | 1.000 (0.000)
distance | loading | 4 | 0.975 (0.279) | 0.706 (0.335) | 0.544 (0.091) | 14.000 (2.582) |  |  | 1.000 (0.000)
solver | grid | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) |  |  | 1.000 (0.000)
solver | nsga2 | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.750 (7.274) |  | 0.988 (0.024) | 1.000 (0.000)
knee | utopia | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) |  |  | 1.000 (0.000)
knee | hyperplane | 4 | 0.642 (0.343) | 0.833 (0.137) | 0.606 (0.109) | 17.250 (6.850) |  |  | 1.000 (0.000)
knee | sparsest_within_10pct | 4 | 0.718 (0.314) | 0.827 (0.102) | 0.599 (0.191) |  |  |  | 1.000 (0.000)
knee | median_complexity | 4 | 1.000 (0.058) | 0.809 (0.122) | 0.523 (0.018) |  |  |  | 1.000 (0.000)
labels | default | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) |  |  | 1.000 (0.000)
labels | narrow | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) |  |  | 0.836 (0.017)
labels | wide | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) |  |  | 0.875 (0.050)
labels | three_labels | 4 | 1.154 (0.090) | 1.000 (0.000) | 0.436 (0.023) | 17.250 (6.850) |  |  | 1.000 (0.000)
Note: The default setting appears under every factor. Settings: B is the number of resamples, M the number of model classes, s and c the percentile levels of the support and the core.
::end

::table tab:front | Size of the Pareto front and conflict between the objectives
Source | Group | n | front size | distinct explanations | fraction collapsed | corr(fidelity, complexity) | corr(fidelity, instability) | corr(complexity, instability)
Phase A | additive | 24 | 66.67 | 13.33 | 0.00 | -0.71 | -0.19 | 0.55
Phase A | interaction | 24 | 67.62 | 13.50 | 0.00 | -0.70 | -0.17 | 0.55
Phase A | redundancy | 24 | 64.04 | 12.79 | 0.00 | -0.70 | -0.16 | 0.54
Phase A | all families | 72 | 66.11 | 13.21 | 0.00 | -0.70 | -0.17 | 0.55
sensitivity, solver | grid | 4 | 86.25 | 17.25 | 0.00 | -0.88 | -0.60 | 0.77
sensitivity, solver | nsga2 | 4 | 647.75 | 17.75 | 0.00 | -0.87 | -0.67 | 0.77
Note: Spearman correlations between the objectives over the feasible configurations; a negative value means that the two objectives conflict.
::end

::table tab:criteria | Pre registered success criteria evaluated on the reduced study
Criterion | Phase | Quantity | PFCA | Reference | Effect or p | Status
1 | A, redundancy family | recovery error | 0.760 | 0.435 | 1.068 (Cohen d) | not met
1 | A, redundancy family | rank stability | 0.950 | 0.481 | 2.050 (Cohen d) | met
2 | B | deletion auc | 0.639 | 0.814 |  | met
2 | B | insertion auc | 0.765 | 0.992 |  | not met
2 | B | surrogate r2 | 0.970 | 1.000 |  | not met
3 | A | support coverage truth | 0.593 | 0.900 |  | not met
3 | A | core coverage truth | 0.284 | 0.500 |  | not met
3 | A | support coverage replicate | 0.703 | 0.900 |  | not met
3 | A | core coverage replicate | 0.349 | 0.500 |  | not met
3 | B | support coverage replicate | 0.810 | 0.900 |  | not met
3 | B | core coverage replicate | 0.444 | 0.500 |  | not met
Note: Criterion 1: favorable paired difference with |Cohen d| of at least 0.5. Criterion 2: per dataset mean not worse than SHAP. Criterion 3: coverage within 0.05 of the nominal 0.90 (support) or 0.50 (core).
::end
