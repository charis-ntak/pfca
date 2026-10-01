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
interaction | PFCA | 20 | 0.769 (0.276) | 0.437 (0.543) | 0.820 (0.359) | 0.387 (0.134) | 0.679 (0.072) | 0.874 (0.106)
interaction | SHAP | 20 | 0.531 (0.142) | 0.945 (0.110) | 0.478 (0.152) | 0.293 (0.098) | 0.950 (0.036) | 0.963 (0.072)
interaction | Bootstrapped SHAP | 20 | 0.486 (0.131) | 0.940 (0.123) | 0.478 (0.152) | 0.287 (0.096) | 0.939 (0.042) | 0.887 (0.101)
interaction | Grouped SHAP | 20 | 0.530 (0.142) | 0.955 (0.094) | 0.871 (0.152) | 0.225 (0.078) | 0.776 (0.048) | 0.815 (0.193)
interaction | LIME | 20 | 1.375 (0.121) | 0.890 (0.159) |  | 0.400 (0.092) | 0.820 (0.051) | 0.209 (0.295)
interaction | PFCA, crisp | 20 | 0.595 (0.206) | 0.884 (0.157) | 0.982 (0.046) | 0.273 (0.064) | 0.706 (0.067) | 0.880 (0.077)
interaction | PFCA, single class | 20 | 0.745 (0.248) | 0.503 (0.472) | 0.769 (0.363) | 0.367 (0.157) | 0.695 (0.079) | 0.891 (0.105)
interaction | PFCA, fixed | 20 | 0.520 (0.204) | 0.901 (0.135) | 0.602 (0.326) | 0.439 (0.147) | 0.796 (0.104) | 0.910 (0.058)
interaction | PFCA, a priori | 20 | 0.463 (0.128) | 0.910 (0.141) | 0.870 (0.161) | 0.244 (0.069) | 0.762 (0.046) | 0.795 (0.171)
redundancy | PFCA | 15 | 0.699 (0.295) | 0.548 (0.498) | 0.986 (0.045) | 0.359 (0.072) | 0.724 (0.055) | 0.909 (0.057)
redundancy | SHAP | 15 | 0.482 (0.129) | 0.913 (0.151) | 0.474 (0.119) | 0.311 (0.107) | 0.987 (0.045) | 0.982 (0.033)
redundancy | Bootstrapped SHAP | 15 | 0.433 (0.123) | 0.920 (0.152) | 0.474 (0.119) | 0.286 (0.093) | 0.974 (0.057) | 0.915 (0.063)
redundancy | Grouped SHAP | 15 | 0.482 (0.129) | 0.913 (0.151) | 0.852 (0.163) | 0.228 (0.045) | 0.814 (0.041) | 0.841 (0.114)
redundancy | LIME | 15 | 1.400 (0.144) | 0.867 (0.176) |  | 0.413 (0.095) | 0.855 (0.045) | -0.008 (0.485)
redundancy | PFCA, crisp | 15 | 0.592 (0.162) | 0.879 (0.173) | 1.000 (0.000) | 0.319 (0.057) | 0.715 (0.051) | 0.908 (0.052)
redundancy | PFCA, single class | 15 | 0.675 (0.260) | 0.646 (0.414) | 0.852 (0.351) | 0.385 (0.116) | 0.717 (0.065) | 0.926 (0.052)
redundancy | PFCA, fixed | 15 | 0.481 (0.180) | 0.801 (0.207) | 0.627 (0.266) | 0.380 (0.150) | 0.849 (0.099) | 0.931 (0.040)
redundancy | PFCA, a priori | 15 | 0.416 (0.119) | 0.907 (0.149) | 0.846 (0.163) | 0.243 (0.049) | 0.800 (0.049) | 0.832 (0.112)
Note: n is the number of cells times seeds. Lower is better for the recovery error and the deletion AUC, higher is better for the other metrics. Empty cells mark metrics that do not apply to a method.
::end

::table tab:coverage | Interval coverage and sign confidence calibration in Phase A, mean (SD) over all cells and seeds
Method | support, truth | core, truth | support, replicate | core, replicate | sign confidence ECE
PFCA | 0.619 (0.165) | 0.299 (0.112) | 0.700 (0.064) | 0.346 (0.055) | 0.088 (0.033)
Bootstrapped SHAP |  |  | 0.662 (0.045) | 0.323 (0.039) | 0.106 (0.038)
PFCA, crisp | 0.728 (0.106) | 0.360 (0.080) | 0.720 (0.084) | 0.386 (0.133) | 0.079 (0.034)
PFCA, single class | 0.631 (0.154) | 0.319 (0.111) | 0.690 (0.069) | 0.343 (0.053) | 0.089 (0.037)
PFCA, fixed | 0.670 (0.179) | 0.320 (0.122) | 0.688 (0.061) | 0.331 (0.057) | 0.096 (0.040)
PFCA, a priori | 0.731 (0.098) | 0.360 (0.086) | 0.695 (0.059) | 0.339 (0.052) | 0.088 (0.041)
Note: Nominal coverage is 0.90 for the support and 0.50 for the core. ECE is the expected calibration error of the sign confidence against the sign agreement with the replicate attribution.
::end

::table tab:duplication | Relative change of the attribution when the duplicate features are added, redundancy family of Phase A, mean (SD)
Method | duplication change | n
PFCA | 0.295 (0.484) | 15
SHAP | 0.498 (0.104) | 15
Bootstrapped SHAP | 0.527 (0.029) | 15
Grouped SHAP | 0.051 (0.021) | 15
LIME | 0.486 (0.089) | 15
PFCA, crisp | 0.032 (0.025) | 15
PFCA, single class | 0.179 (0.206) | 15
PFCA, fixed | 0.318 (0.263) | 15
PFCA, a priori | 0.018 (0.006) | 15
Note: The change is computed between the explanation of the problem without the duplicates and the explanation of the same instances with the duplicates, on the attribution of the item that contains the duplicated feature.
::end

::table tab:mixed | Linear mixed model contrasts of every method against feature level SHAP in Phase A, per family and metric
Family | Metric | Method | Estimate | SE | 95% CI | p
additive | recovery error | Bootstrapped SHAP | -0.030 | 0.040 | [-0.108, 0.049] | 0.460
additive | recovery error | Grouped SHAP | +0.000 | 0.040 | [-0.079, 0.079] | 0.996
additive | recovery error | LIME | +0.953 | 0.040 | [0.874, 1.031] | <0.001
additive | recovery error | PFCA | +0.386 | 0.040 | [0.307, 0.465] | <0.001
additive | recovery error | PFCA, a priori | -0.041 | 0.040 | [-0.120, 0.038] | 0.309
additive | recovery error | PFCA, crisp | +0.156 | 0.040 | [0.078, 0.235] | <0.001
additive | recovery error | PFCA, fixed | +0.015 | 0.040 | [-0.063, 0.094] | 0.700
additive | recovery error | PFCA, single class | +0.354 | 0.040 | [0.275, 0.432] | <0.001
additive | rank stability | Bootstrapped SHAP | -0.000 | 0.049 | [-0.097, 0.097] | 1.000
additive | rank stability | Grouped SHAP | +0.355 | 0.049 | [0.258, 0.451] | <0.001
additive | rank stability | PFCA | +0.354 | 0.049 | [0.258, 0.451] | <0.001
additive | rank stability | PFCA, a priori | +0.342 | 0.049 | [0.246, 0.439] | <0.001
additive | rank stability | PFCA, crisp | +0.475 | 0.049 | [0.378, 0.571] | <0.001
additive | rank stability | PFCA, fixed | +0.133 | 0.049 | [0.036, 0.230] | 0.007
additive | rank stability | PFCA, single class | +0.385 | 0.049 | [0.288, 0.481] | <0.001
additive | deletion auc | Bootstrapped SHAP | -0.004 | 0.029 | [-0.062, 0.053] | 0.878
additive | deletion auc | Grouped SHAP | -0.076 | 0.029 | [-0.133, -0.019] | 0.009
additive | deletion auc | LIME | +0.093 | 0.029 | [0.036, 0.150] | 0.001
additive | deletion auc | PFCA | +0.084 | 0.029 | [0.027, 0.141] | 0.004
additive | deletion auc | PFCA, a priori | -0.064 | 0.029 | [-0.121, -0.007] | 0.027
additive | deletion auc | PFCA, crisp | -0.009 | 0.029 | [-0.066, 0.048] | 0.755
additive | deletion auc | PFCA, fixed | +0.132 | 0.029 | [0.075, 0.189] | <0.001
additive | deletion auc | PFCA, single class | +0.071 | 0.029 | [0.014, 0.129] | 0.014
interaction | recovery error | Bootstrapped SHAP | -0.045 | 0.039 | [-0.122, 0.032] | 0.256
interaction | recovery error | Grouped SHAP | -0.001 | 0.039 | [-0.078, 0.076] | 0.977
interaction | recovery error | LIME | +0.844 | 0.039 | [0.767, 0.921] | <0.001
interaction | recovery error | PFCA | +0.238 | 0.039 | [0.161, 0.315] | <0.001
interaction | recovery error | PFCA, a priori | -0.068 | 0.039 | [-0.145, 0.009] | 0.085
interaction | recovery error | PFCA, crisp | +0.065 | 0.039 | [-0.012, 0.142] | 0.100
interaction | recovery error | PFCA, fixed | -0.011 | 0.039 | [-0.088, 0.066] | 0.786
interaction | recovery error | PFCA, single class | +0.214 | 0.039 | [0.137, 0.291] | <0.001
interaction | rank stability | Bootstrapped SHAP | +0.000 | 0.057 | [-0.112, 0.112] | 1.000
interaction | rank stability | Grouped SHAP | +0.394 | 0.057 | [0.281, 0.506] | <0.001
interaction | rank stability | PFCA | +0.342 | 0.057 | [0.230, 0.455] | <0.001
interaction | rank stability | PFCA, a priori | +0.392 | 0.057 | [0.280, 0.504] | <0.001
interaction | rank stability | PFCA, crisp | +0.504 | 0.057 | [0.392, 0.617] | <0.001
interaction | rank stability | PFCA, fixed | +0.124 | 0.057 | [0.012, 0.237] | 0.030
interaction | rank stability | PFCA, single class | +0.292 | 0.057 | [0.179, 0.404] | <0.001
interaction | deletion auc | Bootstrapped SHAP | -0.006 | 0.033 | [-0.071, 0.059] | 0.858
interaction | deletion auc | Grouped SHAP | -0.067 | 0.033 | [-0.132, -0.003] | 0.042
interaction | deletion auc | LIME | +0.107 | 0.033 | [0.042, 0.172] | 0.001
interaction | deletion auc | PFCA | +0.094 | 0.033 | [0.030, 0.159] | 0.004
interaction | deletion auc | PFCA, a priori | -0.049 | 0.033 | [-0.114, 0.016] | 0.139
interaction | deletion auc | PFCA, crisp | -0.020 | 0.033 | [-0.085, 0.045] | 0.548
interaction | deletion auc | PFCA, fixed | +0.146 | 0.033 | [0.081, 0.211] | <0.001
interaction | deletion auc | PFCA, single class | +0.074 | 0.033 | [0.009, 0.139] | 0.025
redundancy | recovery error | Bootstrapped SHAP | -0.048 | 0.047 | [-0.140, 0.043] | 0.302
redundancy | recovery error | Grouped SHAP | +0.000 | 0.047 | [-0.092, 0.092] | 0.995
redundancy | recovery error | LIME | +0.919 | 0.047 | [0.827, 1.010] | <0.001
redundancy | recovery error | PFCA | +0.217 | 0.047 | [0.125, 0.309] | <0.001
redundancy | recovery error | PFCA, a priori | -0.066 | 0.047 | [-0.158, 0.026] | 0.161
redundancy | recovery error | PFCA, crisp | +0.110 | 0.047 | [0.018, 0.202] | 0.019
redundancy | recovery error | PFCA, fixed | -0.001 | 0.047 | [-0.093, 0.091] | 0.984
redundancy | recovery error | PFCA, single class | +0.193 | 0.047 | [0.101, 0.285] | <0.001
redundancy | rank stability | Bootstrapped SHAP | -0.000 | 0.054 | [-0.106, 0.106] | 1.000
redundancy | rank stability | Grouped SHAP | +0.378 | 0.054 | [0.272, 0.484] | <0.001
redundancy | rank stability | PFCA | +0.513 | 0.054 | [0.407, 0.619] | <0.001
redundancy | rank stability | PFCA, a priori | +0.372 | 0.054 | [0.266, 0.478] | <0.001
redundancy | rank stability | PFCA, crisp | +0.526 | 0.054 | [0.420, 0.633] | <0.001
redundancy | rank stability | PFCA, fixed | +0.153 | 0.054 | [0.047, 0.259] | 0.005
redundancy | rank stability | PFCA, single class | +0.378 | 0.054 | [0.272, 0.484] | <0.001
redundancy | deletion auc | Bootstrapped SHAP | -0.025 | 0.032 | [-0.088, 0.038] | 0.439
redundancy | deletion auc | Grouped SHAP | -0.083 | 0.032 | [-0.146, -0.020] | 0.010
redundancy | deletion auc | LIME | +0.102 | 0.032 | [0.039, 0.165] | 0.001
redundancy | deletion auc | PFCA | +0.048 | 0.032 | [-0.015, 0.111] | 0.132
redundancy | deletion auc | PFCA, a priori | -0.068 | 0.032 | [-0.130, -0.005] | 0.035
redundancy | deletion auc | PFCA, crisp | +0.008 | 0.032 | [-0.055, 0.071] | 0.802
redundancy | deletion auc | PFCA, fixed | +0.069 | 0.032 | [0.006, 0.132] | 0.031
redundancy | deletion auc | PFCA, single class | +0.074 | 0.032 | [0.011, 0.137] | 0.022
Note: Model: linear mixed model, random intercept per seed; sample size, dimension and correlation as categorical fixed effects. A negative estimate means a smaller value than SHAP.
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
Phase A | interaction | 20 | 64.90 | 12.95 | 0.00 | -0.67 | -0.11 | 0.52
Phase A | redundancy | 15 | 63.07 | 12.60 | 0.00 | -0.73 | -0.14 | 0.50
Phase A | all families | 59 | 65.15 | 13.02 | 0.00 | -0.70 | -0.15 | 0.53
sensitivity, solver | grid | 4 | 86.25 | 17.25 | 0.00 | -0.88 | -0.60 | 0.77
sensitivity, solver | nsga2 | 4 | 647.75 | 17.75 | 0.00 | -0.87 | -0.67 | 0.77
Note: Spearman correlations between the objectives over the feasible configurations; a negative value means that the two objectives conflict.
::end

::table tab:criteria | Pre registered success criteria evaluated on the reduced study
Criterion | Phase | Quantity | PFCA | Reference | Effect or p | Status
1 | A, redundancy family | recovery error | 0.699 | 0.482 | 0.835 (Cohen d) | not met
1 | A, redundancy family | rank stability | 0.986 | 0.474 | 4.065 (Cohen d) | met
2 | B | deletion auc | 0.639 | 0.814 |  | met
2 | B | insertion auc | 0.765 | 0.992 |  | not met
2 | B | surrogate r2 | 0.970 | 1.000 |  | not met
3 | A | support coverage truth | 0.619 | 0.900 |  | not met
3 | A | core coverage truth | 0.299 | 0.500 |  | not met
3 | A | support coverage replicate | 0.700 | 0.900 |  | not met
3 | A | core coverage replicate | 0.346 | 0.500 |  | not met
3 | B | support coverage replicate | 0.810 | 0.900 |  | not met
3 | B | core coverage replicate | 0.444 | 0.500 |  | not met
Note: Criterion 1: favorable paired difference with |Cohen d| of at least 0.5. Criterion 2: per dataset mean not worse than SHAP. Criterion 3: coverage within 0.05 of the nominal 0.90 (support) or 0.50 (core).
::end
