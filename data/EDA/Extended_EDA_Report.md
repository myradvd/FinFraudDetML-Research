# Extended fraud EDA: held-out evidence and next modeling decisions

## Scope and evaluation protocol
This builds on the existing first-round EDA rather than repeating prevalence, univariate buckets or hourly summaries. Input: 20,000 rows, 353 frauds. No new data, dates or entity keys were invented.

The primary protocol is **5-fold stratified cross-validation repeated 2 times**, seed 20261003; each row's predictions are averaged across repetitions. All learned medians, peer references, P95 amount / P75 age thresholds, standardization, one-hot vocabularies, rule coefficients, anomaly models and predictive models are trained within folds only. Fixed EDA rules (1h>=4, 24h>=11, age<=30, known merchants, night) remain pre-stated. Missing numeric fields use train-fitted medians, never zero imputation. No oversampling is used. LightGBM uses 160 trees, 15 leaves, learning rate .04, min child count 70 and L2=5; these are fixed experiments, not tuned test-set winners.

**AP denotes scikit-learn average precision**, not trapezoidal integration of a PR curve. Confidence intervals use 300 paired outcome-stratified row bootstraps of fixed OOF predictions; they preserve prevalence and measure observation uncertainty conditional on the fitted models. They do not refit models or measure all training/split uncertainty; repeated CV folds have overlapping training sets, and no entity ID permits cluster bootstrap. The best model is selected from these comparisons, so its score is not an untouched final-test estimate.

Rates always include volume, fraud count, Wilson 95% interval, baseline lift and an unreliable flag for <30 rows or <5 frauds. Bootstrap ranking frequencies are conditional on a fixed candidate set. SHAP represents model association, not causation. All findings below are **EXPLORATORY**, except the three pre-stated focused interaction tests labeled **CONFIRMATORY** and corrected jointly across training folds with Holm. Here CONFIRMATORY means pre-stated for this extension only: hypotheses originated from an EDA of this same dataset. Multiplicity correction does not undo that prior selection, so this is not independent confirmation.

## Executive summary: evidence and recommended action
1. **EXPLORATORY — modeling baseline:** Highest observed AP is logistic: **0.2272**, bootstrap CI [0.1805, 0.2796]. **Action: use as candidate, not as a final-test claim**; compare the full model and calibration tables below before deployment.
2. **EXPLORATORY — semantic ambiguity:** The recorded counts and spend are not jointly consistent with a nonnegative, same-currency, inclusive current-transaction spend window. **Action: retain observed values and zero-spend flag; do not infer inclusive history or currency** from column names.
3. **EXPLORATORY — peer/relative features:** merchant_relative: delta -0.0005 [-0.0016, +0.0010]; country_relative: delta -0.0010 [-0.0017, -0.0003]; average_ticket_ratio: delta +0.0008 [-0.0005, +0.0024]; burst_share: delta +0.0015 [-0.0002, +0.0036]. **Action: drop the country-relative candidate from the amount-only logistic extension and test the other ratios further**; a poor standalone logistic extension does not establish irrelevance in a nonlinear full model.
4. **CONFIRMATORY — interaction hypotheses:** Night×device passes training-fold Holm-corrected LRTs, while amount×device and burst×amount do not. None of the three OOF AP increment intervals excludes zero. **Action: no evidence to prioritize explicit products over their main effects in these focused logistic comparisons**, despite descriptive high-risk combinations.
5. **EXPLORATORY — silent fraud:** **77 frauds (21.8%)** trigger none of the seven known positive flags. **Action: retain continuous and categorical information beyond binary rules**; inspect exported silent records and profiles.
6. **EXPLORATORY — anomaly feature:** Incremental AP is +0.0064, CI [-0.0026, +0.0143]. **Action: use only if positive held-out increment survives further validation**; anomalous is not synonymous with fraudulent.
7. **EXPLORATORY — ablations/redundancy:** Removal CIs are entirely below zero for **device, categorical, account_age**. **Action: retain these groups**; keep amount and velocity as main-effect candidates given point-estimate losses, while their incremental importance remains less certain. Do not infer irrelevance from correlated SHAP attribution.
8. **EXPLORATORY — data limitation:** Learning curve AP at 20% is 0.1782; at 100% 0.2113. Full-data gain versus 80% is +0.0057, CI [-0.0080, +0.0168]. **Action: no proven plateau and no conclusive last-step gain**; lack of significance is not proof of enough data.
9. **EXPLORATORY — weighted models and rank stability:** Unweighted boosting Brier is 0.0153, versus 0.0938 with balanced weights. **Action: prefer unweighted probabilities unless weighting is followed by separately validated calibration**; top-risk and SHAP rankings remain conditional on fixed fits.
10. **EXPLORATORY — missingness structure:** 975 rows are partially missing; the overall missing-group Fisher p-value is 0.454. **Action: no convincing standalone missingness effect**; indicators remain optional because their ablation CI overlaps zero.

## 1. Data semantics probes — EXPLORATORY
| probe                    | segment                  |   volume |   fraud_count |   fraud_rate |   wilson95_low |   wilson95_high |     lift | reliability                |
|:-------------------------|:-------------------------|---------:|--------------:|-------------:|---------------:|----------------:|---------:|:---------------------------|
| amount > spend           | amount > spend           |     3904 |            71 |       0.0182 |         0.0144 |          0.0229 |   1.0304 | adequate sample            |
| amount > spend           | comparison complement    |    15964 |           278 |       0.0174 |         0.0155 |          0.0196 |   0.9866 | adequate sample            |
| amount == spend          | comparison complement    |    19868 |           349 |       0.0176 |         0.0158 |          0.0195 |   0.9952 | adequate sample            |
| amount == spend          | amount == spend          |        0 |             0 |     nan      |       nan      |        nan      | nan      | unreliable: absent segment |
| zero spend               | comparison complement    |    19311 |           343 |       0.0178 |         0.0160 |          0.0197 |   1.0063 | adequate sample            |
| zero spend               | zero spend               |      557 |             6 |       0.0108 |         0.0049 |          0.0233 |   0.6103 | adequate sample            |
| zero spend and count >=1 | comparison complement    |    19162 |           339 |       0.0177 |         0.0159 |          0.0197 |   1.0023 | adequate sample            |
| zero spend and count >=1 | zero spend and count >=1 |      552 |             5 |       0.0091 |         0.0039 |          0.0210 |   0.5132 | adequate sample            |

### Interpretation
If spend includes the current nonnegative transaction on the same amount scale, amount should not exceed spend. Observed violations and zero-spend records contradict that strict interpretation. The most defensible working interpretation is that spend excludes the current transaction, uses a different window, or contains placeholder/inconsistent zeros; the dataset cannot choose among these. Count's minimum of 1 suggests an inclusive count or a floored count, but does not prove it. Given the inconsistent spend evidence, do not confidently declare inclusive counts. The most likely count convention is inclusive or floor-at-one; leave this explicitly unresolved.

There are **557 zero-spend records**, of which **552 have an observed positive 24h count and 5 have a missing count**. Thus the new prompt's statement of 557 zero-spend *with observed count>=1* is not exact. All 557 zero-spend records are profiled numerically and categorically against the rest; these profiles do not establish source semantics. Exactly zero observed records have amount equal to spend at half-cent tolerance; its absent rate is undefined, not 0% measured risk.


### Account age distribution
|          n |   minimum |      p25 |   median |       p75 |       p95 |       max |   fractional_count |   at_4000 |
|-----------:|----------:|---------:|---------:|----------:|----------:|----------:|-------------------:|----------:|
| 19863.0000 |    5.0000 | 342.0000 | 790.0000 | 1483.0000 | 3209.8000 | 4000.0000 |             0.0000 |  354.0000 |
Integer values, a minimum of 5 and an upper cap at 4,000 are compatible with days and potentially a synthetic cap; other units remain possible. No unit can be verified. Treat age as an ordinal numeric field, not a known calendar duration.


### Country amount distributions
| country   |   size |     mean |   median |    min |       max |       p90 |
|:----------|-------:|---------:|---------:|-------:|----------:|----------:|
| AU        |    584 | 732.6243 | 122.1200 | 2.5000 | 5939.9500 | 2832.5380 |
| GB        |    505 | 307.3847 |  74.4200 | 2.5000 | 4871.1000 |  653.3240 |
| ID        |    840 | 250.7113 |  83.8000 | 2.5000 | 5205.2000 |  562.2050 |
| JP        |    382 | 899.4482 | 134.4350 | 2.6100 | 5898.4000 | 3707.2980 |
| MY        |   1225 | 372.4126 |  76.1900 | 2.5000 | 9000.0000 |  755.7960 |
| PH        |    581 | 298.6006 |  84.8200 | 4.1400 | 4638.9900 |  708.4200 |
| SG        |  13931 | 199.4106 |  71.4600 | 2.5000 | 7643.4000 |  455.8000 |
| TH        |    644 | 253.5648 |  86.4200 | 3.4700 | 4572.6800 |  525.7700 |
| US        |    693 | 277.3688 |  84.1100 | 2.5000 | 4340.5900 |  699.9460 |
| Unknown   |    107 | 235.3177 |  76.1000 | 5.5700 | 3906.9500 |  393.2120 |
| VN        |    508 | 251.2523 |  86.9650 | 2.7700 | 4982.7600 |  517.4840 |
Country-specific distributions alone cannot identify currency or exchange-rate normalization. Country peer medians therefore remain statistical references, not FX conversions. No monetary cost or cross-currency comparison is asserted.


## 2. Relative / peer-group features — EXPLORATORY
| feature              |   single_score_AP |   single_score_ci95_low |   single_score_ci95_high |   AP_change |   ci95_low |   ci95_high |
|:---------------------|------------------:|------------------------:|-------------------------:|------------:|-----------:|------------:|
| merchant_relative    |            0.0287 |                  0.0246 |                   0.0363 |     -0.0005 |    -0.0016 |      0.0010 |
| country_relative     |            0.0368 |                  0.0320 |                   0.0442 |     -0.0010 |    -0.0017 |     -0.0003 |
| average_ticket_ratio |            0.0210 |                  0.0191 |                   0.0236 |      0.0008 |    -0.0005 |      0.0024 |
| burst_share          |            0.0204 |                  0.0185 |                   0.0228 |      0.0015 |    -0.0002 |      0.0036 |
| anomaly              |            0.1469 |                  0.1104 |                   0.1925 |      0.0064 |    -0.0026 |      0.0143 |
Single scores are evaluated in the pre-stated increasing-risk direction, without choosing orientation from held-out labels. Merchant/country references use training medians with a training-global median fallback. Average ticket is spend/max(count,1); the denominator is floored at 1 before ratio and log1p. Burst share is 1h/max(24h,1). Missing score inputs use training medians. Raw missingness still exists in the main pipeline. Deciles are descriptive bins of OOF scores, not deployment cutoffs; tables are exported and summarized in the decile chart. Relative-feature logistic tests compare log1p(amount) alone versus log1p(amount) plus the one score, with in-fold imputation/scaling.


## 3. Plausibility mismatches — EXPLORATORY
| hypothesis                 | segment                    |   volume |   fraud_count |   fraud_rate |   wilson95_low |   wilson95_high |    lift | reliability     |
|:---------------------------|:---------------------------|---------:|--------------:|-------------:|---------------:|----------------:|--------:|:----------------|
| card_present_non_SG        | card_present_non_SG        |     2150 |            27 |       0.0126 |         0.0086 |          0.0182 |  0.7115 | adequate sample |
| card_present_non_SG        | complement                 |    17850 |           326 |       0.0183 |         0.0164 |          0.0203 |  1.0347 | adequate sample |
| cash_transfer_card_present | cash_transfer_card_present |      547 |            13 |       0.0238 |         0.0139 |          0.0402 |  1.3465 | adequate sample |
| cash_transfer_card_present | complement                 |    19453 |           340 |       0.0175 |         0.0157 |          0.0194 |  0.9903 | adequate sample |
| old_account_new_device     | complement                 |    19482 |           339 |       0.0174 |         0.0157 |          0.0193 |  0.9859 | adequate sample |
| old_account_new_device     | old_account_new_device     |      518 |            14 |       0.0270 |         0.0162 |          0.0449 |  1.5313 | adequate sample |
| high_amount_young_account  | complement                 |    19967 |           346 |       0.0173 |         0.0156 |          0.0192 |  0.9818 | adequate sample |
| high_amount_young_account  | high_amount_young_account  |       33 |             7 |       0.2121 |         0.1068 |          0.3775 | 12.0182 | adequate sample |
| luxury_bank_transfer       | complement                 |    19927 |           345 |       0.0173 |         0.0156 |          0.0192 |  0.9809 | adequate sample |
| luxury_bank_transfer       | luxury_bank_transfer       |       73 |             8 |       0.1096 |         0.0566 |          0.2016 |  6.2090 | adequate sample |
These are hypotheses, not verified impossible combinations. Foreign-country card-present may simply mean international usage, and cash-transfer/card-present may represent legitimate category/channel semantics. Discard standalone segments flagged unreliable, and retain adequate-volume segments only when incremental predictive validation supports them. Top-quartile age and high-amount thresholds are estimated within training folds.


## 4. Interactions — CONFIRMATORY statistical tests, EXPLORATORY predictive increments
| interaction   |   main_AP |   interaction_AP |   AP_change |   ci95_low |   ci95_high |
|:--------------|----------:|-----------------:|------------:|-----------:|------------:|
| amount_device |    0.1092 |           0.1091 |     -0.0001 |    -0.0036 |      0.0039 |
| burst_amount  |    0.0997 |           0.0976 |     -0.0021 |    -0.0102 |      0.0058 |
| night_device  |    0.0607 |           0.0593 |     -0.0014 |    -0.0052 |      0.0015 |
Each pair compares a focused two-main-effect logistic model with the same model plus their product: log amount × device; burst indicator × log amount; night × device. Missing main effects use training medians. The LRT fits are unpenalized nested statsmodels logistic regressions on training folds only, while OOF scores use C=1 regularized logistic models. Holm correction covers all fold-by-hypothesis LRTs; these overlapping training-fold tests are not independent replication. Pointwise paired OOF AP intervals are not simultaneous significance tests. Separate statistical and predictive conclusions are mandatory.


### Training-only likelihood-ratio results
|   fold | interaction   |      LR |   df |      p | converged   |   holm_p |
|-------:|:--------------|--------:|-----:|-------:|:------------|---------:|
|      0 | amount_device |  1.7345 |    1 | 0.1878 | True        |   1.0000 |
|      0 | burst_amount  |  1.3317 |    1 | 0.2485 | True        |   1.0000 |
|      0 | night_device  | 17.1903 |    1 | 0.0000 | True        |   0.0008 |
|      1 | amount_device |  2.0281 |    1 | 0.1544 | True        |   1.0000 |
|      1 | burst_amount  |  0.3892 |    1 | 0.5327 | True        |   1.0000 |
|      1 | night_device  | 19.9067 |    1 | 0.0000 | True        |   0.0002 |
|      2 | amount_device |  1.2063 |    1 | 0.2721 | True        |   1.0000 |
|      2 | burst_amount  |  1.6580 |    1 | 0.1979 | True        |   1.0000 |
|      2 | night_device  | 20.5208 |    1 | 0.0000 | True        |   0.0002 |
|      3 | amount_device |  6.5986 |    1 | 0.0102 | True        |   0.2041 |
|      3 | burst_amount  |  0.2827 |    1 | 0.5949 | True        |   1.0000 |
|      3 | night_device  | 11.7872 |    1 | 0.0006 | True        |   0.0125 |
|      4 | amount_device |  2.4764 |    1 | 0.1156 | True        |   1.0000 |
|      4 | burst_amount  |  0.0428 |    1 | 0.8362 | True        |   1.0000 |
|      4 | night_device  | 20.5374 |    1 | 0.0000 | True        |   0.0002 |
|      5 | amount_device |  2.0044 |    1 | 0.1568 | True        |   1.0000 |
|      5 | burst_amount  |  1.7898 |    1 | 0.1809 | True        |   1.0000 |
|      5 | night_device  | 15.0445 |    1 | 0.0001 | True        |   0.0025 |
|      6 | amount_device |  6.3916 |    1 | 0.0115 | True        |   0.2179 |
|      6 | burst_amount  |  0.0703 |    1 | 0.7909 | True        |   1.0000 |
|      6 | night_device  | 14.7959 |    1 | 0.0001 | True        |   0.0028 |
|      7 | amount_device |  4.7415 |    1 | 0.0294 | True        |   0.5300 |
|      7 | burst_amount  |  0.0002 |    1 | 0.9901 | True        |   1.0000 |
|      7 | night_device  | 14.3217 |    1 | 0.0002 | True        |   0.0034 |
|      8 | amount_device |  1.7357 |    1 | 0.1877 | True        |   1.0000 |
|      8 | burst_amount  |  0.0549 |    1 | 0.8148 | True        |   1.0000 |
|      8 | night_device  | 17.9544 |    1 | 0.0000 | True        |   0.0006 |
|      9 | amount_device |  0.2904 |    1 | 0.5899 | True        |   1.0000 |
|      9 | burst_amount  |  3.1521 |    1 | 0.0758 | True        |   1.0000 |
|      9 | night_device  | 28.4462 |    1 | 0.0000 | True        |   0.0000 |

## 5. Signal coverage and silent fraud — EXPLORATORY
|   segment |   volume |   fraud_count |   fraud_rate |   wilson95_low |   wilson95_high |    lift | reliability     |
|----------:|---------:|--------------:|-------------:|---------------:|----------------:|--------:|:----------------|
|         0 |    10169 |            77 |       0.0076 |         0.0061 |          0.0095 |  0.4290 | adequate sample |
|         1 |     7334 |            97 |       0.0132 |         0.0109 |          0.0161 |  0.7494 | adequate sample |
|         2 |     1990 |            73 |       0.0367 |         0.0293 |          0.0459 |  2.0784 | adequate sample |
|         3 |      370 |            42 |       0.1135 |         0.0851 |          0.1499 |  6.4314 | adequate sample |
|         4 |       71 |            26 |       0.3662 |         0.2637 |          0.4824 | 20.7477 | adequate sample |
|         5 |       55 |            30 |       0.5455 |         0.4152 |          0.6697 | 30.9039 | adequate sample |
|         6 |       11 |             8 |       0.7273 |         0.4343 |          0.9025 | 41.2053 | unreliable      |

### Leading exact positive-flag intersections
| segment                                                                         |   volume |   fraud_count |   fraud_rate |   wilson95_low |   wilson95_high |    lift | reliability     |
|:--------------------------------------------------------------------------------|---------:|--------------:|-------------:|---------------:|----------------:|--------:|:----------------|
| NONE                                                                            |    10169 |            77 |       0.0076 |         0.0061 |          0.0095 |  0.4290 | adequate sample |
| risky_merchant                                                                  |     1884 |            39 |       0.0207 |         0.0152 |          0.0282 |  1.1728 | adequate sample |
| night                                                                           |     3640 |            30 |       0.0082 |         0.0058 |          0.0117 |  0.4670 | adequate sample |
| new_device + night                                                              |      260 |            21 |       0.0808 |         0.0534 |          0.1203 |  4.5762 | adequate sample |
| risky_merchant + night                                                          |      615 |            18 |       0.0293 |         0.0186 |          0.0458 |  1.6583 | adequate sample |
| new_device                                                                      |     1043 |            17 |       0.0163 |         0.0102 |          0.0259 |  0.9235 | adequate sample |
| high_amount + risky_merchant                                                    |      500 |            13 |       0.0260 |         0.0153 |          0.0440 |  1.4731 | adequate sample |
| burst_1h + high_24h + new_device + risky_merchant + night                       |       15 |            11 |       0.7333 |         0.4805 |          0.8910 | 41.5486 | unreliable      |
| burst_1h + high_24h + new_device + high_amount + risky_merchant                 |       20 |            10 |       0.5000 |         0.2993 |          0.7007 | 28.3286 | unreliable      |
| new_device + risky_merchant + night                                             |       50 |             9 |       0.1800 |         0.0977 |          0.3080 | 10.1983 | adequate sample |
| new_device + risky_merchant                                                     |      139 |             8 |       0.0576 |         0.0294 |          0.1095 |  3.2608 | adequate sample |
| burst_1h + high_24h + new_device + night + young_account                        |        8 |             6 |       0.7500 |         0.4093 |          0.9285 | 42.4929 | unreliable      |
| burst_1h + high_24h + new_device + night                                        |       17 |             6 |       0.3529 |         0.1731 |          0.5870 | 19.9967 | unreliable      |
| burst_1h + high_24h + new_device + high_amount + risky_merchant + young_account |        6 |             6 |       1.0000 |         0.6097 |          1.0000 | 56.6572 | unreliable      |
| new_device + high_amount + risky_merchant                                       |       24 |             6 |       0.2500 |         0.1200 |          0.4490 | 14.1643 | unreliable      |
| burst_1h + high_24h + new_device + risky_merchant                               |       10 |             4 |       0.4000 |         0.1682 |          0.6873 | 22.6629 | unreliable      |
| high_amount                                                                     |      221 |             4 |       0.0181 |         0.0071 |          0.0456 |  1.0255 | unreliable      |
| high_amount + risky_merchant + night                                            |       88 |             4 |       0.0455 |         0.0178 |          0.1111 |  2.5753 | unreliable      |
| young_account                                                                   |      416 |             3 |       0.0072 |         0.0025 |          0.0210 |  0.4086 | unreliable      |
| burst_1h + high_24h + risky_merchant                                            |        8 |             3 |       0.3750 |         0.1368 |          0.6943 | 21.2465 | unreliable      |
Known flags: 1h>=4; 24h>=11; device=1; amount>=training P95; luxury/cash_transfer/electronics merchant; night 23–04; age<=30. NONE means no observed positive flag, not necessarily all flags confidently false: missing-flag status is separately exported. Exact intersections partition records, while risk ranking segments can overlap.

Silent fraud numeric profiles cover every numeric predictor; categorical/device profiles and full records are exported, including IDs for audit only. ID is excluded from every model, anomaly detector and SHAP calculation.

Categorical composition tables include `share_within_group` to compare silent fraud with non-fraud distributions. Their groups are defined by the outcome, so their 100%/0% fraud rates are tautological profile bookkeeping, not predictive evidence. Numeric account age/unit limitations remain unchanged.


### Silent fraud numeric comparison
| feature               | group         |   rows |   missing_pct |      mean |   median |      p25 |       p75 |
|:----------------------|:--------------|-------:|--------------:|----------:|---------:|---------:|----------:|
| transaction_amount    | silent fraud  |     77 |        0.0000 |  159.7651 |  68.7300 |  25.3100 |  210.1800 |
| transaction_amount    | non-fraud     |  19647 |        0.0000 |  246.9477 |  74.2600 |  31.6650 |  198.9450 |
| transaction_amount    | flagged fraud |    276 |        0.0000 |  685.4813 | 326.5000 |  74.9950 |  854.6500 |
| transaction_hour      | silent fraud  |     77 |        0.0000 |   13.2857 |  13.0000 |   8.0000 |   18.0000 |
| transaction_hour      | non-fraud     |  19647 |        0.0000 |   11.4655 |  11.0000 |   5.0000 |   17.0000 |
| transaction_hour      | flagged fraud |    276 |        0.0000 |    9.6920 |   8.5000 |   2.0000 |   17.0000 |
| transactions_last_24h | silent fraud  |     77 |        2.5974 |    4.3200 |   4.0000 |   3.0000 |    5.0000 |
| transactions_last_24h | non-fraud     |  19647 |        0.7838 |    4.5853 |   4.0000 |   3.0000 |    6.0000 |
| transactions_last_24h | flagged fraud |    276 |        1.0870 |    8.5092 |   6.0000 |   4.0000 |   11.0000 |
| spend_last_24h        | silent fraud  |     77 |        0.0000 |  375.8127 | 149.1800 |  87.3200 |  325.6300 |
| spend_last_24h        | non-fraud     |  19647 |        0.6515 |  539.2488 | 214.5100 |  93.3200 |  562.8000 |
| spend_last_24h        | flagged fraud |    276 |        1.4493 | 1755.3991 | 993.9500 | 316.9700 | 2295.6050 |
| account_age           | silent fraud  |     77 |        0.0000 |  816.2468 | 455.0000 |  93.0000 | 1142.0000 |
| account_age           | non-fraud     |  19647 |        0.6820 | 1059.1975 | 796.0000 | 350.0000 | 1486.0000 |
| account_age           | flagged fraud |    276 |        1.0870 |  723.8462 | 280.0000 |  61.0000 | 1179.0000 |
| new_device            | silent fraud  |     77 |        0.0000 |    0.0000 |   0.0000 |   0.0000 |    0.0000 |
| new_device            | non-fraud     |  19647 |        0.6311 |    0.0853 |   0.0000 |   0.0000 |    0.0000 |
| new_device            | flagged fraud |    276 |        0.0000 |    0.4891 |   0.0000 |   0.0000 |    1.0000 |
| transactions_last_1h  | silent fraud  |     77 |        0.0000 |    0.8312 |   1.0000 |   0.0000 |    1.0000 |
| transactions_last_1h  | non-fraud     |  19647 |        0.7024 |    0.8547 |   1.0000 |   0.0000 |    1.0000 |
| transactions_last_1h  | flagged fraud |    276 |        1.4493 |    2.2169 |   1.0000 |   0.0000 |    4.0000 |

## 6. Rule score versus logistic and boosting — EXPLORATORY
| model             | metric         |   value |   ci95_low |   ci95_high |
|:------------------|:---------------|--------:|-----------:|------------:|
| rule_points       | AP             |  0.1838 |     0.1421 |      0.2380 |
| rule_points       | ROC_AUC        |  0.7385 |     0.7074 |      0.7720 |
| rule_points       | Brier          |  0.0156 |     0.0150 |      0.0161 |
| rule_points       | recall_1pct    |  0.2068 |     0.1700 |      0.2408 |
| rule_points       | precision_1pct |  0.3650 |     0.3000 |      0.4250 |
| rule_points       | recall_2pct    |  0.2748 |     0.2308 |      0.3216 |
| rule_points       | precision_2pct |  0.2425 |     0.2037 |      0.2838 |
| rule_points       | recall_5pct    |  0.3824 |     0.3271 |      0.4334 |
| rule_points       | precision_5pct |  0.1350 |     0.1155 |      0.1530 |
| logistic          | AP             |  0.2272 |     0.1805 |      0.2796 |
| logistic          | ROC_AUC        |  0.7742 |     0.7446 |      0.8013 |
| logistic          | Brier          |  0.0151 |     0.0145 |      0.0157 |
| logistic          | recall_1pct    |  0.2351 |     0.1955 |      0.2691 |
| logistic          | precision_1pct |  0.4150 |     0.3450 |      0.4750 |
| logistic          | recall_2pct    |  0.2890 |     0.2421 |      0.3443 |
| logistic          | precision_2pct |  0.2550 |     0.2137 |      0.3038 |
| logistic          | recall_5pct    |  0.4136 |     0.3626 |      0.4646 |
| logistic          | precision_5pct |  0.1460 |     0.1280 |      0.1640 |
| logistic_weighted | AP             |  0.2068 |     0.1636 |      0.2549 |
| logistic_weighted | ROC_AUC        |  0.7631 |     0.7357 |      0.7904 |
| logistic_weighted | Brier          |  0.1566 |     0.1541 |      0.1589 |
| logistic_weighted | recall_1pct    |  0.2323 |     0.1911 |      0.2748 |
| logistic_weighted | precision_1pct |  0.4100 |     0.3374 |      0.4850 |
| logistic_weighted | recall_2pct    |  0.2833 |     0.2323 |      0.3286 |
| logistic_weighted | precision_2pct |  0.2500 |     0.2050 |      0.2900 |
| logistic_weighted | recall_5pct    |  0.3768 |     0.3258 |      0.4349 |
| logistic_weighted | precision_5pct |  0.1330 |     0.1150 |      0.1535 |
| boosting          | AP             |  0.2159 |     0.1724 |      0.2615 |
| boosting          | ROC_AUC        |  0.7649 |     0.7377 |      0.7939 |
| boosting          | Brier          |  0.0153 |     0.0147 |      0.0159 |
| boosting          | recall_1pct    |  0.2238 |     0.1841 |      0.2564 |
| boosting          | precision_1pct |  0.3950 |     0.3250 |      0.4526 |
| boosting          | recall_2pct    |  0.2890 |     0.2450 |      0.3286 |
| boosting          | precision_2pct |  0.2550 |     0.2162 |      0.2900 |
| boosting          | recall_5pct    |  0.3966 |     0.3470 |      0.4434 |
| boosting          | precision_5pct |  0.1400 |     0.1225 |      0.1565 |
| boosting_weighted | AP             |  0.2037 |     0.1620 |      0.2499 |
| boosting_weighted | ROC_AUC        |  0.7550 |     0.7281 |      0.7833 |
| boosting_weighted | Brier          |  0.0938 |     0.0918 |      0.0955 |
| boosting_weighted | recall_1pct    |  0.2266 |     0.1898 |      0.2635 |
| boosting_weighted | precision_1pct |  0.4000 |     0.3350 |      0.4650 |
| boosting_weighted | recall_2pct    |  0.2720 |     0.2266 |      0.3144 |
| boosting_weighted | precision_2pct |  0.2400 |     0.2000 |      0.2775 |
| boosting_weighted | recall_5pct    |  0.3711 |     0.3271 |      0.4236 |
| boosting_weighted | precision_5pct |  0.1310 |     0.1155 |      0.1495 |
| boosting_anomaly  | AP             |  0.2223 |     0.1771 |      0.2679 |
| boosting_anomaly  | ROC_AUC        |  0.7602 |     0.7315 |      0.7894 |
| boosting_anomaly  | Brier          |  0.0153 |     0.0147 |      0.0158 |
| boosting_anomaly  | recall_1pct    |  0.2125 |     0.1756 |      0.2479 |
| boosting_anomaly  | precision_1pct |  0.3750 |     0.3100 |      0.4376 |
| boosting_anomaly  | recall_2pct    |  0.2776 |     0.2280 |      0.3244 |
| boosting_anomaly  | precision_2pct |  0.2450 |     0.2012 |      0.2863 |
| boosting_anomaly  | recall_5pct    |  0.3824 |     0.3343 |      0.4377 |
| boosting_anomaly  | precision_5pct |  0.1350 |     0.1180 |      0.1545 |

### Paired model AP comparisons (candidate minus reference)
| reference         | candidate   |   AP_change |   ci95_low |   ci95_high |
|:------------------|:------------|------------:|-----------:|------------:|
| rule_points       | logistic    |      0.0433 |     0.0162 |      0.0669 |
| boosting          | logistic    |      0.0113 |    -0.0045 |      0.0324 |
| logistic_weighted | logistic    |      0.0204 |     0.0092 |      0.0296 |
| boosting_weighted | boosting    |      0.0122 |    -0.0026 |      0.0265 |
The points score is 10 times the logistic linear predictor on the seven binary EDA flags with fold-fitted median imputation and missing indicators. Coefficients and intercepts are exported per fold; negative weights are allowed rather than manually forced positive. The sigmoid yields probabilities, so score ordering and probability ranking agree. No rounding was used in evaluation; rounding would require new validation. The score is transparent but its weights remain learned, not an independent hand-written rule benchmark.

Alert budgets choose exactly floor(budget × evaluation rows) highest-score transactions. Ties use stable row order, which is arbitrary and not chronology; point-score ties can make exact budget metrics tie-sensitive. Budgets do not require labels, unlike an optimized fraud decision threshold. CIs are conditional on observed prevalence. Class-weighted probabilities are evaluated as emitted, without post-hoc calibration; calibration plots/Brier scores reveal any distortion. An untouched final test or nested CV would be needed for an unbiased post-selection final performance claim.


### Mean fold-trained rule coefficients
| feature                        |    mean |    std |     min |     max |
|:-------------------------------|--------:|-------:|--------:|--------:|
| burst_1h                       |  1.0044 | 0.1865 |  0.7208 |  1.3654 |
| high_24h                       |  1.0645 | 0.1665 |  0.7856 |  1.3404 |
| high_amount                    |  0.6312 | 0.1022 |  0.4358 |  0.7684 |
| missingindicator_burst_1h      |  0.2103 | 0.1887 | -0.0172 |  0.5483 |
| missingindicator_high_24h      |  0.4222 | 0.3174 | -0.0557 |  0.9109 |
| missingindicator_new_device    | -0.3195 | 0.1402 | -0.5044 | -0.1586 |
| missingindicator_young_account |  0.2055 | 0.2547 | -0.1219 |  0.5057 |
| new_device                     |  1.5756 | 0.0642 |  1.4842 |  1.7033 |
| night                          |  0.6375 | 0.0530 |  0.5398 |  0.7115 |
| risky_merchant                 |  1.1350 | 0.0743 |  1.0298 |  1.2739 |
| young_account                  |  0.6044 | 0.1224 |  0.3221 |  0.7665 |

## 7. Isolation Forest anomaly feature — EXPLORATORY
|   segment |   volume |   fraud_count |   fraud_rate |   wilson95_low |   wilson95_high |   lift | reliability     |
|----------:|---------:|--------------:|-------------:|---------------:|----------------:|-------:|:----------------|
|         1 |     2000 |             9 |       0.0045 |         0.0024 |          0.0085 | 0.2550 | adequate sample |
|        10 |     2000 |           141 |       0.0705 |         0.0601 |          0.0826 | 3.9943 | adequate sample |
|         2 |     2000 |            19 |       0.0095 |         0.0061 |          0.0148 | 0.5382 | adequate sample |
|         3 |     2000 |            14 |       0.0070 |         0.0042 |          0.0117 | 0.3966 | adequate sample |
|         4 |     2000 |            15 |       0.0075 |         0.0046 |          0.0123 | 0.4249 | adequate sample |
|         5 |     2000 |            26 |       0.0130 |         0.0089 |          0.0190 | 0.7365 | adequate sample |
|         6 |     2000 |            34 |       0.0170 |         0.0122 |          0.0237 | 0.9632 | adequate sample |
|         7 |     2000 |            36 |       0.0180 |         0.0130 |          0.0248 | 1.0198 | adequate sample |
|         8 |     2000 |            31 |       0.0155 |         0.0109 |          0.0219 | 0.8782 | adequate sample |
|         9 |     2000 |            28 |       0.0140 |         0.0097 |          0.0202 | 0.7932 | adequate sample |
Isolation Forest uses raw non-label predictors, in-fold median/scaling/one-hot preprocessing, 120 trees and max_samples=256. It is fit on all training records, not fraud-filtered records. The positive anomaly score is minus score_samples; it is added to an otherwise identical unweighted boosting model. The train/test distinction of the fitted detector can affect its score feature, but no held-out row or label is used in fitting. Compare its paired AP increment in Section 2 rather than assuming tail enrichment guarantees predictive improvement.


## 8. OOF SHAP, interactions, redundancy and ablations — EXPLORATORY
| feature                          |   mean_absolute_SHAP |   ci95_low |   ci95_high |   top10_frequency |
|:---------------------------------|---------------------:|-----------:|------------:|------------------:|
| transaction_amount               |               0.2393 |     0.2362 |      0.2425 |            1.0000 |
| account_age                      |               0.2155 |     0.2133 |      0.2181 |            1.0000 |
| log_peer_merchant_category       |               0.1451 |     0.1433 |      0.1467 |            1.0000 |
| transaction_channel_card_present |               0.1330 |     0.1319 |      0.1341 |            1.0000 |
| new_device                       |               0.1287 |     0.1261 |      0.1311 |            1.0000 |
| spend_last_24h                   |               0.1027 |     0.1013 |      0.1042 |            1.0000 |
| log_avg_ticket_ratio             |               0.0998 |     0.0985 |      0.1011 |            1.0000 |
| log_peer_country                 |               0.0908 |     0.0898 |      0.0917 |            1.0000 |
| hour_cos                         |               0.0724 |     0.0715 |      0.0733 |            1.0000 |
| merchant_category_retail         |               0.0632 |     0.0619 |      0.0644 |            1.0000 |
SHAP main values are computed for all 20,000 held-out rows from first-repeat unweighted boosting fits, in raw log-odds units. SHAP interaction values use a uniform random sample of 250 held-out rows per fold (1,250 total); the export ranks mean absolute off-diagonal entries. Off-diagonal SHAP interaction convention divides each pair contribution across symmetric entries, so these are attribution strengths, not independent interaction tests. One-hot category levels are separate encoded features, and top-10 stability is among encoded features, not raw grouped columns.


### Leading SHAP interactions
| feature_a                  | feature_b                        |   mean_absolute_interaction |
|:---------------------------|:---------------------------------|----------------------------:|
| log_avg_ticket_ratio       | log_peer_merchant_category       |                      0.0292 |
| transaction_amount         | log_peer_country                 |                      0.0192 |
| account_age                | transaction_channel_card_present |                      0.0181 |
| transaction_amount         | spend_last_24h                   |                      0.0168 |
| log_peer_merchant_category | log_peer_country                 |                      0.0164 |
| transaction_amount         | account_age                      |                      0.0160 |
| account_age                | new_device                       |                      0.0153 |
| spend_last_24h             | new_device                       |                      0.0146 |
| account_age                | log_avg_ticket_ratio             |                      0.0141 |
| spend_last_24h             | account_age                      |                      0.0126 |
| transaction_amount         | log_peer_merchant_category       |                      0.0124 |
| transaction_amount         | transaction_channel_card_present |                      0.0122 |
| spend_last_24h             | log_avg_ticket_ratio             |                      0.0119 |
| account_age                | log_peer_merchant_category       |                      0.0117 |
| hour_sin                   | log_peer_merchant_category       |                      0.0115 |

### Training-fold redundancy correlations
|   fold | a                    | b                     |   spearman |
|-------:|:---------------------|:----------------------|-----------:|
|      0 | transactions_last_1h | transactions_last_24h |     0.3943 |
|      0 | transaction_amount   | spend_last_24h        |     0.5779 |
|      1 | transactions_last_1h | transactions_last_24h |     0.3928 |
|      1 | transaction_amount   | spend_last_24h        |     0.5819 |
|      2 | transactions_last_1h | transactions_last_24h |     0.4011 |
|      2 | transaction_amount   | spend_last_24h        |     0.5823 |
|      3 | transactions_last_1h | transactions_last_24h |     0.3983 |
|      3 | transaction_amount   | spend_last_24h        |     0.5786 |
|      4 | transactions_last_1h | transactions_last_24h |     0.3965 |
|      4 | transaction_amount   | spend_last_24h        |     0.5789 |

### What is redundant, and what is merely correlated?
Raw amount versus log amount, raw spend versus log spend, and raw age versus log age are monotonic duplicate representations. For trees they largely duplicate split orderings; for logistic regression, including raw and log forms supplies different basis functions and can represent nonlinearity. High-amount and burst indicators also duplicate information already present in their continuous parents. SHAP credit can be shared/arbitrarily shifted across these representations.

The 1h/24h count Spearman correlation is only approximately 0.39–0.40, and amount/recent-spend is approximately 0.58. These are related, not interchangeable fields; do not remove either solely on correlation. Group ablations do not settle individual within-group redundancy. A lean-tree follow-up could remove monotonic duplicates and threshold flags while preserving continuous parents, but that simplified model has not been evaluated here.


### Ablation: AP after removal minus full-model AP
| group       |   AP_change |   ci95_low |   ci95_high |
|:------------|------------:|-----------:|------------:|
| amount      |     -0.0127 |    -0.0256 |      0.0013 |
| velocity    |     -0.0113 |    -0.0278 |      0.0080 |
| device      |     -0.0377 |    -0.0548 |     -0.0177 |
| categorical |     -0.0169 |    -0.0315 |     -0.0038 |
| time        |     -0.0048 |    -0.0152 |      0.0069 |
| account_age |     -0.0187 |    -0.0315 |     -0.0068 |
| missingness |     -0.0019 |    -0.0058 |      0.0010 |
Removal deletes raw features, associated missing indicators, and all derived features needing that group. Missingness-only removal deletes explicit indicators, not the fact that medians may retain a statistical trace of missingness. There are no native NaNs after preprocessing. The comparison refits identical-complexity models inside every fold; negative change means the full model benefited from the group. Overlapping transformations make this a group dependency experiment, not an additive decomposition.


## 9. Learning curve — EXPLORATORY
|   training_fraction |     AP |   ci95_low |   ci95_high |   delta_to_full_AP_change |   delta_to_full_ci95_low |   delta_to_full_ci95_high |
|--------------------:|-------:|-----------:|------------:|--------------------------:|-------------------------:|--------------------------:|
|              0.2000 | 0.1782 |     0.1377 |      0.2229 |                    0.0331 |                   0.0119 |                    0.0563 |
|              0.4000 | 0.1894 |     0.1486 |      0.2389 |                    0.0219 |                   0.0015 |                    0.0419 |
|              0.6000 | 0.1960 |     0.1526 |      0.2453 |                    0.0153 |                  -0.0003 |                    0.0299 |
|              0.8000 | 0.2056 |     0.1628 |      0.2550 |                    0.0057 |                  -0.0080 |                    0.0168 |
|              1.0000 | 0.2113 |     0.1679 |      0.2578 |                    0.0000 |                   0.0000 |                    0.0000 |
Learning curves use the first stratified 5-fold repetition with fixed held-out folds and nested training subsets at 20/40/60/80/100%. Both classes are reduced proportionally, so class prevalence is approximately retained and only the number of frauds changes along with total data. All preprocessing/references are refit on each subset. Fold-level counts are exported. This tests data sensitivity of the fixed 160-tree model, not whether a differently tuned model would plateau. A small/non-significant final increment is inconclusive with only 353 total frauds and wide intervals.


## 10. Stability — EXPLORATORY
### Ten highest eligible risk segments by observed fraud rate
| segment                    |   volume |   fraud_count |   fraud_rate |   wilson95_low |   wilson95_high |    lift | reliability     |   top10_bootstrap_frequency | stability_label                              |
|:---------------------------|---------:|--------------:|-------------:|---------------:|----------------:|--------:|:----------------|----------------------------:|:---------------------------------------------|
| burst_1h & high_amount     |       59 |            24 |       0.4068 |         0.2909 |          0.5341 | 23.0470 | adequate sample |                      1.0000 | stable top-10                                |
| high_24h & risky_merchant  |      121 |            48 |       0.3967 |         0.3140 |          0.4857 | 22.4756 | adequate sample |                      1.0000 | stable top-10                                |
| new_device & high_amount   |       81 |            31 |       0.3827 |         0.2845 |          0.4916 | 21.6836 | adequate sample |                      0.9967 | stable top-10                                |
| burst_1h & new_device      |      144 |            54 |       0.3750 |         0.3001 |          0.4564 | 21.2465 | adequate sample |                      1.0000 | stable top-10                                |
| high_24h & high_amount     |       64 |            23 |       0.3594 |         0.2529 |          0.4818 | 20.3612 | adequate sample |                      0.9933 | stable top-10                                |
| high_24h & new_device      |      161 |            56 |       0.3478 |         0.2786 |          0.4242 | 19.7069 | adequate sample |                      1.0000 | stable top-10                                |
| burst_1h & risky_merchant  |      149 |            49 |       0.3289 |         0.2586 |          0.4078 | 18.6322 | adequate sample |                      1.0000 | stable top-10                                |
| high_24h & night           |      124 |            37 |       0.2984 |         0.2249 |          0.3840 | 16.9058 | adequate sample |                      0.9600 | stable top-10                                |
| new_device & young_account |      103 |            25 |       0.2427 |         0.1701 |          0.3338 | 13.7518 | adequate sample |                      0.5333 | unreliable as top-10: unstable or low-ranked |
| burst_1h & high_24h        |      255 |            61 |       0.2392 |         0.1910 |          0.2952 | 13.5533 | adequate sample |                      0.4467 | unreliable as top-10: unstable or low-ranked |
Candidate set: seven flags, their pairwise positive intersections, and each observed merchant/country/channel level. Eligibility requires >=30 rows and >=5 frauds. Bootstrap ranking recomputes rates and sample eligibility; segments below 70% top-10 retention are labeled unreliable as top-10 claims. Low-ranked segments are not inherently erroneous; instability refers to their high-risk rank claim. Complete candidate and SHAP stability tables are exported. Correlated and overlapping segments are not independent discoveries. SHAP stability bootstraps fixed OOF attributions, not refitted models.


## 11. Missingness structure — EXPLORATORY
|   segment |   volume |   fraud_count |   fraud_rate |   wilson95_low |   wilson95_high |   lift | reliability     |
|----------:|---------:|--------------:|-------------:|---------------:|----------------:|-------:|:----------------|
|         0 |    19025 |           333 |       0.0175 |         0.0157 |          0.0195 | 0.9917 | adequate sample |
|         1 |      913 |            19 |       0.0208 |         0.0134 |          0.0323 | 1.1791 | adequate sample |
|         2 |       62 |             1 |       0.0161 |         0.0029 |          0.0859 | 0.9138 | unreliable      |

|   odds_ratio |   fisher_p |
|-------------:|-----------:|
|       1.1755 |     0.4543 |
Pairwise co-missing counts and expected overlap under independent missingness are exported with a heatmap. High overlap lift with tiny counts is not robust evidence. The partially missing-group Fisher test is exploratory and not multiplicity-adjusted, so it is not a confirmatory missingness claim. Because clean categorical Unknown is used for modeling, categorical profiles retain that category; explicit missing flags are evaluated via the missingness ablation.


## Final feature list and modeling setup
| group       | action                       | evidence                                     |
|:------------|:-----------------------------|:---------------------------------------------|
| amount      | test further / consider drop | removal delta -0.0127, CI [-0.0256, +0.0013] |
| velocity    | test further / consider drop | removal delta -0.0113, CI [-0.0278, +0.0080] |
| device      | use                          | removal delta -0.0377, CI [-0.0548, -0.0177] |
| categorical | use                          | removal delta -0.0169, CI [-0.0315, -0.0038] |
| time        | test further / consider drop | removal delta -0.0048, CI [-0.0152, +0.0069] |
| account_age | use                          | removal delta -0.0187, CI [-0.0315, -0.0068] |
| missingness | test further / consider drop | removal delta -0.0019, CI [-0.0058, +0.0010] |
**Candidate full feature list:** raw/log amount; 1h and 24h counts; raw/log spend; burst, burst share, average-ticket deviation, zero-spend; device; raw/log age, young/old flags; merchant/country/channel one-hot categories; raw hour, cyclic hour and night; high-amount indicator; targeted high×device, burst×high, night×device; train-referenced merchant/country amount ratios; explicit missingness indicators. The model experiment deliberately includes potentially redundant features so ablations and SHAP can diagnose them.

**Recommended starting model:** the evaluated **unweighted, regularized logistic model** is the leading candidate by observed AP and is more transparent than boosting. Its exact tested feature list is the full candidate list above, including optional/redundant transforms; pruning it has not been evaluated by the boosting ablations. Keep unweighted LightGBM as the nonlinear challenger. Balanced weighting underperforms in observed AP and greatly worsens emitted probability calibration in both model families.

**Recommended lean feature list for a follow-up experiment (not an already validated model):** amount/log amount; observed 1h/24h counts and log spend; new-device status; raw/log age; merchant, country and channel; cyclic hour and night. Keep explicit missingness flags optional. Test merchant-relative and average-ticket/burst-share transforms separately; do not automatically include the country-relative logistic extension, the three explicit product terms, or Isolation Forest on the strength of these inconclusive/negative increments. The full nonlinear model's SHAP ranking can still value peer features through interactions/redundancy, so removal there must be separately tested.

**Selection discipline:** keep groups supported by a negative removal CI; treat weak peer/anomaly gains and inconclusive groups as optional candidates, not certainties. Do not retrospectively present the full model's AP as the performance of a newly pruned feature list. Follow-up pruning/interaction changes require fresh nested validation or a separately reserved test set.

**Modeling setup:** stratified folds only, no invented chronology, all transformations in-fold, no SMOTE. Choose class-weighted versus unweighted candidates using the AP/budget/calibration evidence above. Retain the points score and logistic baseline for auditability; choose alert budgets from review capacity, not from test-label threshold search. At deployment refit the chosen pipeline and rules on the available training data, but performance here remains OOF exploratory validation. No entity key means repeated-user leakage cannot be assessed, and no time key means prospective drift cannot be measured. Currency, account-age units and feature availability at scoring remain unresolved; no additional data is assumed obtainable.

## Reproducibility and artifacts
Run the provided Python script from the directory containing the original CSV. The script defaults to 5 folds × 2 repetitions and 300 bootstraps; flags `--repeats` and `--bootstrap` change these. All random seeds are fixed. Packages: pandas, numpy, scipy, scikit-learn, lightgbm, shap, statsmodels, matplotlib, tabulate. Machine/package-version differences can cause small numeric differences. Package versions and configuration are saved. Charts are PNG; OOF predictions, per-fold rule weights, SHAP arrays, all result tables and this report are bundled. No predictive use of ID or post-outcome information was intentionally introduced, but source-feature leakage cannot be certified from this schema alone.
