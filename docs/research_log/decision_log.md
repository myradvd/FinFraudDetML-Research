# Datathon Track 2 — Fraud Detection: Project Context & Guidelines

> Read this before any work on the notebook. Update the **Decisions log** when something is agreed.
> Last updated: 2026-10-03

## 1. Goal
- **Deliverable:** one `.ipynb` containing the complete, reproducible ML solution:
  data loading & preprocessing → EDA → feature engineering → model training → evaluation → prediction generation (+ anything needed to reproduce).
- **Task:** binary classification. Predict `fraud` (0/1) per transaction.
- **Model philosophy:** simple, explainable, highly efficient. Prefer a model we can fully justify in the report over a marginally better black box.
- **Engineering target:** OOF PR-AUC (average precision, on probabilities) **≥ 0.22**, fallback **≥ 0.20**. Also report F1 and Recall at a chosen threshold.

## 2. Scoring
| Weight | Component |
|---|---|
| 60% | Results: **PR-AUC** on the hidden test labels |
| 40% | Technical report: understanding of the problem, **why this model**, and **the model's assumptions** |

**Confirmed by organisers (2026-10-03): results are judged on PR-AUC, F1 and Recall.** (The earlier RMSE/MAE/R² rubric text was wrong for this track.)

Implication for the notebook: every step needs a short markdown "why" cell. Assumptions must be stated explicitly (for example, logistic regression assumes log-odds linear in features, independent observations, and no perfect multicollinearity).

## 3. Files
| Path | Contents |
|---|---|
| `data/Track 2 Training Dataset.csv` | 20,000 rows, 12 cols, includes `fraud` |
| `data/Track 2 Testing Dataset.csv` | 12,000 rows, 11 cols, **no `fraud`** |
| `format/sample_submission.csv` | `id,prediction` — 12,000 rows, ids match test order exactly |
| `data/EDA/EDA_report.md` | Univariate/bivariate EDA — **training data only** |
| `data/EDA/Extended_EDA_Report.md` | CV experiments, ablations, SHAP — **training data only** |

**Submission (confirmed by user, 2026-10-03):** `prediction` = **fraud probability, a float in [0,1]**, even though the sample file shows integer 0s. PR-AUC is computed on the ranking of these probabilities.
- ⚠️ **Unresolved:** how F1/Recall are computed from probabilities is unknown. Most likely the organisers threshold at 0.5. Unweighted, calibrated probabilities at a 1.8% base rate rarely exceed 0.5, so few rows would be flagged and F1/Recall would be low. Ask the organisers, or decide on a mitigation (for example a monotone rescaling that keeps the ranking, and therefore PR-AUC, but moves the chosen F1-optimal cut-off to 0.5). Report OOF F1/Recall at the chosen threshold anyway.

**Columns:** `id` (exclude from features), `transaction_amount`, `transaction_hour` (0–23), `merchant_category` (11), `country` (10), `transaction_channel` (4), `transactions_last_24h`, `spend_last_24h`, `account_age` (0–4000, likely days, capped), `new_device` (0/1), `transactions_last_1h`, `fraud`.
There is no timestamp and no customer/account id, so only stratified random CV is possible.

## 4. Key facts from the training EDA
- **353 frauds / 20,000 = 1.765%.** A random ranker gets PR-AUC ≈ 0.018. Accuracy is meaningless here.
- **Already-measured OOF baselines (5-fold × 2 repeats):**
  - **Unweighted regularised logistic: AP 0.227** [0.18, 0.28]
  - LightGBM (unweighted): 0.216
  - Weighted logistic: 0.207
  - 7-flag rule score: 0.184
  - → **0.22 is reachable but sits mid-CI. Do not over-tune to CV noise.**
- **Strongest signals:** `new_device` (7.5% vs 1.2%), 1h count ≥4 (16%), 24h count ≥11 (20%), amount ≥ P95 (~6%), luxury/cash_transfer/electronics merchants, account_age ≤30, night 23–04 (2.7%), card_present is low risk.
- **Combinations are very strong** (≈38–40% fraud): high amount × new device, burst × high amount, high 24h × risky merchant. However, explicit product terms did **not** add AP in logistic tests. Night×device passed the LRT significance test.
- **Group ablations:** device, categorical and account_age clearly help. Amount and velocity probably help. Time and missingness indicators are inconclusive.
- **Class weighting hurts** both AP and calibration → train **unweighted**. **No SMOTE.**
- 21.8% of frauds trigger no rule flag → keep continuous features, not only binary flags.
- Missingness ~0.5–0.8% per column in train, not significantly related to fraud.
- `spend_last_24h` semantics are unclear (amount > spend in 3,904 rows; 557 zero-spend rows). Use as-is, don't over-interpret.

## 5. ⚠️ Train → test distribution shift (checked 2026-10-03)
| Feature | Train | Test |
|---|---|---|
| Rows with any missing value | 4.9% | **13.9%** |
| Per-column missing | 0.5–0.8% | **2.0–2.4%** |
| Amount P90 / P95 | 530 / 978 | **1,707 / 3,580** |
| Amount mean | 253 | **524** |
| Account age median | 790 | **1,158** |
| new_device rate | 9% | **15%** |
| Country AU / JP / SG | 2.9 / 1.9 / 70% | 5.0 / 4.3 / 64% |
| Channel ecommerce / mobile_app | 27.7 / 26.0% | 20.8 / 30.5% |

No unseen category levels. Ranges stay within train bounds (amount max 9,000; age max 4,000).

**Consequences:**
1. Don't let a linear model extrapolate on **raw** amount. Use `log1p` and/or clip to train quantiles.
2. Fixed thresholds learned on train (for example "amount ≥ train P95") will fire ~5× more often on test, so prefer continuous or log features over hard flags.
3. Missingness handling matters ~3× more on test. Use a robust median impute; avoid large-weight missing indicators.
4. Run **adversarial validation** (train-vs-test classifier) in the EDA to quantify the shift and decide on mitigations (for example importance weighting or more robust features).
5. Test prevalence is unknown and may differ (more new devices and high amounts point to possibly *higher* prevalence).

## 6. Working rules
- All imputation, scaling, encoding and thresholds are fitted **inside CV folds** (sklearn `Pipeline` + `ColumnTransformer`).
- CV: `RepeatedStratifiedKFold(5×2 or 5×3)`, fixed seed. Metric: `sklearn.metrics.average_precision_score`. Report mean ± std.
- Never tune on anything that touches test labels (we don't have them anyway). Use the test features **only** for shift diagnostics and the final predict.
- Keep it lean. Add a feature only if its OOF AP gain beats CV noise.
- Final model: refit on all 20k training rows → predict test probabilities → write `submission.csv` (`id,prediction`) in sample order.
- Environment: `sklearn` is **not installed** in the current Python, so set up a venv/requirements first (pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn, shap optional).

## 7. Notebook plan (status)
1. [ ] Setup & data loading
2. [ ] Preprocessing
3. [ ] EDA (training + train/test shift)
4. [ ] Feature engineering
5. [ ] Model training
6. [ ] Evaluation
7. [ ] Prediction & submission
8. [ ] Report notes (problem, model choice, assumptions, limitations)

## 8. Decisions log
- 2026-10-03: Metrics are PR-AUC, F1 and Recall. **Submit probabilities** (confirmed by user). OOF PR-AUC ≥0.22 is the target. How F1/Recall get thresholded is still open (see §3). Unweighted models, no SMOTE. Leading candidate: regularised logistic regression; challenger: shallow LightGBM.
- 2026-10-03: Fixed `src/TabPFN_fraud_improved_with_submission (1).ipynb`. It had been submitting binary 0/1, and now submits a continuous [0,1] score: the rank blend, mapped monotonically so the OOF-F1-optimal share sits at 0.5 (flagged → [0.51,1]). PR-AUC is unchanged, and F1/Recall at the 0.5 cut-off match CV. Validation was already correct: AP on scores, F1/Recall on thresholded labels.
- 2026-10-03: **Final model = regularised logistic regression** (`src/fraud_logistic.ipynb`, copied from `tabpfn.ipynb`, which was left untouched). It uses 9 numeric features (`transactions_last_24h, new_device, log_amt, log_spend, log_age, night, burst, high24, night_x_device`) plus 3 one-hot categoricals, C=0.3, median impute without missing indicators. OOF PR-AUC 0.232 (5×2), same as the original 24-feature spec (0.230) and better than LightGBM (0.216). Max VIF 2.6. Includes an odds-ratio table. TabPFN is an optional challenger (`RUN_TABPFN`). The submission flags the OOF-F1-optimal share (1.25%) at ≥0.5.
- 2026-10-03: **`fraud_logistic.ipynb` is logistic regression only** (the user asked for no TabPFN; LightGBM was removed too). L2, C=0.3, **unweighted**. `log_spend` dropped, leaving 8 numeric features: `transactions_last_24h, new_device, log_amt, log_age, night, burst, high24, night_x_device` + 3 one-hot categoricals.
  - Threshold: F1-max point on the full OOF PR curve, which flags the top 1.18%.
  - Results: OOF PR-AUC 0.233 (5×2), bootstrap CI [0.183, 0.279]; F1 0.296, recall 0.246, precision 0.370, MCC 0.292, 1.7 FP per TP. Max VIF 2.1.
  - The user approved: unweighted final model plus a comparison table; missingness tested and documented but kept out; WoE removed because it isn't better. All of these are reproducible in the notebook's "Design choices tested" table (§9).

- 2026-10-03: `fraud_logistic.ipynb` restructured. **Part 1 – Train & predict** is 3 code cells (data+features → CV+threshold → final fit+`submission.csv`); run alone, it gives an identical submission. **Part 2 – Exploration & analysis** is optional and reuses Part 1 objects: candidate features (`engineer_full`), missingness, screening, design-choices table, bootstrap CI, PR curve, metrics, submission-format check, assumptions/VIF, odds ratios. Part 1's `engineer()` builds only the 8 final features.

- 2026-10-03: New `src/fraud_blend.ipynb`: a simplified `tabpfn.ipynb` (5 code cells) blending logistic (24 features, as in tabpfn) + **XGBoost (replaces LightGBM)** + TabPFN by rank average. The XGBoost spec came from the user: exact `scale_pos_weight`=neg/pos, `binary:logistic`, `aucpr`, depth 3, min_child_weight 5, subsample .8, colsample .7, L1=1, L2=5. Local OOF without TabPFN: XGB **0.183** (LightGBM was 0.210), logistic+XGB blend 0.202 < logistic alone 0.224. **Cause: the exact scale_pos_weight** — unweighted XGB gets 0.204 and its blend 0.219; sqrt-ratio gets 0.196. Waiting on the user's decision; leaderboard check pending (needs Colab for TabPFN).

- 2026-10-03: `fraud_blend.ipynb` now uses **LightGBM again** (the user's choice; XGBoost removed), with the same params as tabpfn.ipynb and `scale_pos_weight=1`. CV: unweighted 0.210 > 3 (0.206) > sqrt ratio (0.200) > exact ratio (0.184). Model factories take no arguments. Local OOF without TabPFN: logistic 0.2236, LightGBM 0.2098, blend of the two 0.2197. The full 3-model blend needs Colab plus a leaderboard upload.

- 2026-10-03: **LightGBM tuned** in `fraud_blend.ipynb` (logistic + LightGBM + TabPFN). Random search: 70 configs, 5×2 CV, top 5 re-scored on fresh seed-1 splits. Chosen: n_estimators 900, lr 0.03, num_leaves 4, min_child_samples 150, colsample 0.85, L2 10, L1 0.5, min_split_gain 0.01, cat_smooth 100, min_data_per_group 50, unweighted. Over 3 split seeds: LightGBM 0.207 → 0.219, logistic+LightGBM blend 0.220 → 0.228. In the notebook's own CV (seed 0), without TabPFN: LightGBM 0.2168, blend 0.2248. The winning pattern is tiny trees with large leaves.

- 2026-10-03: The user renamed the blend notebook to `src/fraud_blend_NEW.ipynb` and deleted `fraud_logistic.ipynb`. Round-2 tuning (3 CV seeds, logistic+LightGBM only): **the blend's logistic now uses the lean 8 features without missing flags** (+0.0038 blend PR-AUC, better on 3/3 seeds; logistic alone 0.2295 vs 0.2249). Rejected: LightGBM 5-seed bagging (+0.0001, noise), logistic C 0.1/1 (noise), LightGBM subsample 0.8 or raw-only features (−0.007), learned blend weights (no gain on held-out halves). Notebook CV seed 0 without TabPFN: logistic 0.2280, LightGBM 0.2168, blend 0.2292. Next lever: TabPFN settings, which need Colab.

- 2026-10-03: The `fraud_blend_NEW.ipynb` threshold is now chosen from the confusion matrix. Fine grid of flag shares (0.2–6%, 0.05% steps); TP/FP/FN/TN, precision, recall and F1 at each; max of a 5-point smoothed F1; the confusion matrix is printed. The user wants PR-AUC maximised; the threshold doesn't affect PR-AUC (ranking only), so it targets F1. Local result without TabPFN: top 0.95%, precision 0.405, recall 0.218, F1 0.284 (TP 77, FP 113, FN 276).

- 2026-10-03: Round 3 (research-driven, 3 CV seeds + an adversarial shift-weighted AP proxy; train-vs-test AUC 0.67). **Added CatBoost as a 4th blend member** (depth 4, 800 iterations, lr 0.03, L2 10, native categoricals): blend +0.0052 (3/3 seeds), shift-weighted +0.0067. Rejected: frequency encoding (+0.0006), LightGBM focal loss γ=1 (noise) / γ=2 (broke), covariate-shift importance weighting (−0.004 to −0.005). Notebook seed 0 without TabPFN: logistic 0.2280, LightGBM 0.2168, CatBoost 0.2404, blend 0.2372; threshold top 1.65%, F1 0.290. Untested (Colab only): TabPFN `balance_probabilities`, `n_estimators`.

- 2026-10-03: On the leaderboard, the 4-model blend with CatBoost scored **worse** than the original 3-model blend (per the user). New `src/fraud_candidates.ipynb` writes 4 CSVs from one run: `submission_tabpfn_catboost.csv` (recommended, also `submission.csv`), `submission_catboost.csv`, `submission_lightgbm.csv`, `submission_tabpfn.csv`. Same 24-feature preprocessing; per-candidate confusion-matrix threshold. Local (no TabPFN): CatBoost 0.2404 (F1 0.308 at top 1.65%), LightGBM 0.2168. CatBoost random search: 20/30 configs, none beat the current settings (depth 4, 800 iterations, lr 0.03, L2 10); stopped early. The user renamed the blend notebook to `fraud_blend_NEW_withCatBoost.ipynb`.

- 2026-10-03: **Leaderboard (hidden test PR-AUC):** original tabpfn.ipynb blend (TabPFN+LightGBM+logistic) **0.19354**, CatBoost 0.19248, LightGBM 0.19006, CatBoost+TabPFN 0.18804, TabPFN 0.18750. The spread (0.006) is below the noise: bootstrap of OOF on test-sized samples (12k rows, ~212 frauds) gives a single-model SD of ~0.030 and a paired same-rows difference SD of ~0.012. All models drop ~0.04 from CV (~0.23) to test (~0.19), uniformly, so it's a dataset shift/test difficulty effect, not a model choice. Recommendation: submit the original 3-model blend (best observed and the most diversified); stop model-swapping; avoid picking the luckiest of many uploads (winner's curse); spend the remaining time on the report.

- 2026-10-03: **Shift diagnosis (to reach the 0.23 leaderboard target).** Ids are sequential (train 1–20000, test 20001–32000). There is no drift within train (all stats flat across id quintiles; Spearman(id, fraud) ≈ 0.001); test is a **step change** (amount P90 530→~1,700 with the bulk only ×1.15, new_device 9→15%, burst 2.3→4.4%, missing rows 5→14%, account age ×1.4). Big old-device transactions: 4.6%→12.1%. The covariate-shift-weighted proxy predicted ~0.34 but the real test is 0.19, so P(fraud|x) also differs: risky flags fire more often in test without fraud. Ranked strategies: (1) per-dataset percentile features (transductive), (2) adversarial feature pruning, (3) rely on flag combinations, (4) clip tails at the train P95–P99, (5) missingness robustness. Plan: one notebook, 3–4 single-change variants on the original 3-model blend; a real fix should show ≥ +0.04 (paired noise ±0.012).

- 2026-10-03: **The private leaderboard is on unseen data with an unknown shift, so the goal is a general, robust solution, not the public score.** Decided stance: no synthetic data for *training* (no new P(y|x) information); synthetic **stress-test validation sets** (scaled tails, 3× missing, more new devices/bursts, dropped categories) for model selection. No external real-world data (different schemas; check the rules). More folds only reduce estimate variance. Robustness candidates: monotone constraints (LightGBM), missing-value augmentation, clipping to the train range, a diverse rank ensemble (keep), and no public-LB selection. Proposed next: build the stress-test suite, then test those on the original 3-model blend.

- 2026-10-03: **Stress-test suite** (3 seeds; perturb the held-out fold only: S1 test-like scales, S2 3× missing, S3 flags without fraud, S4 category noise, S5 all). S5 drops models ~0.03, similar to the real public drop (0.23→0.19), so the suite looks realistic. Robustness ranking (mean S1–S5 / worst): logit+lgb+cat 0.2251/0.2053 ≈ logit+cat 0.2247/0.2023 > logit+lgb 0.2216/0.2045 > lgb+cat 0.2177 ≈ logit 0.2176 > cat 0.2123 > lgb 0.2081. Logistic is the robust anchor (unaffected by S1); two tree models together add little (lgb+cat is the worst blend). Rank correlations on the real test: logit-lgb 0.82, lgb-cat 0.86, logit-cat 0.88. TabPFN can't be stress-tested locally (public: worst alone, 0.1875). Candidate robust final: logistic+LightGBM+CatBoost (no TabPFN, so reproducible without a token or GPU).

- 2026-10-03: New `src/fraud_catboost_logistic.ipynb`: CatBoost (same params as `fraud_catboost.ipynb`, averaged over 5 seeds) + lean logistic (8 numeric + 3 one-hot, C=0.3), 50/50 rank blend, confusion-matrix threshold, writes `submission_catboost_logistic.csv` and `model.pkl` (dict of the plain sklearn/CatBoost models + flag share; reload check in the notebook). OOF PR-AUC over 3 split seeds: logistic 0.2295, CatBoost 1 seed 0.2279, 5-seed 0.2299 (+0.002, 3/3), **blend 0.2380 (+0.008 over either, 3/3)**; 60/40 0.2375, 70/30 0.2370. Light tuning (CatBoost depth 3/5, L2 20, lr 0.02×1200, rsm 0.5; logistic C 0.1/1) was stopped unfinished at the user's request; current settings kept. Notebook seed 0: logistic 0.2280, CatBoost 0.2410, blend 0.2405, flags top 1.65% (F1 0.293).
- 2026-10-03: New `src/fraud_catboost_logistic_shift.ipynb` (**not run yet**; the user will run it in Colab): CatBoost unchanged (24 features, 5 seeds, missing-value augmentation); logistic swaps the `burst`/`high24` flags and raw `transactions_last_24h` for `log_tx_1h`/`log_tx_24h`; blend 80/20 CatBoost/logistic; writes `submission_catboost_logistic_v2.csv` and `model_v2.pkl`. The earlier 70/30 + augmentation version (`fraud_catboost_logistic.ipynb`, seed-0 OOF: logistic 0.2278, CatBoost 0.2351, blend 0.2395) is unchanged; the shift experiments (relative features, per-dataset scaling, LogisticRegressionCV, depth 3) were stopped unfinished.
- 2026-10-03: **Public-test diagnosis + `fraud_catboost_logistic_shift.ipynb` updated (not fully run; the user runs it in Colab).** No leakage found (no duplicate rows within train or across train/test; all fitting in-fold). Cause of the CV ~0.24 vs public ~0.19 gap and of the blend losing to CatBoost: test has more flagged rows that look ordinary. New-device 9→15%, with the extra rows less night/risky and older accounts. 1h counts 5–9 are 3–4× more common, mostly small-amount, old-account, everyday-merchant users. Our top 2% is ~80% new-device and 60–70% bursts. Logistic regression gives each flag a fixed bonus (72% bursts in its test top 2%), so it hurts most. The AU/JP/MY ×10 amounts are not the cause (already ranked low). Changes: velocity counts capped at the train P99 (1h 5, 24h 16); a label-blind test-like perturbation of validation folds (new devices +6 pts, +2.4% extreme-velocity rows, AU/JP/MY amounts ×5–12 on 25% of their rows, missing 3×), reported as test-like PR-AUC; the blend weight is chosen from {1, .9, .8, .7, .5} by test-like PR-AUC. A smoke test with tiny CatBoost ran end to end; the test-like score drops ~0.03 below clean, like the real leaderboard drop.
- 2026-10-03: **Shift-notebook round 2 (measured, single split seed 0, CatBoost 1 seed, clean + refined test-like PR-AUC).** The test-like check is now label-blind and per-merchant: new-device and 5–9/h burst rates by merchant taken from the test features, AU/JP/MY big amounts on known devices, 3× missing. Results: the velocity cap **hurt** (CatBoost clean 0.2322→0.2259, test-like 0.2128→0.2031; logistic 0.2303→0.2221) → removed. Missing-value augmentation **hurts CatBoost** (0.2404→0.2322, test-like 0.2191→0.2128) but helps logistic slightly → logistic only. Rejected: logistic new-device context terms (−0.007), CatBoost without flags (−0.009), CatBoost depth 3 (−0.006). Final: CatBoost (no augmentation, depth 4, 24 features, 5 seeds) + logistic (log counts, augmentation), weight picked by test-like score; at 70/30 clean 0.2442, test-like 0.2233. Diagnosis: new-device extras are SG/card-present/everyday merchants, where training already shows low risk (3%), and CatBoost handles them (9% of its top 2%, like training frauds); account age only adds old accounts (young tail unchanged). The notebook passed a smoke test; the user runs the full notebook in Colab.
- 2026-10-04: The user reported the round-1/round-2 shift changes made the model worse. `src/fraud_catboost_logistic_shift.ipynb` was rebuilt from the v2 spec (CatBoost 24 features + logistic with log counts, augmentation for both, 80/20, no velocity cap, no test-like check or weight search) plus edge-case hardening: name-based columns, category normalisation (unseen → Unknown), impossible values → NaN, 1h ≤ 24h, logistic MinMax clip learned in-fold, amount/hour added to the augmentation at 1%, NaN-safe final scores with an id check against the sample file, and an edge-case cell. The features are verified identical to before on train and test. The user's earlier round-2 + hardening version is kept as `fraud_catboost_logistic_shift-new.ipynb`. Not fully run.

- 2026-10-05: **Repo restructured for publication.** Pipeline package `fraudlab/` (`python -m fraudlab all`: eda → select → train → predict → verify; the retrained model and submission are byte-identical to the 0.19586 upload). Notebooks cut to 3 (`01_final_catboost`, `02_model_comparison` = model zoo of 6 learners + 4 blends, `03_shift_reanchoring`) + `final_notebook/`. The bagged MLP was dropped from the zoo (~25 min, CV 0.183). Zoo result: CV leader catboost+logistic 50/50 0.2392 ≈ catboost 0.2384 (tied); the pre-stated rule picks catboost on public evidence (0.196 vs 0.183). All public scores are in `results/leaderboard.csv`.

## 9. Research insights for this dataset (from the pasted papers, CV-tested 2026-10-03)
Sources: BAF account-opening paper ("Objective over Architecture", Sun et al. 2025), a logistic-regression fraud paper (bank card / online payment case studies), and an IEEE-CIS feature-engineering paper.

**Tested on our data** (5×3 repeated CV, paired against the final model, PR-AUC 0.2413; the "Design choices tested" table in `fraud_logistic.ipynb` reproduces it):
| Idea (source) | AP Δ | Verdict |
|---|---|---|
| Balanced class weights (all papers) | −0.022 at C=0.3, −0.016 at C=0.01; Brier 0.015 → 0.16 | rejected. Recall comes from the threshold instead. |
| 1:5 cost weight (IEEE cost-sensitive) | −0.0035 to −0.017 | rejected |
| L1 / Lasso (LR paper) | C=0.3 −0.003, C=0.1 −0.004 | rejected; keep L2 |
| L2 C by CV (LR paper) | C=0.1 +0.0008 (noise), C=1 −0.0018 | C=0.3 kept |
| Missing indicators (IEEE) | −0.0012 | rejected. Fisher p ≥ 0.17 per column; missing merchant/device have 0 frauds; test has 3× more missing. |
| Target encoding / WoE for categoricals (IEEE) | −0.0010 | rejected |
| Binned WoE scorecard | −0.016 to −0.020 | rejected |
| Interactions amt×device, burst×amt, high24×risky (LR/IEEE) | −0.0004 / −0.0026 / −0.0020 | rejected |
| Splines on amount/age/velocity (LR paper's "nonlinear" limitation) | −0.005 to −0.019 | rejected; the linear form on log/flag features is adequate |
| Outlier clipping at P99 (LR paper boxplot removal) | −0.0008 | rejected; frauds sit in the amount tail |
| Drop weak `log_spend` (LR paper feature selection) | +0.0010 (SE 0.0005) | **applied** |
| Original 24-feature spec + missing flags | −0.0041 | the simpler model wins |

**Applied as practice (evaluation and operating point, mostly from the BAF paper):**
- Operating point from the full PR curve, not 0.5.
- Fraud-class metrics only: no accuracy or weighted F1, which BAF §3.2.1 shows are misleading.
- MCC, FP-per-TP workload, alert-budget table, bootstrap CI.
- In-fold preprocessing (IEEE).
- MI + chi-square screening table (LR paper §3.1.4) as documentation.

**Why the papers' headline numbers don't transfer:**
- BAF's recall jump (78% at 5% precision) comes from class weighting moving the operating point. For PR-AUC (ranking), weighting doesn't help and here it hurts.
- The LR paper's F1 0.87–0.89 is on different, much richer bank data, using SMOTE and mean imputation. It isn't comparable.

**Not applicable here:** chronological splits, rolling/card-level aggregates, device/email/network features (no timestamp, account id or identity fields). SMOTE is excluded by project rule (calibration).

**Untested next ideas** (try only if needed; keep only if > 2 SE better):
- Frequency encoding of categoricals.
- Adversarial-validation importance weights for the train→test shift.
- Choosing the flag share under a higher assumed test prevalence, since test has more new devices and larger amounts.
- Nested CV for an unbiased F1 estimate.
- 2026-10-04: **Monotone CatBoost tried and rejected** (`src/fraud_catboost_monotone.ipynb`, kept as a negative result). Constraints: ↑ amount, spend, 24h/1h counts, new_device, night, night×device; ↓ account age; raw numerics + hour sin/cos + 3 categoricals, no step flags or ratios. CV 5×2 + S1–S5 stress: standalone +0.006 clean / +0.008 S5 vs unconstrained (within noise); 40/40/20 CatBoost + monotone + logistic blend +0.003 clean / **+0.007 ± 0.003 S5** vs the current 80/20 blend. The notebook's own CV (seed 0, 2 seeds): monotone 0.2280 < CatBoost 0.2353; blend 0.2399 vs 0.2391. CatBoost enforces the constraints only approximately (largest dip 0.002 on spend after missing-value augmentation). **Public: monotone alone ~0.15, blend ~0.17** (vs ~0.19), a drop beyond the noise. **Lesson: the stress suite does not predict the real test shift.** It favoured both this and the round-2 test-like weight choice, and both lost on public. Don't use it alone to select models. Forcing new-device/burst/amount to always raise risk amplifies the flags that misfire on test (monotone test top 2%: 65% bursts vs 60% unconstrained).
- 2026-10-04: **Four new notebooks** (`src/nb1_catboost_rf_logistic`, `nb2_catboost_optimised`, `nb3_shift_aware_features`, `nb4_offline_tabpfn_like`), all run locally; outputs + upload plan in `results/` (README.md). Finding first: adversarial "test-like" validation (PR-AUC on the 30% most test-like training rows) does **not** reproduce the public ranking. Test-like training rows are *richer* in fraud (125 of 353 frauds in the top 20%), yet the public test scores lower → concept shift: extra flagged test rows are mostly legitimate. CV (seed 0) / test-like: CatBoost 0.2384/0.403; **NB2 top-3 CatBoost config average 0.2436/0.415** (current params remain best single; one-hot, rsm, depthwise worse); **NB3 CatBoost + within-dataset context features (profile frequency, isolation-forest rarity) 0.2400/0.415**, plus a label-free **flag-excess correction** (odds × train share / test share per new_device×burst×high24 cell, shrunk, capped; CV can't score it); NB1 RF 0.207, CatBoost+RF blends ≈ CatBoost; NB4 kNN 0.179, bagged MLP 0.183, blends hurt. Suggested uploads: NB3 context+excess, NB2 top-3, NB3 catboost+excess (isolates the correction vs 0.1925), NB1 catboost+rf 60/40.
- 2026-10-04: **Public results, plain CatBoost variants:** 3-seed mapped (`results/catboost/submission_catboost.csv`) **0.19586 (best so far)**; same config 1 seed as raw probabilities 0.19189 (vs 0.1925 mapped earlier) → **the checker is ranking-based PR-AUC; output scaling doesn't matter**, so post-processing can't move PR-AUC. 10 seeds mapped 0.19211. Spread 0.192–0.196 across seed counts is seed noise (more seeds didn't help), so 0.19586 is partly a lucky seed draw. Also tried and rejected today on CV + label-free test check (not uploaded): NB6 context-only (CV blend 0.206) and hybrid (0.214) vs plain 0.239, decoupled CatBoost + relative logistic (90/10 +0.0007, top-2% new-device share rose to 84%), multi-seed (no CV gain). Gemini plan (adversarial CV, Bayesian TE, heavy regularisation, PR-AUC post-processing) = repeats of tested ideas or invalid.
- 2026-10-04: **P1 (test rows as weight-0.3 "not fraud" examples, `src/nb7_p1_test_as_negatives.ipynb`): public 0.18778** vs 0.19586 for the identical config without it (same 3 seeds). CV had predicted the cost (−0.0095 ± 0.008). Rejected. Note: here CV and public agreed on direction and size.
- 2026-10-04: **Addition 1** (`src/best_catboost-addition1.ipynb`: best CatBoost unchanged; test amount + 24h spend quantile-mapped per non-SG country onto that country's training distribution, SG untouched): **public 0.19544** vs 0.19586, no effect. **Addition 2** (`src/best_catboost-addition2.ipynb`): same plus 1h/24h counts (random tie-break, inverted-CDF). Per-country burst rates back to training levels (ID 20% → 7.8%, MY 13.8% → 5.3%); test top 2% non-SG 89% → 78%, AU/JP/MY 31% → 21%, bursts 62.5% → 55.8%, new device 77.8% → 80.8% (edge of the +3 pt limit); 51/240 top rows replaced. File: `results/catboost_addition2/submission_best_catboost_addition2.csv`. **Public 0.1844 (worse, −0.0115).** The extra non-SG bursts carry real fraud signal; the shift-correction line is falsified.
