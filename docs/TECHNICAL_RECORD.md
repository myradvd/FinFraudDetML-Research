# Main handoff: everything needed to write the technical report

**Project:** Datathon Track 2, credit-card fraud detection (binary classification).
**Final model:** CatBoost, `notebooks/01_final_catboost.ipynb`. **Public leaderboard PR-AUC 0.19586** (team best). Out-of-fold CV PR-AUC 0.2384.
**Written:** 2026-10-04, from the full project record:
- `docs/research_log/decision_log.md`: decision log;
- `docs/research_log/research_brief_2026-10-04.md`;
- `docs/eda/EDA_report.md` and `docs/eda/Extended_EDA_Report.md`;
- every experiment notebook (now in `notebooks/`) and every leaderboard upload.

The final model's metrics in §9 were recomputed for this document.

> **How to use this document.** The report is 40% of the grade and is judged on three things: (1) understanding of the problem, (2) *why this model*, (3) the model's assumptions.
> - §1–§5: problem understanding.
> - §6–§9: why this model, with evidence.
> - §10–§11: assumptions and limitations.
> - §12: what didn't work. Negative results are strong evidence of rigour.
> - §13: a suggested report outline mapping each section to the material here.
> - Every number here is measured. Where something is uncertain or unrecorded, it says so.

> **Repository note (2026-10-05).** The repo was later trimmed to three notebooks (`notebooks/01`–`03`) plus the final notebook. Notebook names below that no longer exist refer to experiments whose results are kept here and in `results/leaderboard.csv`.

---

## Contents
1. Problem framing
2. Data description and data-quality audit
3. Exploratory data analysis: all findings
4. From EDA to design decisions (traceability table)
5. The train → test distribution shift
6. Evaluation methodology
7. Modelling journey: chronological, with every result
8. The final model: complete specification and the reason for every choice
9. Final results
10. Assumptions
11. Limitations and threats to validity
12. What did not work, and what we learned
13. Suggested report structure and key messages
14. Reproducibility appendix
15. Glossary

---

## 1. Problem framing

### 1.1 Task
- **Goal:** predict, for each card transaction, the probability that it is fraudulent (`fraud` = 1).
- **Training data:** 20,000 labelled transactions. **Test data:** 12,000 unlabelled transactions.
- **Submission:** `id,prediction`, one row per test transaction in the order of `format/sample_submission.csv`. `prediction` is a probability in [0, 1] (confirmed by the organisers 2026-10-03; the sample file's integer zeros are only placeholders).

### 1.2 Scoring
| Component | Weight | Notes |
|---|---|---|
| Results | 60% | **PR-AUC, F1 and Recall** (confirmed by the organisers 2026-10-03; an earlier RMSE/MAE/R² rubric text was wrong for this track) |
| Technical report | 40% | Understanding of the problem, why this model, the model's assumptions |

- **Public leaderboard:** each upload is scored by PR-AUC on the 12,000-row test set.
- **Final ranking:** a **private leaderboard on unseen data** with an unknown relationship to the public test set. The final notebook will be run on a **new test file**.

### 1.3 Why PR-AUC (and not accuracy or ROC-AUC)
- **Fraud is rare:** 353 of 20,000 = **1.765%**, about 55.7 legitimate transactions per fraud.
- **Accuracy is meaningless here:** predicting "never fraud" scores **98.235%**.
- **PR-AUC (average precision) measures how well frauds are ranked above legitimate transactions,** with a focus on the top of the ranking, where an investigator's alert budget lives. **A random ranker scores ≈ the base rate, 0.018.**
- **ROC-AUC is reported only as a secondary metric:** with this imbalance it is dominated by the huge number of easy negatives.
- **We verified on the leaderboard that the checker's PR-AUC depends only on the *ranking*:**

  | Same model (seed 0) | Public PR-AUC |
  |---|---|
  | Rescaled scores | 0.1925 |
  | Raw probabilities | 0.1919 |

  Calibration and thresholds therefore cannot change PR-AUC.

### 1.4 F1 and Recall: the threshold question
- **F1/Recall need 0/1 decisions.** How the organisers threshold the probabilities is **unknown**; we assume the common default of 0.5.
- **Unweighted, calibrated probabilities at a 1.8% base rate rarely exceed 0.5,** so a naive submission would flag almost nothing: recall ≈ 0.
- **Our solution:**
  1. choose the share of transactions to flag from the out-of-fold F1 curve (top 1.65%);
  2. **map the scores monotonically** so exactly that share sits at ≥ 0.5 (flagged rows → [0.51, 1], others → [0, 0.49]).
- The ordering is untouched, so PR-AUC is unchanged, while F1/Recall at a 0.5 threshold match the CV-optimal operating point.
- This was also a bug fix. An early TabPFN notebook submitted **hard 0/1 labels**, which ruins PR-AUC. It was corrected to continuous scores on 2026-10-03.

### 1.5 Why this is hard
1. **Extreme imbalance:** only 353 frauds to learn from. Every estimate is noisy (§6.4).
2. **No timestamp and no account/customer id:** no sequence features, no time-based validation, no per-account history beyond the provided 1h/24h aggregates.
3. **A substantial train → test distribution shift** (§5), including evidence that **the meaning of the risk signals changes** (concept shift). That is the central difficulty of this dataset.

---

## 2. Data description and data-quality audit

### 2.1 Schema
| Column | Type | Range / levels | Missing (train) |
|---|---|---|---|
| `id` | identifier | unique; train 1–20000, test 20001–32000 | 0 |
| `transaction_amount` | float | 2.5 – 9,000 | 0 |
| `transaction_hour` | int | 0–23 | 0 |
| `merchant_category` | categorical | 11 levels | 113 (0.57%) |
| `country` | categorical | 10 levels (SG, MY, ID, TH, US, AU, PH, VN, GB, JP) | 107 (0.54%) |
| `transaction_channel` | categorical | card_present, ecommerce, mobile_app, bank_transfer | 123 (0.62%) |
| `transactions_last_24h` | count | 1–27 | 159 (0.80%) |
| `spend_last_24h` | float | 0 – 15,670 | 132 (0.66%) |
| `account_age` | int | 5–4,000 (capped at 4,000, likely days) | 137 (0.69%) |
| `new_device` | 0/1 | | 124 (0.62%) |
| `transactions_last_1h` | count | 0–9 | 142 (0.71%) |
| `fraud` | target 0/1 | 353 positives | 0 |

### 2.2 Data-quality checks (all passed)
- **Integrity:** unique ids; no exact duplicate rows; no duplicate feature rows with conflicting labels; no duplicates between train and test.
- **Valid values:** binary labels; positive amounts; integer hours 0–23; non-negative integer counts; observed 1h count never exceeds the 24h count.
- **No ordering signal:** ids are sequential, but row order vs id has Spearman 0.0038, and Spearman(id, fraud) ≈ 0.001. All statistics are flat across id quintiles, so there is **no drift inside the training set**. The id carries no signal and is excluded from all models.
- **No hidden repeat accounts** (checked 2026-10-04): rows sharing account age × country collide at exactly the rate expected by chance (75% of rows in groups, both real and with shuffled country). Account-level features can't be reconstructed.

### 2.3 Semantic ambiguities (state them in the report)
- **`spend_last_24h` vs amount:**
  - the amount exceeds the 24h spend in **3,904 rows**;
  - **557 rows have zero spend**, while the 24h count is ≥ 1 (552 observed);
  - so spend does **not** consistently include the current transaction on the same scale. It may exclude it, use a different window, or contain placeholder zeros.
  - **Decision:** use the values as they are; don't over-interpret.
- **`account_age`:** integers, minimum 5, capped at 4,000 (354 rows at the cap). Compatible with days, but the unit can't be verified. Treated as an ordinal numeric field.
- **Currency:** country-level amount distributions differ (AU/JP have much higher means), but currency or exchange rates can't be identified from the data.
- **Feature availability at scoring time** (whether the 1h/24h windows include the current transaction or any future information) can't be verified. This is a leakage caveat for the report.

### 2.4 Missing values
- **Train:** 0.5–0.8% missing per column; 4.9% of rows have at least one blank (975 rows partially missing).
- **Test:** 2.0–2.4% per column; **13.9%** of rows: about 3× more.
- **Missingness doesn't predict fraud in training** (overall Fisher p = 0.454):
  - all 8 per-column tests have Benjamini–Hochberg q ≥ 0.43;
  - blank merchant: **0 frauds / 113**; blank device: **0 / 124**. These are small samples, not evidence of zero risk.
- **Decisions:**
  - no zero-filling (it would invent meaning);
  - categorical blanks → level `Unknown`;
  - numeric blanks → model-native handling (CatBoost) or an in-fold median (logistic);
  - explicit missing indicators were tested and rejected (−0.0012 AP).

---

## 3. Exploratory data analysis: all findings
All rates are on the 20,000 training rows. Lift is relative to the base rate of 1.765%. Wilson 95% intervals are in the EDA reports.

### 3.1 Amount
| Amount bucket | Rows | Frauds | Fraud rate | Lift |
|---|---|---|---|---|
| < 10 | 883 | 11 | 1.25% | 0.71× |
| 10–50 | 6,708 | 74 | 1.10% | 0.63× |
| 50–100 | 4,070 | 43 | 1.06% | 0.60× |
| 100–500 | 6,210 | 104 | 1.67% | 0.95× |
| 500–1,000 | 1,146 | 62 | **5.41%** | 3.07× |
| ≥ 1,000 | 983 | 59 | **6.00%** | 3.40× |

- **Fraud median amount 224 vs 74 for legitimate** (means 571 vs 247). Top amount decile: 5.6% fraud.
- **Risk is flat below ~500 and jumps above it** (non-linear), which motivates log transforms and trees.
- **Large amounts are not deterministic:** the largest legitimate amount (9,000) exceeds the largest fraud (5,447), so outliers were **kept**.
- **No round-amount signal:** integer amounts have 1 fraud in 198 rows; multiples of 10 have 0 in 24.

### 3.2 Velocity (transaction counts)
| 1h count | Rows | Fraud rate | Lift |
|---|---|---|---|
| 0 | 8,659 | 1.37% | 0.78× |
| 1 | 7,353 | 1.31% | 0.74× |
| 2–3 | 3,380 | 1.75% | 0.99× |
| 4–5 | 299 | **13.4%** | 7.6× |
| 6–10 | 167 | **21.0%** | 11.9× |

- **By exact count:** 5/h: 23%; 6/h: 30%; 7–9/h: 17–19% (risk peaks, then levels off).
- **≥ 4 in the last hour:** **16.1% fraud (75/466)**, 9.1× the base rate. It covers 2.3% of transactions and **21% of all frauds**.
- **24h count ≥ 11:** **19.7% fraud (73/370)**, 11.2×. Below 11, the fraud rate stays between 1.0% and 1.7%.
- **24h spend:** ≥ 2,500: 7.5%; ≤ 500: ~1.0%.
- **1h and 24h counts are related but distinct** (Spearman 0.39); amount vs spend 0.58. Neither was dropped for correlation.

### 3.3 Categorical variables
- **Merchant:**
  - highest: luxury 5.29%, cash_transfer 4.74%, electronics 3.94%, travel 2.38%;
  - lowest: grocery 0.76%, transport 0.88%, retail 0.96%;
  - bias-corrected Cramér's V 0.10.
- **Country:** ID 4.05%, VN 3.74%, PH 3.44%, JP 3.40%, US 3.17%, GB 3.17%, AU 2.91%, TH 1.71%, MY 1.47%, **SG 1.31%**.
  - SG has 70% of rows (13,931) but a below-average fraud rate.
  - Non-SG overall: 2.8% vs 1.3% in SG, and 48% of all frauds.
- **Channel:** bank_transfer 2.84%, ecommerce 2.40%, mobile_app 1.68%, **card_present 1.02%** (lowest).
- **New device: 7.50% (135/1,800) vs 1.21% for known devices**, 6.2× (Fisher p = 3e-51). Cramér's V 0.137: the strongest single signal.
- **Encoding:** low cardinality (11 / 10 / 4 levels), so native categorical handling or one-hot works; high-cardinality target encoding is unnecessary.

### 3.4 Time of day
- **Night (23:00–04:59): 2.67% vs 1.46%** at other hours (1.83×, p = 6e-8). Peak at 02:00 (3.36%); trough at 21:00 (0.48%).
- **Encoding:** hour is cyclic, so it's encoded as sine/cosine (23:00 and 00:00 are neighbours), plus a night flag.

### 3.5 Account age
- **Fraud median 335 days vs 796.**
- **≤ 30 days: 4.27% fraud (32/749)**, 2.6×.
- **Lowest age decile (≤ 72 days): 4.6%;** above ~230 days it's flat at ~1.1–1.5%.

### 3.6 Combinations (the key EDA insight)
| Combination | Rows | Fraud rate | Lift |
|---|---|---|---|
| Burst (1h ≥ 4) + high amount (≥ P95) | 60 | **40.0%** | 22.7× |
| 24h ≥ 11 + risky merchant | 121 | **39.7%** | 22.5× |
| High amount + new device | 82 | **37.8%** | 21.4× |
| Burst + new device | 144 | **37.5%** | 21.2× |
| 24h ≥ 11 + new device | 161 | 34.8% | 19.7× |
| Night + new device | 397 | 16.9% | 9.6× |
| High amount + young account | 33 | 21.2% | 12.0× |

**Number of fired flags** (out of 7: 1h ≥ 4, 24h ≥ 11, new device, amount ≥ P95, risky merchant, night, age ≤ 30):

| Flags fired | Fraud rate |
|---|---|
| 0 | 0.76% |
| 1 | 1.3% |
| 2 | 3.7% |
| 3 | 11% |
| 4 | 37% |
| 5 | 55% |

**Fraud concentrates where several signals coincide.** That is the core argument for a tree model that learns interactions, and for the `n_flags` and `night_x_device` features.

### 3.7 Silent fraud
- **77 frauds (21.8%) fire none of the 7 flags.** They look like legitimate transactions: median amount 69, no new device.
- **Design consequence:** keep the continuous and categorical information, not just binary rules. A rule-points score reached only **AP 0.184**.

### 3.8 Statistical tests (training folds, from the extended EDA)
- **Interactions** (likelihood-ratio tests, Holm-corrected within training folds):
  - **night × new device is significant in all 10 folds** (Holm p ≤ 0.013);
  - amount × device and burst × amount are not;
  - none of the three improves out-of-fold AP for logistic regression (all paired CIs include 0). So explicit interaction products were not prioritised for logistic regression, but `night_x_device` was kept.
- **Group ablations** (LightGBM, removal effect on AP, 95% CI):
  - **device −0.038 [−0.055, −0.018];**
  - **account age −0.019 [−0.032, −0.007];**
  - **categoricals −0.017 [−0.032, −0.004];**
  - amount −0.013 [−0.026, +0.001]; velocity −0.011 [−0.028, +0.008];
  - time −0.005 (CI includes 0); missingness indicators −0.002 (CI includes 0).
- **Learning curve** (AP by training fraction):

  | Training data | 20% | 40% | 60% | 80% | 100% |
  |---|---|---|---|---|---|
  | AP | 0.178 | 0.189 | 0.196 | 0.206 | 0.211 |

  Still rising, with no clear plateau: more data would likely help.
- **Class weighting:**
  - unweighted boosting: Brier score 0.0153; with balanced weights 0.0938 (badly calibrated);
  - AP: **weighted logistic 0.207 vs unweighted 0.227;** weighted boosting 0.204 vs unweighted 0.216.
- **Isolation-forest anomaly score as a feature:** +0.006 AP, CI [−0.003, +0.014]. Not adopted (anomalous ≠ fraudulent).
- **Peer/relative features in logistic regression** (amount relative to merchant or country median): no gain. Country-relative −0.0010, CI excludes 0.
- **EDA model baseline** (5-fold × 2 repeats):

  | Model | AP |
  |---|---|
  | Unweighted logistic | **0.227** [0.181, 0.280] |
  | LightGBM | 0.216 |
  | Weighted logistic | 0.207 |
  | Rule points | 0.184 |

---

## 4. From EDA to design decisions (traceability table)
| EDA finding (§3) | Design decision | Where |
|---|---|---|
| 1.77% fraud; accuracy meaningless | Optimise and report PR-AUC; F1/Recall at a chosen operating point; ROC-AUC secondary | Whole project |
| Class weighting lowered AP and destroyed calibration | **Train unweighted.** Recall comes from the threshold, not from weights | Final model |
| Amount risk flat to ~500, then a jump; heavy right tail | `log_amt`, `log_spend`; trees for non-linearity; **no outlier clipping** (frauds live in the tail; P99 clipping −0.0008) | Features |
| Velocity risk jumps at 1h ≥ 4 and 24h ≥ 11 | `burst` and `high24` flags alongside the raw counts | Features |
| Account age risk concentrated ≤ 30 days | `young` flag + `log_age` + raw age | Features |
| Night 23–04 riskier; hour is cyclic | `hour_sin`, `hour_cos`, `night` | Features |
| Luxury / cash transfer / electronics riskiest | `risky_merchant` flag + native merchant category | Features |
| New device the strongest single signal; night × device significant in all folds | `new_device` + `night_x_device` | Features |
| Fraud concentrates where several flags coincide | `n_flags` (count of 6 indicators) + **a tree model that learns interactions** | Features, model choice |
| 21.8% silent frauds | Keep the continuous and categorical inputs, not only rules | Features |
| Fraud = unusual *for that account* | `amt_to_avg_ticket` (amount vs the account's 24h average ticket), `share_1h` (1h share of 24h activity) | Features |
| Missingness not significantly related to fraud | No missing indicators; native NaN handling; `Unknown` category | Preprocessing |
| Low-cardinality categoricals | CatBoost native categoricals (ordered target statistics) | Model |
| No timestamp / no account id | Stratified K-fold CV; no time features beyond hour; no entity features | Validation |
| Learning curve still rising; 353 frauds | Strong regularisation (shallow trees, L2), seed averaging, simple models preferred | Model |
| Spend semantics inconsistent | Use as given; no derived "amount > spend" feature (1.82% vs 1.74%: no signal) | Features |

---

## 5. The train → test distribution shift (the central challenge)

### 5.1 What changed
| Statistic | Train | Test |
|---|---|---|
| Rows with any blank | 4.9% | **13.9%** |
| Per-column blanks | 0.5–0.8% | 2.0–2.4% |
| new_device rate | 9% | **15%** |
| 1h count ≥ 4 ("burst") | 2.3% | **4.4%** |
| Amount P90 / P95 | 530 / 978 | **1,707 / 3,580** |
| Amount mean | 253 | 524 |
| Median amount AU / JP / MY | 122 / 134 / 76 | **1,297 / 1,609 / 164** |
| Account age median | 790 | 1,158 (only *old* accounts added) |
| Country SG / AU / JP | 70% / 2.9% / 1.9% | 63% / 4.9% / 4.2% |
| Channel ecommerce / mobile_app | 27.7% / 26.0% | 20.8% / 30.5% |

- **Same categories:** no unseen levels, and ranges stay within training bounds (amount max 9,000; age max 4,000).
- **Adversarial validation:** a classifier separating train rows from test rows reaches **AUC 0.67**, a substantial shift.
- **The shift is a step change, not gradual:** all training statistics are flat across id order, and the test set (ids 20001+) jumps.

### 5.2 Where the shift is concentrated
- **New-device rate by merchant:**
  - utilities 10.7% → **34.8%**, retail 10.7% → 23.0%, transport 10.8% → 21.2%, grocery 8.7% → 17.1%, Unknown 9% → 22%;
  - luxury, electronics, cash_transfer and travel unchanged or down;
  - SG overall: 9% → 18%.
- **Bursts (1h ≥ 4) by merchant:** gaming 4.9% → 12.4%, food_delivery 2.6% → 7.5%, transport 2.2% → 5.3%, electronics 4.0% → 7.2%, luxury 3.8% → 6.5%.
- **Bursts by country:** ID 7.9% → 20.0%, MY 5.3% → 13.8%, GB 3.6% → 6.6%, TH 3.1% → 5.9%.
- **Amounts by country (test/train ratio at the 10/25/50/75/90% quantiles):**

  | Country | 10% | 25% | 50% | 75% | 90% |
  |---|---|---|---|---|---|
  | **SG (control)** | 1.02 | 1.01 | 0.98 | 0.87 | 1.03 |
  | AU | 2.2 | 3.7 | 10.6 | 7.3 | 1.7 |
  | JP | 2.1 | 3.8 | 12.0 | 5.0 | 1.4 |
  | MY | 1.6 | 1.5 | 2.1 | 8.1 | 5.0 |
  | Others (GB, ID, PH, TH, US, VN) | 1.1–1.5 | 1.2–1.8 | 1.1–2.1 | 1.2–1.7 | 1.2–2.3 |

  **Singapore is unchanged at every quantile, while every other country's amounts rose.** AU/JP is not a clean ×10 currency change: a new group of mid-to-large transactions appears.

### 5.3 Evidence that P(fraud | x) changed, not only P(x)
1. **Every model drops by about the same amount** from CV (~0.23–0.24) to the public test (~0.19). That points to a dataset effect, not a model-choice effect.
2. **A covariate-shift-weighted CV proxy predicted ~0.34** for the test set. Reweighting training rows to look like test rows *raises* the expected score, because the test-like training rows are fraud-rich. The real score was 0.19.
3. **The most test-like training rows contain *more* fraud:** 125 of 353 frauds sit in the 20% most test-like rows. Covariate-shift reasoning predicts a *higher* test score; the real one is lower.
4. **The test set has many more fraud-looking rows** (new devices, bursts, large amounts) **but scores lower.** The same signals are therefore fraud less often in the test set.
5. **The mean predicted fraud probability is higher on test (0.0221) than on training out-of-fold (0.0172):** the model sees more "risk" in test.
6. **Correcting the signals in either direction lost on the leaderboard (§7.11–7.13):**
   - treating the extra flagged test rows as legitimate (0.1878), or re-anchoring non-SG activity to training levels (0.1844): both hurt;
   - amplifying the flags (monotone constraints, ~0.15): also hurt.

   So the test set's flagged rows are a **mix** of legitimate and fraudulent transactions that these 10 columns can't separate. Plain CatBoost's ranking sits at a local optimum.

### 5.4 Where the test ranking concentrates (label-free diagnostics)
| Top 2% of the ranking | New device | 1h ≥ 4 | Non-SG | AU/JP/MY |
|---|---|---|---|---|
| Training frauds (reference) | 38% | 21% | 48% | 14% |
| Best model, training out-of-fold top 2% | 74% | 41% | 67.5% | 21% |
| **Best model, test top 2%** | **78%** | **62.5%** | **89%** | **31%** |

- **The new-device rows in the test top 2% are mostly at electronics (57), cash_transfer (32) and luxury (30):** merchants where training new-device fraud was 20–27%, and whose new-device rate didn't change.
- **The everyday-merchant new-device surge** (utilities, retail; training new-device fraud 2–3%) **is already ranked low by the model.**

---

## 6. Evaluation methodology

### 6.1 Cross-validation design
- **Stratified K-fold (5 folds)**, so each fold keeps the 1.77% fraud rate. Main experiments used 5×2 or 5×3 repeated CV with fixed seeds.
- **Why not time-based or grouped CV:** there is no timestamp and no account id. Random stratified CV is the only valid option. **Hour and id must not be used as an invented chronology.**
- **All preprocessing is fitted inside the training folds:**
  - imputation medians, scalers, encoders, thresholds;
  - CatBoost's own categorical statistics, missing-value augmentation, blend weights.
- **Out-of-fold predictions:** every training row is scored once, by a model that never saw it. These feed PR-AUC, the threshold choice and the calibration checks.

### 6.2 Metrics
- **Primary:** average precision (scikit-learn `average_precision_score`; a step-wise estimate, not a trapezoid).
- **Also reported:** ROC-AUC; Brier score (calibration); precision / recall / F1 / MCC at the chosen operating point; precision and recall at fixed alert budgets (0.5%, 1%, 2%, 5% of transactions); false positives per true positive (investigator workload).
- **Never:** accuracy or weighted F1.

### 6.3 Uncertainty
- **Bootstrap 95% CI for out-of-fold AP:** outcome-stratified, with fixed predictions.
- **Paired comparisons:** variant minus baseline on the same folds, with the SE across folds and the number of folds won. **A change was kept only if it beat the noise (> 2 SE).**
- **Typical paired SE on CV:** 0.002–0.004.

### 6.4 How noisy the leaderboard is (important for interpreting results)
- **SD of AP on a test-sized sample** (12,000 rows, ~212 frauds), bootstrapped from the final model's out-of-fold predictions: **0.030**. Paired difference between two similar models on the same rows: SD ≈ **0.012**.
- **Seed noise measured on the leaderboard:** the same CatBoost with 1, 3 and 10 seeds scored 0.1925, **0.19586** and 0.1921. So **±0.004 is noise**.
- **Consequence:** public differences below ~0.01 are not evidence. The team avoided picking "the luckiest upload" (winner's curse), and 0.19586 should be read as a ~0.193 model.

### 6.5 The CV-to-leaderboard gap, and the search for a better proxy
- **CV over-predicts the public score by ~0.045, uniformly across model families:**

  | Model | CV | Public |
  |---|---|---|
  | CatBoost | 0.238 | 0.196 |
  | LightGBM | 0.217 | 0.190 |
  | Logistic blends | 0.24 | 0.183 |

- **Proxies built to predict the leaderboard better, all of which failed:**
  1. **Adversarial "test-like" validation** (PR-AUC on the 30% most test-like training rows): did not reproduce the public order. It rated the monotone model equal to or better than CatBoost; on the leaderboard monotone scored 0.15 vs 0.19.
  2. **Covariate-shift importance weighting:** predicted ~0.34; reality 0.19.
  3. **A synthetic stress-test suite:** label-blind perturbations of the validation folds:
     - S1: test-like amount/spend/age scales;
     - S2: 3× blanks;
     - S3: extra new devices and bursts without fraud;
     - S4: 10% category noise;
     - S5: all together.

     S5 reproduced the size of the public drop (~0.03), but the suite **mis-ranked** models: it favoured two changes that then lost on public.
- **What CV *does* get right:** the **sign of changes that hurt.** For example, P1 (test rows as "not fraud") lost −0.0095 on CV and −0.008 on public. The context features lost −0.03 on CV and were rejected without upload.
- **Final validation rule:**
  - use CV to choose *within* a design (settings, features) and to reject harmful changes;
  - use the leaderboard (sparingly, one hypothesis per upload) to choose *between* designs.

---

## 7. Modelling journey: chronological, with every result
CV figures are out-of-fold PR-AUC. "Public" is the leaderboard PR-AUC.

### 7.1 Phase 1: interpretable baseline, regularised logistic regression (2026-10-03)
- **Model:** L2 logistic regression, C = 0.3, **unweighted**. Median imputation, standardisation and one-hot categoricals, all fitted in-fold.
- **Final logistic spec:**
  - 8 numeric features: `transactions_last_24h`, `new_device`, `log_amt`, `log_age`, `night`, `burst`, `high24`, `night_x_device`;
  - plus the 3 one-hot categoricals.
- **Results** (5×2 CV): **OOF PR-AUC 0.233**, bootstrap CI [0.183, 0.279]. At the F1-optimal top 1.18%: F1 0.296, recall 0.246, precision 0.370, MCC 0.292, 1.7 false positives per true positive.
- **Assumptions checked:** max VIF 2.1 (no multicollinearity problem). An odds-ratio table was produced.
- **Design choices tested, from published fraud-modelling papers** (5×3 CV, paired against 0.2413):

| Idea (source) | AP change | Verdict |
|---|---|---|
| Balanced class weights (all papers) | −0.022 (C = 0.3), −0.016 (C = 0.01); Brier 0.015 → 0.16 | Rejected |
| 1:5 cost weight (IEEE cost-sensitive) | −0.0035 to −0.017 | Rejected |
| L1 / Lasso | −0.003 (C = 0.3), −0.004 (C = 0.1) | Rejected; keep L2 |
| C chosen by CV | C = 0.1: +0.0008 (noise); C = 1: −0.0018 | C = 0.3 kept |
| Missing-value indicators | −0.0012 | Rejected |
| Target / WoE encoding | −0.0010 | Rejected |
| Binned WoE scorecard | −0.016 to −0.020 | Rejected |
| Interactions amount×device, burst×amount, high24×risky merchant | −0.0004 / −0.0026 / −0.0020 | Rejected |
| Splines on amount / age / velocity | −0.005 to −0.019 | Rejected |
| P99 outlier clipping | −0.0008 | Rejected (frauds live in the tail) |
| Drop `log_spend` | **+0.0010** (SE 0.0005) | **Applied** |
| Original 24 features + missing flags | −0.0041 | The simpler model wins |

- **Practices adopted from the papers:** an operating point taken from the full PR curve (not 0.5); fraud-class metrics only (no accuracy); MCC; false positives per true positive; an alert-budget table; bootstrap CIs; in-fold preprocessing; mutual-information and χ² screening as documentation.
- **Why the papers' headline numbers don't transfer:**
  - the BAF paper's high recall comes from class weighting moving the operating point, which doesn't help a ranking metric;
  - the logistic-regression paper's F1 0.87–0.89 is on much richer data, with SMOTE.
- **SMOTE was excluded on principle:** synthetic minority samples distort calibration and add no new information about P(fraud | x).

### 7.2 Phase 2: blends with TabPFN, XGBoost and LightGBM (2026-10-03)
- **XGBoost with the exact `scale_pos_weight` = negatives/positives:** 0.183. Unweighted 0.204; √ratio 0.196. Blending it with logistic regression *lowered* logistic's score. **Confirms: don't weight.**
- **LightGBM by `scale_pos_weight`:** unweighted 0.210 > 3 (0.206) > √ratio (0.200) > exact ratio (0.184).
- **LightGBM tuning:**
  - random search over 70 configurations, 5×2 CV, top 5 re-scored on fresh splits;
  - winner: 900 trees, lr 0.03, **4 leaves**, min_child_samples 150, colsample 0.85, L2 10, L1 0.5, min_split_gain 0.01, unweighted;
  - LightGBM **0.207 → 0.219**; logistic + LightGBM blend 0.220 → 0.228 (over 3 split seeds);
  - **lesson:** tiny trees with large leaves win on 353 frauds.
- **Round 2:**
  - the blend's logistic uses the lean 8 features: +0.0038 (better on 3/3 seeds);
  - rejected: 5-seed LightGBM bagging (+0.0001), logistic C 0.1/1 (noise), LightGBM subsample 0.8 or raw-only features (−0.007), learned blend weights (no gain on held-out halves).
- **Original TabPFN + LightGBM + logistic rank blend:** public **0.19354** (best until 2026-10-04).

### 7.3 Phase 3: CatBoost (2026-10-03)
- **CatBoost** (depth 4, 800 iterations, lr 0.03, L2 10, native categoricals) **as a 4th blend member:** blend +0.0052 (3/3 seeds).
  - **Standalone:** CV 0.2404, F1 0.308 at the top 1.65%.
  - **Rejected in the same round:** frequency encoding (+0.0006), LightGBM focal loss (γ = 1 noise, γ = 2 broke), covariate-shift importance weighting (−0.004 to −0.005).
- **CatBoost hyperparameter random search:** 20 of 30 configurations evaluated (stopped early); none beat depth 4 / 800 / lr 0.03 / L2 10.
- **Leaderboard round:**

  | Submission | Public PR-AUC |
  |---|---|
  | CatBoost | 0.19248 |
  | LightGBM | 0.19006 |
  | CatBoost + TabPFN | 0.18804 |
  | TabPFN | 0.18750 |
  | 4-model blend with CatBoost | worse than 0.1935 (exact number not recorded) |

  All within ~0.006: below the noise.
- **`fraud_catboost_logistic.ipynb`** (5-fold, 3 split seeds):

| Model | Split 0 | Split 1 | Split 2 | Mean |
|---|---|---|---|---|
| Logistic | 0.2280 | 0.2321 | 0.2284 | 0.2295 |
| CatBoost, 1 seed | 0.2404 | 0.2238 | 0.2196 | 0.2279 |
| CatBoost, 5-seed average | 0.2410 | 0.2277 | 0.2211 | 0.2299 |
| **Blend 50/50** | 0.2405 | 0.2388 | 0.2348 | **0.2380** |
| Blend 60/40 / 70/30 | | | | 0.2375 / 0.2370 |

  Seed averaging gave +0.002 (3/3). The blend gave +0.008 over either model (3/3) on CV. **But on the public board, CatBoost + logistic 50/50 scored 0.18320,** *below* CatBoost alone.

### 7.4 Robustness work: the stress suite and the shift notebooks (2026-10-03)
- **Stress-suite robustness ranking** (mean S1–S5 / worst scenario):

  | Model | Mean | Worst |
  |---|---|---|
  | logistic + LightGBM + CatBoost | 0.2251 | 0.2053 |
  | logistic + CatBoost | 0.2247 | 0.2023 |
  | logistic + LightGBM | 0.2216 | 0.2045 |
  | LightGBM + CatBoost | 0.2177 | – |
  | logistic | 0.2176 | – |
  | CatBoost | 0.2123 | – |
  | LightGBM | 0.2081 | – |

  Logistic regression was unaffected by scale shifts (S1). **This ranking turned out *not* to match the leaderboard.**
- **Diagnosis of why blends lost on public:**
  - logistic regression gives every flagged row a *fixed* bonus: 72% of its test top 2% are bursts, against 21% of training frauds (CatBoost: 62%);
  - so blending it in pulls the ranking toward the misfiring signals;
  - the large AU/JP amounts were not the cause (those rows already rank low; confirmed again on 2026-10-04, §7.13).
- **Shift notebook, round 2** (test-like perturbed validation, CatBoost 1 seed):

| Change | Clean CV | Test-like CV | Verdict |
|---|---|---|---|
| Logistic: log counts instead of burst/high24 flags | 0.2278 → 0.2303 | – | Kept (for logistic) |
| CatBoost **without** missing-value augmentation | 0.2322 → **0.2404** | 0.2128 → 0.2191 | **No augmentation for CatBoost** |
| Logistic **with** missing-value augmentation | 0.2300 → 0.2303 | 0.2067 → 0.2082 | Kept (for logistic) |
| Velocity counts capped at the training P99 | 0.2322 → 0.2259 | 0.2128 → 0.2031 | Removed |
| Logistic new-device context terms | −0.007 | −0.005 | Rejected |
| CatBoost without the flag features | −0.009 | −0.009 | Rejected (the flags help CatBoost) |
| CatBoost depth 3 | 0.2404 → 0.2348 | 0.2191 → 0.2158 | Rejected |

  The resulting shift notebooks (rounds 1–2) **scored worse on public** (exact numbers not recorded).

### 7.5 Session of 2026-10-04: contextual (relative) features
Paired against the current notebook, 5×2 CV, 2 CatBoost seeds:

| Feature group | Clean CatBoost change | Stress change |
|---|---|---|
| Velocity percentile within merchant / merchant×country×channel / country | −0.006 | – |
| Amount relative to merchant/country, training statistics | −0.000 | −0.012 (amounts ×2) |
| Same, statistics from the scored set itself | −0.001 | −0.014 |
| Account-age percentile within country/merchant/channel | +0.000 | +0.002 |

- **All rejected.** Velocity distributions barely differ between merchants in training (median 1 per hour almost everywhere), so these features duplicate the raw ones.

### 7.6 Monotone constraints (`07_monotone_catboost.ipynb`)
- **Constraints:**
  - increasing: amount, spend, 1h/24h counts, new_device, night, night×device;
  - decreasing: account age;
  - free: hour and the categoricals.
- **CV:** standalone +0.006, within noise. The CatBoost + monotone + logistic blend: +0.007 ± 0.003 on stress scenario S5.
- **CatBoost enforces the constraints only approximately:** the largest violating dip was 0.002.
- **Public: monotone alone ~0.15; 40/40/20 blend ~0.17.** A clear loss: forcing "more signal = more risk" in every context amplifies the flags that misfire on test.

### 7.7 Four alternative notebooks (2026-10-04)
| Notebook | Idea | CV | Public |
|---|---|---|---|
| NB1 | CatBoost + random forest (600 trees, min leaf 25, max_features 0.35) + logistic | RF 0.207; blends 0.237–0.238 (CatBoost 0.238) | RF blends < 0.19 |
| NB2 | CatBoost configuration search (8 configs: one-hot categoricals, rsm 0.5, random_strength 3, border_count 32 + L2 20, depth 5 / lr 0.02 / 1200, Depthwise depth 5, combinations) | **Baseline best single config (0.2396)**; top-3 config average 0.2436 | Top-3 average < 0.19 |
| NB3 | Within-dataset context features (profile frequency, isolation-forest rarity) + "flag-excess" correction (test odds × train/test share per new_device×burst×high24 cell) | +0.0016 | **0.18930** |
| NB4 | Offline TabPFN-like: in-context kNN (k = 150, logistic-weighted metric) and 10 bagged MLPs | kNN 0.179, MLP 0.183; blends hurt | Not uploaded |

NB2 search detail (1 seed, CV):

| Config | CV |
|---|---|
| baseline | **0.2396** |
| slow_deep | 0.2393 |
| onehot + coarse + random strength | 0.2388 |
| coarse | 0.2378 |
| onehot + rsm | 0.2369 |
| onehot | 0.2320 |
| rsm + random strength | 0.2315 |
| depthwise | 0.2218 |

**This is the strongest evidence that the final settings are well tuned.**

### 7.8 Leaderboard probes (NB5): one change each to CatBoost (CV, 1 seed)
| Probe | CV |
|---|---|
| Raw 10 features only | 0.222 |
| CatBoost default settings | 0.213 |
| No `new_device` | 0.168 (its test top 2% was still 51% new-device rows) |
| No velocity flags | 0.230 |
| SqrtBalanced class weights | 0.198 |

**Every simplification or reweighting lost on CV,** which supports the 24-feature, unweighted design.

### 7.9 Context-relative CatBoost + logistic (NB6), 2026-10-04
- **Features:** every shift-prone feature re-expressed relative to its merchant or country baseline, **computed within each dataset** (test rows against test baselines):
  - merchant-relative counts;
  - new-device "surprise" per merchant × channel;
  - amount vs merchant × country median;
  - age percentile within merchant;
  - Laplace-smoothed hour rarity per merchant;
  - an out-of-fold context-risk score with per-merchant slopes;
  - no hard-coded thresholds.
- **Results:**

  | Variant | CV blend | Plain CatBoost |
  |---|---|---|
  | Context-only | 0.206 | 0.239 |
  | Hybrid (raw + context) | 0.214 | 0.239 |

  **The test top 2% was still ≥ 79% new device.** Rejected without upload.
- **A decoupled ensemble** (plain CatBoost + logistic on the relative features only) didn't help either: 90/10 +0.0007. The relative logistic's top 2% was **94%** new device.

### 7.10 Multi-seed averaging, 2026-10-04
- **CV, vs 1 seed:** 5 seeds −0.0006, 10 seeds −0.0021. No gain.
- **Public:** 10 seeds 0.19211, **3 seeds 0.19586**, 1 seed 0.1925. Noise.

### 7.11 Test rows as weak "not fraud" examples (P1, NB7)
- **Idea:** the 12,000 test rows added as weight-0.3 negatives, labels assumed "not fraud" because fraud is rare.
- **Results:** CV −0.0095 ± 0.008. **Public 0.18778** (vs 0.19586 for the identical configuration without it). CV predicted the loss correctly.

### 7.12 Metric probe
- **Same CatBoost, seed 0, raw probabilities:** public 0.19189 vs 0.1925 rescaled. The checker is ranking-based PR-AUC, so post-processing can't help.

### 7.13 Re-anchoring non-SG test inputs to their market's training scale (2026-10-04)
- **Addition 1:** amounts + 24h spend, quantile-mapped per non-SG country; model unchanged. Test top 2%: non-SG 89% → 84%, AU/JP/MY 31% → 23%. **Public 0.19544:** no effect.
- **Addition 2:** also the 1h/24h counts. Per-country burst rates back to training levels (ID 20% → 7.8%). Top 2%: non-SG → 78%, bursts → 56%. **Public 0.1844:** worse. **The extra non-SG bursts in test carry real fraud.**

### 7.14 Complete leaderboard history
| # | Submission | Public PR-AUC |
|---|---|---|
| 1 | **Plain CatBoost, 3 seeds (final)** | **0.19586** |
| 2 | Addition 1: re-anchored non-SG amounts | 0.19544 |
| 3 | TabPFN + LightGBM + logistic rank blend | 0.19354 |
| 4 | Plain CatBoost, 1 seed | 0.19248 |
| 5 | Plain CatBoost, 10 seeds | 0.19211 |
| 6 | Plain CatBoost, 1 seed, raw probabilities (metric probe) | 0.19189 |
| 7 | LightGBM | 0.19006 |
| 8 | CatBoost + context features + flag-excess correction (NB3) | 0.18930 |
| 9 | CatBoost + TabPFN | 0.18804 |
| 10 | CatBoost + test rows as negatives (P1) | 0.18778 |
| 11 | TabPFN | 0.18750 |
| 12 | Addition 2: re-anchored non-SG amounts + counts | 0.18440 |
| 13 | CatBoost + logistic 50/50 | 0.18320 |
| 14 | CatBoost + monotone CatBoost + logistic 40/40/20 | ~0.17 |
| 15 | Monotone CatBoost | ~0.15 |
| – | 4-model blend with CatBoost; shift notebooks rounds 1–2; NB1/NB2 candidates (RF blends, top-3 config average) | all < 0.19 (exact numbers not recorded) |

**Leaderboard context (snapshot 2026-10-04):** 1st 0.2508, 3rd 0.2417, 11 teams ≥ 0.217. The team was 23rd at 0.1935 before the 0.19586 upload.

---

## 8. The final model: complete specification and the reason for every choice
**Notebook:** `notebooks/01_final_catboost.ipynb`, a cleaned-up copy of the CatBoost part of `notebooks/04_catboost_rf_logistic_blends.ipynb`, which produced the uploaded file.

### 8.1 Why CatBoost (the model-choice argument)
1. **It holds up best on the real test set:** best public score of every family tried (CatBoost 0.1925–0.1959 vs LightGBM 0.1901, TabPFN 0.1875, RF and logistic blends ≤ 0.1893, monotone 0.15). The most important criterion, given the shift.
2. **It learns combinations of signals,** which is where fraud concentrates (pairs reach 38–40% fraud; risk rises sharply with the number of fired flags). A linear model needs those products specified by hand, and gives each flag a fixed bonus that misfires on the shifted test set.
3. **Native, leakage-safe categorical handling:** ordered target statistics (each row's category statistic is computed only from rows before it in a random permutation). No manual target encoding with its overfitting risk.
4. **Native missing-value handling:** blanks go to their own branch, with no invented imputed values.
5. **Strong regularisation for small positive counts:** symmetric (oblivious) trees, shallow depth and L2 leaf shrinkage suit 353 frauds.
6. **Simple and explainable enough for the report:** one model family, 24 named features, feature importances, no stacking.

### 8.2 Data cleaning (`sanitise`): robustness for unseen data
Changes nothing on the current files. Exists because the final ranking uses **unseen, possibly messier data**:
- **Columns:** selected by name (`reindex`); a missing column becomes all-blank; extra columns are ignored.
- **Numbers:** parsed defensively; text and ±infinity become blank.
- **Impossible values become blank:**
  - negative amounts, spend, counts or ages;
  - hours outside 0–23;
  - `new_device` values other than 0/1.
- **Consistency:** the 1h count is capped at the 24h count (logical consistency).
- **Categories:** spelling normalised to training (trim + lower-case lookup: `" sg"` → `SG`); unseen levels → blank → `Unknown`.
- **Blanks are never zero-filled.**

### 8.3 Features: 24, fixed order
| # | Feature | Definition | Why (EDA evidence) | Importance* |
|---|---|---|---|---|
| 1 | `transaction_amount` | raw | Fraud median 224 vs 74; risk ×3 above 500 | 8.69 |
| 2 | `transaction_hour` | raw 0–23 | Night risk | 1.64 |
| 3 | `merchant_category` | categorical (native) | 0.76%–5.29% by merchant | 2.59 |
| 4 | `country` | categorical (native) | 1.31% (SG)–4.05% (ID) | 3.81 |
| 5 | `transaction_channel` | categorical (native) | card_present 1.02% vs bank_transfer 2.84% | 6.36 |
| 6 | `transactions_last_24h` | raw | ≥ 11: 19.7% fraud | 2.92 |
| 7 | `spend_last_24h` | raw | ≥ 2,500: 7.5% | 5.12 |
| 8 | `account_age` | raw | Fraud median 335 vs 796 days | 6.43 |
| 9 | `new_device` | 0/1 | 7.5% vs 1.2%; strongest single signal | 6.33 |
| 10 | `transactions_last_1h` | raw | 5–6/h: 23–30% | 3.47 |
| 11 | `log_amt` | log1p(amount) | Heavy tail; evenly spread split points | **10.15** |
| 12 | `log_spend` | log1p(spend) | Same | 3.60 |
| 13 | `log_age` | log1p(age) | Risk concentrated at small ages | 8.27 |
| 14–15 | `hour_sin`, `hour_cos` | sin/cos(2π·hour/24) | Hour is cyclic (23:00 next to 00:00) | 3.88 / 2.30 |
| 16 | `night` | hour ∈ {23, 0–4} | 2.67% vs 1.46% (p = 6e-8) | 0.15 |
| 17 | `burst` | 1h ≥ 4 | 16.1% fraud, 9×; 21% of frauds | 0.76 |
| 18 | `high24` | 24h ≥ 11 | 19.7%, 11× | 0.27 |
| 19 | `young` | age ≤ 30 days | 4.27%, 2.6× | 0.16 |
| 20 | `amt_to_avg_ticket` | log1p(amount) − log1p(spend ÷ max(24h count, 1)) | Fraud = unusual *for this account* | 5.84 |
| 21 | `share_1h` | 1h count ÷ max(24h count, 1) | Sudden burst relative to the account's day | 2.79 |
| 22 | `risky_merchant` | luxury / cash_transfer / electronics | The three highest-risk merchants (3.9–5.3%) | 2.34 |
| 23 | `night_x_device` | night × new_device | Significant LRT in 10/10 folds; 16.9% fraud | 3.42 |
| 24 | `n_flags` | count of {burst, high24, young, new_device, night, risky_merchant} | Risk climbs steeply with the number of flags (0.8% → 55%) | **8.69** |

\*CatBoost `PredictionValuesChange` importance, mean over the 3 final models, summing to 100 (computed 2026-10-04).

- **The model leans on amount (raw + log ≈ 19), account age (≈ 15), `n_flags` (8.7), channel, new device and the account-relative ratio.**
- **The single threshold flags (`burst`, `high24`, `young`, `night`) carry little *direct* importance:** their information is also in their continuous parents and in `n_flags`. Removing them still cost CV (§7.4: no flags −0.009; §7.8: raw-only 0.222).
- **Duplicate raw/log representations don't hurt trees** and were kept for split stability.
- **The column order is fixed** (`FEATURES`), because CatBoost results can depend on it.

### 8.4 Hyperparameters, and how each was chosen
| Setting | Value | How chosen / evidence |
|---|---|---|
| `depth` | 4 | Depth 3: −0.006 (0.2404 → 0.2348). Depth 5 and depth 5 / lr 0.02 / 1200 ("slow_deep"): 0.2393 ≤ 0.2396. Depthwise depth 5: 0.2218 |
| `learning_rate` × `iterations` | 0.03 × 800 | Random search (20 configs) and the NB2 search: nothing better. Slower (0.02 × 1200) ties |
| `l2_leaf_reg` | 10 | L2 20–30 (with coarse borders): 0.2378–0.2388 < 0.2396 |
| `random_strength` / `rsm` | defaults (1 / 1.0) | random_strength 3 + rsm 0.5: 0.2315; rsm 0.5 + one-hot: 0.2369 |
| Categorical handling | native ordered target statistics | one-hot (`one_hot_max_size` 12): 0.2320; explicit TE/WoE: no gain |
| `border_count` | default (254) | 32 (coarse): 0.2378 |
| Class weights | **none** | SqrtBalanced 0.198 vs 0.222 in the same probe; weighting hurt every family (−0.016 to −0.04) and destroyed calibration |
| Missing-value augmentation | **none** | Hurt CatBoost: 0.2404 → 0.2322 clean, 0.2191 → 0.2128 test-like |
| Seeds | 0, 1, 2, averaged | Averaging cuts single-fit variance (+0.002 on CV for 5 seeds); 3 seeds produced the uploaded file. More seeds gave no further gain |
| `thread_count` | **3** | **Reproducibility.** CatBoost's result depends on the thread count (up to 0.11 difference in individual probabilities between 3 threads and all cores). 3 produced the 0.19586 file |
| Loss | Logloss (default) | PR-AUC is a ranking metric. Focal loss (tested with LightGBM) was noise or broke. Ranking losses (YetiRank / PairLogit) were not tested |

### 8.5 Training procedure
1. Clean and engineer features (§8.2–8.3). The test set uses the training category lists.
2. **Cross-validation:** 5-fold stratified (`random_state = 0`), 2 seeds per fold, giving out-of-fold predictions for all 20,000 rows.
3. **Threshold:**
   - scan flag shares from 0.2% to 6% in 0.05% steps;
   - at each share compute the confusion matrix on the out-of-fold scores;
   - take the maximum of the 5-point smoothed F1, so a one-off spike isn't picked: **top 1.65%**.
4. **Final fit:** all 20,000 rows, seeds 0, 1, 2; average the predicted probabilities.
5. **Submission:** map scores monotonically so the top 1.65% sit at ≥ 0.51 (linear within [0.51, 1]) and the rest within [0, 0.49]. Write `id,prediction` in sample order, after format checks.

### 8.6 Post-processing (scores → submitted values)
- **The mapping:** `prediction = 0.51 + 0.49·(s − t)/(max − t)` if s ≥ t, otherwise `0.49·(s − min)/(t − min)`, where t is the score at the 98.35th percentile of the scored file.
- **Order-preserving,** so PR-AUC is unchanged (verified on the leaderboard).
- **On a new test file it always flags the top 1.65% of that file.** This is a deliberate choice: a fixed *share* is robust to the probability scale shifting between datasets. A fixed probability cut-off would flag a varying number of rows on a shifted dataset.

### 8.7 Reproducibility
- **Byte-identical reproduction** of the uploaded file (MD5 `d666afc9c1454c2933b2aca8e21933fb`):
  - re-running `04_catboost_rf_logistic_blends.ipynb`;
  - `01_final_catboost.ipynb`;
  - a fresh environment with only `requirements.txt`;
  - real Jupyter (`nbconvert --execute`) from a bare environment.
- **Pinned versions:** catboost 1.2.10, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1; Python ≥ 3.11.
  - pandas 3 vs 2 matters: the cleaning step handles text and blanks differently;
  - CatBoost versions can change trees.

---

## 9. Final results (training out-of-fold, final configuration, recomputed 2026-10-04)
| Metric | Value |
|---|---|
| **PR-AUC (average precision)** | **0.2384**, bootstrap 95% CI **[0.194, 0.292]** (13.5× the random baseline 0.0177) |
| ROC-AUC | 0.7746 |
| Brier score | 0.01503 (constant base-rate prediction: 0.01734) |
| Operating point (top 1.65%) | 330 flagged: **TP 104, FP 226, FN 249**; precision 0.315, **recall 0.295, F1 0.305**; 2.2 false alarms per caught fraud |
| **Public leaderboard PR-AUC** | **0.19586** |

**Alert-budget table** (investigator capacity view):

| Flag share | Flagged | TP | FP | Precision | Recall | F1 | FP per TP |
|---|---|---|---|---|---|---|---|
| 0.5% | 100 | 61 | 39 | **0.610** | 0.173 | 0.269 | 0.64 |
| 1.0% | 200 | 78 | 122 | 0.390 | 0.221 | 0.282 | 1.56 |
| **1.65%** | 330 | 104 | 226 | 0.315 | 0.295 | **0.305** | 2.17 |
| 2.0% | 400 | 110 | 290 | 0.275 | 0.312 | 0.292 | 2.64 |
| 5.0% | 1,000 | 140 | 860 | 0.140 | 0.397 | 0.207 | 6.14 |
| 10.0% | 2,000 | 165 | 1,835 | 0.083 | 0.467 | 0.140 | 11.12 |

- **The top 100 alerts are 61% fraud** (34× the base rate). **Half of all frauds are not caught even at a 10% alert rate:** the "silent fraud" limitation (§3.7).

**Calibration by decile** (mean predicted → observed fraud rate):

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Predicted | 0.004 | 0.005 | 0.006 | 0.007 | 0.008 | 0.009 | 0.011 | 0.014 | 0.021 | 0.088 |
| Observed | 0.007 | 0.006 | 0.004 | 0.006 | 0.008 | 0.012 | 0.011 | 0.013 | 0.029 | 0.083 |

- **Well calibrated:** the unweighted training pays off. Probabilities are meaningful as probabilities, before the order-preserving mapping.
- **Shift signal:** the mean predicted probability is **0.0172 on training (out-of-fold) vs 0.0221 on test.**

**Expected private-leaderboard behaviour:**
- **Noise:** ±0.03 SD for a single submission on ~200 frauds.
- **Range:** ~0.19 if the private data looks like the public test set; up to ~0.24 if it looks more like training.
- **Rank stability:** the plain model is the least public-board-tuned candidate, which should help its rank on private data compared with teams that picked their luckiest public upload.

---

## 10. Assumptions

### 10.1 Model assumptions (CatBoost)
1. **Rows are independent and identically distributed within a dataset.** There's no account id or timestamp, so repeated customers or sequences can't be modelled. Collisions were checked and are at the chance rate, but entity leakage between folds can't be ruled out.
2. **The training labels are correct and complete.**
3. **The relationship between features and fraud is the same in new data as in training.** **This is the assumption the test set strains:** §5.3 shows the same flags are fraud less often in the test set. It explains the ~0.045 CV → public gap.
4. **Gradient-boosted trees approximate P(fraud | x) through additive, shallow interactions** (depth 4 = up to 4-way). Fraud patterns needing deeper interactions or information outside these 10 columns can't be captured.
5. **Ordered target statistics give unbiased categorical encodings.** This requires enough rows per level; the smallest level has ~380 rows (JP).
6. **Blanks are informative only through CatBoost's learned "missing" branch.** In training, blank merchant/device rows had 0 frauds, so the model may treat blanks as slightly safe. The test set has 3× more blanks.

### 10.2 Data and evaluation assumptions
1. **Stratified random CV estimates performance on data like the training data:** not on the shifted test set.
2. **The 1h/24h aggregates were available at transaction time** and contain no future information (can't be verified).
3. **Amounts are comparable across countries** within each dataset (no currency field).
4. **The organisers threshold at 0.5 for F1/Recall** (unconfirmed). The score mapping is designed around this assumption.

### 10.3 Assumptions of the logistic baseline (for the comparison section)
- **Log-odds linear in the features;** independent observations; no perfect multicollinearity (checked: max VIF 2.1); large sample per parameter.
- **Each flag adds a fixed amount to the log-odds regardless of context:** the property that made it misfire on the shifted test set.

---

## 11. Limitations and threats to validity
- **Concept shift:** the dominant limitation (§5). The model's top ranks on test are dominated by new-device and burst rows (78% / 62.5% of the top 2%, vs 38% / 21% among training frauds). Every attempted correction lost on the leaderboard.
- **Small positive class:**
  - 353 frauds: CV PR-AUC CI width ≈ ±0.05;
  - the learning curve is still rising, so more data would likely help;
  - the leaderboard estimate has ±0.03 SD.
- **Silent fraud:** 21.8% of frauds look legitimate on every recorded field. Recall saturates (47% at a 10% alert rate).
- **No entity or time information:** no velocity beyond the given aggregates, no time-based validation, no detection of account takeover sequences.
- **Unclear semantics:** spend window, account-age unit, currency.
- **Selection bias:** the final model was chosen among many candidates using CV and the public leaderboard, so its scores are slightly optimistic. Nested CV or an untouched holdout would be needed for an unbiased estimate.
- **Operational:** no cost or loss labels, so the operating point optimises F1, not business cost.
- **Engineering:** the final notebook (`final_notebook/predict_catboost.ipynb`) scores any test file in its own row order and has no id-order assertion against the sample file, so a different private test file does not crash it.

---

## 12. What did not work, and what we learned
Negative results are evidence of rigour; present them as a table in the report.

| Approach | Rationale | Result | Lesson |
|---|---|---|---|
| Class weighting / cost-sensitive / SMOTE | Standard imbalance remedy | CV −0.016 to −0.04; calibration destroyed | For a ranking metric, imbalance is handled by the operating point, not by reweighting |
| Logistic regression (alone or blended) | Interpretable; best on CV (0.233) | Blends 0.183 public | Fixed per-flag bonuses misfire when flag prevalence shifts |
| Monotone constraints | Domain logic: more risk signal → more risk | ~0.15 public | Forcing every signal to raise risk in every context amplifies misfiring flags |
| More model families (TabPFN, LightGBM, XGBoost, RF, kNN, MLP) | Diversity | ≤ CatBoost | All models fail the same way: a dataset effect, not a model effect |
| Hyperparameter searches (LightGBM 70 configs; CatBoost 20 + 8 configs) | Tuning | No config beat the chosen one | Tiny, regularised trees are best for 353 frauds; tuning is exhausted |
| Seed averaging (5–10) | Variance reduction | No CV gain; public noise | Leaderboard differences < 0.01 are noise |
| Relative / context features (merchant-, country-, dataset-relative) | Make features shift-invariant | CV −0.006 to −0.03 | Absolute magnitudes carry real signal; relativising throws it away |
| Covariate-shift weighting / adversarial validation | Make CV look like test | CV −0.004; proxy mis-ranked models | The shift is in P(y\|x), which covariate methods can't see |
| Synthetic stress tests | Robustness selection | Favoured changes that lost on public | Synthetic shift ≠ real shift |
| Test rows as "not fraud" (P1) | Teach the test's normal population | 0.1878 public | The test's flagged rows aren't simply legitimate |
| Re-anchoring non-SG amounts / counts to training scale | Undo market or unit shifts | 0.1954 / 0.1844 public | The extra non-SG activity contains real fraud |
| Flag-excess correction (Bayes prior per flag cell) | Down-weight over-represented flags | 0.1893 public | Same conclusion |

**Overall:** CatBoost on well-chosen EDA features is at a local optimum for these 10 columns. Moving its top ranks toward or away from the training-like mix both hurt. Clean CV estimates the *in-distribution* skill (0.238, CI 0.19–0.29). The public score (0.196) reflects a genuine concept shift that no available feature resolves.

---

## 13. Suggested report structure and key messages
1. **Introduction and problem understanding** (§1, §1.5): fraud detection under extreme imbalance; why PR-AUC; F1/Recall need an operating point; the private leaderboard on unseen data.
2. **Data and quality audit** (§2): schema, checks passed, semantic ambiguities, missingness. *Table: column audit.*
3. **Exploratory analysis** (§3): amount, velocity, categoricals, time, age, **combinations** (the key insight), silent fraud, statistical tests. *Figures: fraud rate by amount bucket; by 1h count; by number of flags; the combination table.*
4. **From analysis to design** (§4): the traceability table. A strong, explicit "why" section.
5. **Distribution shift analysis** (§5): what changed, where it's concentrated (SG as a control), evidence of concept shift, top-2% diagnostics. *Table §5.1; quantile-ratio table §5.2.* This shows deep problem understanding.
6. **Validation methodology** (§6): stratified CV, in-fold preprocessing, uncertainty, leaderboard noise, the CV-public gap and the failed proxies.
7. **Model development** (§7): logistic baseline → boosting → CatBoost, with the key tables (§7.1, §7.3, §7.7) and the leaderboard history (§7.14).
8. **Final model** (§8): why CatBoost (§8.1), features (§8.3 with importances), hyperparameters with evidence (§8.4), threshold and post-processing (§8.5–8.6), robustness cleaning (§8.2), reproducibility (§8.7).
9. **Results** (§9): metrics with CI, the alert-budget table, calibration, the public score.
10. **Assumptions** (§10), **limitations** (§11), **negative results** (§12).
11. **Conclusion:** what works (an interaction-aware, regularised, unweighted tree model on EDA-driven features); what limits it (concept shift, silent fraud, 353 positives); next steps with more information (account ids and timestamps for sequence features, cost labels, more data).

**Key messages to make explicit:**
- **"We chose CatBoost because it learns the signal combinations where fraud concentrates, encodes categories without leakage, handles missing values natively, and was the most robust model on the shifted test set, not because it had the highest CV score."** (On CV, logistic regression and blends were as good or better.)
- **"We trained unweighted and set recall through the operating point,** because class weighting hurt ranking quality and calibration in every model family."
- **"Every design choice was tested inside cross-validation folds and kept only if it beat noise (> 2 SE)."**
- **"The CV-to-test gap is a dataset property (concept shift),** demonstrated by: a uniform drop across model families; fraud-richer test-like training rows; and the failure of every correction in both directions."

**Claims to avoid:**
- that the model "achieves 0.24 on unseen data";
- that any feature or model "fixes" the shift;
- causal claims ("new devices cause fraud");
- that missingness means safety;
- that account age is in days or amounts in one currency (both unverified).

---

## 14. Reproducibility appendix
| Item | Location |
|---|---|
| Final notebook (deliverable, scoring only) | `final_notebook/predict_catboost.ipynb` + `final_notebook/model.pkl`; see `final_notebook/README.md` |
| Training notebook | `notebooks/01_final_catboost.ipynb` (saves the same `model.pkl` bundle to `results/catboost/`) |
| One-command pipeline | `python -m fraudlab all` (eda → select → train → predict → verify); see `fraudlab/README.md` |
| Uploaded submission (0.19586) | `results/leaderboard_uploads/01_catboost_3seeds_FINAL.csv`, MD5 `d666afc9c1454c2933b2aca8e21933fb` |
| Every public upload | `results/leaderboard.csv` |
| Requirements | `requirements.txt` (catboost 1.2.10, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, …); tested on Python 3.13 / 3.14 |
| EDA reports | `docs/eda/EDA_report.md`, `docs/eda/Extended_EDA_Report.md` |
| Decision log | `docs/research_log/decision_log.md` |
| Research brief (shift diagnosis, every experiment) | `docs/research_log/research_brief_2026-10-04.md` |
| Notebooks | `notebooks/01_final_catboost`, `02_model_comparison`, `03_shift_reanchoring`; indexed in `notebooks/README.md` |
| Model comparison | `results/model_selection/REPORT.md`; see `results/README.md` |

**Run:** Jupyter *Run All*, with the CSVs in `../data/`. About 1–3 minutes on a CPU. Fixed seeds (CV split 0; models 0, 1, 2) and fixed `thread_count = 3` give byte-identical output.

---

## 15. Glossary
- **PR-AUC / average precision (AP):** the area under the precision–recall curve, computed as the mean precision at each recall step. Equals the base rate for a random ranker.
- **OOF (out-of-fold):** predictions for each training row from a model that didn't train on it.
- **Stratified K-fold:** K splits that each keep the overall fraud rate.
- **Covariate shift:** P(x) changes and P(y | x) stays the same. **Concept shift:** P(y | x) changes.
- **Adversarial validation:** a classifier trained to tell train rows from test rows; its AUC measures the shift.
- **Ordered target statistics:** CatBoost's categorical encoding using only "earlier" rows in a random order; prevents target leakage.
- **Brier score:** mean squared error of predicted probabilities (calibration plus sharpness).
- **Winner's curse:** picking the best of many noisy scores overstates its true value.
- **Operating point / flag share:** the fraction of transactions flagged as fraud; it sets precision, recall and F1.
