# Research brief: Datathon Track 2 fraud detection, stuck at ~0.196 public PR-AUC, need ≥ 0.22

> Written 2026-10-04 for a research agent picking this up cold. Everything below was measured; nothing is a guess unless marked. Deeper history: `docs/research_log/decision_log.md` (decisions log), EDA in `docs/eda/`.

## 1. The task and the bar
- **Task:** binary classification of card transactions, `fraud` 0/1. Submit `id,prediction` (a probability in [0,1]) for 12,000 test rows, in the order of `format/sample_submission.csv`.
- **Scoring:** 60% results (PR-AUC, F1, Recall), 40% technical report.
- **Leaderboard:** the public checker scores PR-AUC on the 12k test rows. **Uploads are limited.** A private leaderboard on unseen data decides the final ranking.
- **Qualification target: ≥ 0.22 public PR-AUC.** Current best: **0.19586**. Leaderboard snapshot: 1st 0.2508, 3rd 0.2417, 11 teams ≥ 0.217, 16 teams ≥ 0.200. We were #23 at 0.1935 before the 0.19586 upload.
- **The metric is ranking-based (verified):** the same model as raw probabilities scored 0.19189 vs 0.1925 with a monotone rescaling. Thresholds and calibration can't move PR-AUC.

## 2. Data
- **Train:** 20,000 rows, 353 frauds (1.765%). **Test:** 12,000 rows, no labels.
- **Columns (10):**
  - `transaction_amount`, `transaction_hour` (0–23)
  - `merchant_category` (11 levels), `country` (10), `transaction_channel` (4)
  - `transactions_last_24h`, `spend_last_24h`, `transactions_last_1h`
  - `account_age` (0–4000, days), `new_device` (0/1)
- **No timestamp, no account id.** Ids are sequential and carry no signal.
  - *Checked today:* rows sharing account_age × country collide exactly at the chance rate, so there are **no hidden repeat accounts** to reconstruct.
- **Strongest training signals (fraud rate):**
  - `new_device` 7.5% vs 1.2%
  - 1h count 5–6: 23–30%; 7–9: 17–19%; 0–2: ~1.3%
  - 24h count top decile 5.4%
  - amount top decile 5.6%
  - account age < 72 days 4.6% vs ~1.1%
  - night 23–04 ~2.7%
  - pairs of signals reach 38–40%
  - 21.8% of frauds fire no simple flag
- **Missing values:** 4.9% of train rows vs 13.9% of test rows have a blank. In training, rows with a missing merchant or device had **0 frauds**.

## 3. The core problem: train → test shift, including a change in what the signals mean
| | Train | Test |
|---|---|---|
| new_device rate | 9% | 15% |
| ≥ 4 tx in last 1h | 2.3% | 4.4% |
| amount P90 / P95 | 530 / 978 | 1,707 / 3,580 |
| median amount AU / JP / MY | 122 / 134 / 76 | 1,297 / 1,609 / 164 (unit change?) |
| account age median | 790 | 1,158 (only old accounts added) |
| rows with a missing value | 4.9% | 13.9% |

**Per merchant (the shift is concentrated):**
- New-device rate:
  - utilities 10.7% → 34.8%, retail 10.7% → 23.0%, transport 10.8% → 21.2%, grocery 8.7% → 17.1%, Unknown 9% → 22%
  - luxury, electronics, cash_transfer, travel unchanged or down
- 1h burst rate (≥ 4): gaming 4.9% → 12.4%, food_delivery 2.6% → 7.5%, transport 2.2% → 5.3%, electronics 4.0% → 7.2%, luxury 3.8% → 6.5%.

**Evidence that P(fraud | x) changed, not only P(x):**
1. Every model drops by the same amount: CV ~0.24 → public ~0.19.
2. An adversarial train-vs-test classifier (AUC 0.67) finds that the most test-like training rows are *richer* in fraud (125 of 353 frauds sit in the top 20% most test-like rows). Covariate-shift reasoning would therefore predict a *higher* test score (a weighted proxy gave ~0.34), yet the real score is lower.
3. The test set has many more fraud-looking rows and scores lower. Most likely the extra flagged rows are largely legitimate, but see P1 below: treating them all as legitimate also lost.

**Every model's test top 2% is ~78% new-device rows and ~60–65% bursts.** Among real training frauds the shares are 38% and 21%. Nothing tried has changed this composition by more than a few points while keeping CV.

## 4. Current best model (public 0.19586)
`results/catboost/submission_catboost.csv`, made by `src/nb1_catboost_rf_logistic.ipynb` (the `catboost` model).
- **CatBoostClassifier:** iterations 800, depth 4, learning_rate 0.03, l2_leaf_reg 10, native categoricals, unweighted, average of seeds 0, 1, 2.
- **24 features:**
  - raw columns
  - log amount / spend / age, hour sin/cos
  - night (23–04), burst (1h ≥ 4), high24 (24h ≥ 11), young (age ≤ 30)
  - amount ÷ average ticket (spend_24h / tx_24h), 1h/24h share
  - risky_merchant (luxury / cash_transfer / electronics), night × new_device, n_flags
- **CV (5-fold stratified, seed 0):** PR-AUC 0.2398.

## 5. Every leaderboard submission so far
| Submission | Public PR-AUC |
|---|---|
| **Plain CatBoost, 3 seeds** | **0.19586** (best) |
| TabPFN + LightGBM + logistic rank blend | 0.19354 |
| Plain CatBoost, 1 seed | 0.19248 |
| Plain CatBoost, 10 seeds | 0.19211 |
| Plain CatBoost, 1 seed, raw probabilities (metric probe) | 0.19189 |
| LightGBM | 0.19006 |
| CatBoost + flag-excess correction + within-dataset context features | 0.18930 |
| CatBoost + TabPFN | 0.18804 |
| CatBoost + unlabelled test rows as weight-0.3 negatives ("P1") | 0.18778 |
| TabPFN | 0.18750 |
| CatBoost + logistic 50/50 rank blend | 0.18320 |
| CatBoost + monotone CatBoost + logistic 40/40/20 | ~0.17 |
| Monotone-constrained CatBoost | ~0.15 |
| Other variants: 4-model blend with CatBoost, shift-notebook rounds 1–2, several NB1–NB4 candidates (RF blends, top-3 CatBoost config average) | all < 0.19; exact numbers not recorded |

**Noise:** the same model with a different seed count moves 0.192–0.196, so ±0.004 is seed noise. The paired-difference SD between two models on 12k rows (~212 frauds) is ~0.012.

## 6. Tried offline and rejected without uploading (CV PR-AUC, 5-fold)
- **Class weighting / scale_pos_weight:** −0.016 to −0.022. SMOTE excluded. Focal loss: noise or broke.
- **Explicit TE/WoE, frequency encoding, missing indicators, splines, explicit interactions, L1:** all within ±0.004 or worse.
- **Hyperparameters:**
  - CatBoost depth 3/5, L2 20–30, random_strength 3, rsm 0.5, one-hot categoricals, coarse borders, Depthwise: none beat the current settings.
  - LightGBM tiny-leaf tuning: 0.217. XGBoost worse.
- **Covariate-shift importance weighting:** −0.004.
- **Adversarial "test-like" validation:** does **not** reproduce the leaderboard order (rated the monotone model ≥ CatBoost).
- **Synthetic stress-test suite** (test-like scales, 3× missing, flags without fraud, category noise): also mis-ranked models. It favoured both changes that later lost on public.
- **Context-relative features:**
  - Baselines computed per dataset (merchant-mean-relative counts, new-device surprise per merchant × channel, amount vs merchant × country median, age percentile, hour rarity, out-of-fold merchant-slope risk score):
    - context-only: CV blend 0.206
    - hybrid with raw features: 0.214
    - plain CatBoost: 0.239
  - The test top 2% was still ≥ 79% new device.
- **Decoupled ensemble** (plain CatBoost + logistic on relative features only): no gain. The relative logistic's top 2% was 94% new device.
- **Model families:** random forest 0.207 (blends ≈ CatBoost); in-context kNN 0.179; bagged MLP 0.183 (blends hurt).
- **Probes** (one change each to CatBoost, CV): raw 10 features 0.222; CatBoost defaults 0.213; no new_device 0.168 (its top 2% is still 51% new device); no velocity flags 0.230; SqrtBalanced weights 0.198.
- **Stability:** seed averaging (5–10 seeds) brings no CV gain.

## 7. Lessons that should constrain the next attempt
1. **Stratified CV over-predicts the test score by ~0.045 uniformly,** but it *does* predict the sign of changes that hurt (P1: CV −0.0095, public −0.008). It cannot see shift-specific gains. No offline proxy built so far ranks known public results correctly.
2. **The bottleneck is the ordering inside the top ~2%.** That region is dominated by new-device / burst rows in every model, because in training those really are the strongest signals.
3. **What can't fix it:** removing or relativising those features loses more on training-like data than it gains. Telling the model that the test population is legitimate (P1) also lost. So the test's flagged rows are a *mix* of legitimate and fraudulent rows that our features can't separate.
4. **Teams at 0.22–0.25 exist, so a real improvement is available.** Assume they find something structural, not seed luck.

## 8. Directions not yet tried (for the research agent to evaluate)
- **Ranking objectives:** CatBoost `YetiRank` / `PairLogit` / `QueryCrossEntropy` with one group, optimising order at the top rather than log-loss.
- **Explicit two-stage model for the new-device population:** a separate model *within* new_device = 1 rows, trained on what distinguishes fraud from legit among new-device rows only, then merged with the general model. The global model may be spending its capacity on "new device or not" rather than on ranking inside that group.
- **Rethinking AU / JP / MY amounts:** a ×10 median jump looks like a currency/unit change in test. Converting test amounts back (÷ estimated factor per country) before scoring, instead of relativising, has not been tried.
- **Test-time multiple imputation for the 13.9% of test rows with blanks:** average the score over plausible filled-in values. Stress tests suggested blanks matter little, but it has never been tested on the real test set.
- **Semi-supervised / pseudo-labelling** with *confident positives* only (the opposite of P1).
- **What the top teams likely do differently:** reason from the data-generating process. This is a synthetic dataset: try to recover the generator's fraud rules from the training data (e.g. which signal *combinations* define fraud) and check whether the test set was generated with the same rules but different marginals.
- **Anything else** that changes *which* rows occupy the top 2% of the test ranking, not just their order. Use the label-free check: share of new-device / burst rows in the test top 2%, against 38% / 21% among training frauds.

## 9. Practical constraints
- **Uploads are limited:** propose at most 1–2 candidates, each with a clear hypothesis about the test set. Don't submit many variants and pick the luckiest.
- **Rules:** no test labels (none exist). Test *features* may be used label-free (transductive statistics, unlabelled rows). Fit preprocessing inside CV folds.
- **Environment:** Python at `~/.venv-wsl/bin/python` (catboost 1.2.10, scikit-learn 1.9.1, torch available; no lightgbm/xgboost/optuna installed locally). The user runs final notebooks in Colab. Data: `data/Track 2 Training Dataset.csv`, `data/Track 2 Testing Dataset.csv`.
- **Notebooks in `src/`:**
  - `nb1`: CatBoost / RF / logistic
  - `nb2`: CatBoost configuration search
  - `nb3`: shift-aware features + flag-excess correction
  - `nb4`: kNN / MLP
  - `nb5`: leaderboard probes
  - `nb6`: context / hybrid features (switch `NB6_FEATURES`)
  - `nb7`: P1, test rows as negatives
  - `fraud_catboost_monotone.ipynb`
  - `fraud_catboost_logistic_shift.ipynb` (hardened CatBoost + logistic)
  - `tabpfn.ipynb` (the 0.1935 blend)
- **Results in `results/`:** `catboost/` (the 0.19586 file), `probes/`, `context_blend/`, `hybrid_blend/`, `decoupled/`, `p1_test_as_negatives/`.

## 10. Update (later on 2026-10-04): corrected diagnosis, which supersedes parts of §3 and §7
- **SG is an unchanged control population.** SG test amounts match training at every quantile (ratio 1.0 at 10/25/50/75/90%), and SG is 63% of test rows. **Every non-SG country's amounts are inflated:** ratio 1.1–2.3× in most, AU 2.2/3.7/10.6/7.3/1.7×, JP 2.1/3.8/12.0/5.0/1.4×, MY 1.6/1.5/2.1/8.1/5.0× across those quantiles. AU/JP is **not** a clean ×10 unit change: a new sub-population of mid/high amounts appears. Non-SG velocity also rose (≥ 4/h: MY 5.3% → 13.8%, GB 3.6% → 6.6%).
- **The test top 2% is 89% non-SG rows** for the best model, against 67% in the same model's out-of-fold top 2% on training data, and 48% non-SG among training frauds. The non-SG share of all rows only rose from 30% to 37%. Non-SG rows are genuinely riskier in training (fraud 2.8% vs 1.3% in SG), but test over-promotes them.
- **The new-device rows in the top 2% are at electronics (57), cash_transfer (32) and luxury (30):** merchants where training new-device fraud is 20–27%, and whose new-device rate did *not* rise. The everyday-merchant new-device surge (utilities, retail, SG; training new-device fraud 2–3%) is **not** what crowds the top; the model already ranks it low. §3's "extra new-device rows crowd the top" story is wrong for the top 2%.
- **Mechanism check:** quantile-mapping each non-SG country's test amount and 24h spend onto that country's training distribution (SG untouched, best model unchanged) cuts the top 2% from 89% to 84% non-SG and from 31% to 23% AU/JP/MY, replacing 14% of the top rows (rank corr. 0.969). It moves in the right direction, but explains only part of the excess.
- **Leaderboard verdict on re-anchoring:** amounts only (Addition 1) **0.19544** (no effect); amounts + transaction counts (Addition 2) **0.1844** (−0.0115). The extra non-SG bursts in test carry real fraud signal.
  - **Pattern across all interventions:** moving the test top 2% *towards* the training-like composition hurts (P1 −0.008, Addition 2 −0.0115), and so does amplifying the flags (monotone ~0.15). Plain CatBoost's ranking sits at a local optimum in both directions.
  - **Conclusion:** the test's extra flagged rows are not mostly legitimate. Correcting for the shift is not the path to 0.22. Whatever the top teams do is something else: a different model of the signal, information we aren't using, or luck on top of a slightly better base model.
