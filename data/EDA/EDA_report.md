# Targeted fraud EDA and modeling decisions

## Executive summary
The dataset contains **20,000 transactions and 353 frauds (1.765%)**, approximately **55.66 non-fraud transactions per fraud**. An always-non-fraud rule would obtain **98.235% accuracy**, making accuracy a poor primary metric. These are descriptive training-data findings, not validated model performance or causal effects.

The strongest observed risk concentrations are high transaction velocity, new devices, and combinations of high amounts with either of those signals. Missingness is sparse and has no statistically convincing univariate association with fraud after adjustment. There is no timestamp or account/customer identifier, so chronological and entity-separated evaluation cannot be established from the supplied schema.

## 1. Dataset health and ordering
- All IDs are unique; no exact duplicates, duplicate feature-only rows, conflicting feature-group labels, or conflicting ID labels were found.
- All labels are present and binary. All amounts are positive, hours are integers in 0–23, counts are nonnegative integers, and observed 1h counts never exceed observed 24h counts.
- Account ages are nonnegative integers and observed device status is binary. Units for account age and amounts, timezone, and source provenance were not supplied.
- Zero recent spend occurs in 557 rows; this is not automatically an error, but requires clarification because observed 24h counts are at least 1. Current amount exceeds recent spend in many rows, so current-transaction inclusion cannot be assumed.
- ID suffixes are not monotonic; row-order versus numeric-ID Spearman correlation is 0.0038. This does not prove random sampling or establish chronology. Hour-of-day alone cannot reconstruct dates.
- No rows were removed for this EDA. Missing entries were retained as missing/Unknown, not replaced with zero.

### Complete column audit
| column                | dtype   |   unique |   missing_count |   missing_pct |
|:----------------------|:--------|---------:|----------------:|--------------:|
| id                    | object  |    20000 |               0 |         0     |
| transaction_amount    | float64 |    14134 |               0 |         0     |
| transaction_hour      | int64   |       24 |               0 |         0     |
| merchant_category     | object  |       11 |             113 |         0.565 |
| country               | object  |       10 |             107 |         0.535 |
| transaction_channel   | object  |        4 |             123 |         0.615 |
| transactions_last_24h | float64 |       27 |             159 |         0.795 |
| spend_last_24h        | float64 |    16987 |             132 |         0.66  |
| account_age           | float64 |     3366 |             137 |         0.685 |
| new_device            | float64 |        2 |             124 |         0.62  |
| transactions_last_1h  | float64 |       10 |             142 |         0.71  |
| fraud                 | int64   |        2 |               0 |         0     |

## 2. Missingness versus fraud
| column                |   missing_count |   missing_pct |   fraud_count_missing | fraud_rate_missing   | fraud_rate_present   | fisher_p   | action                                                    | fdr_q   |
|:----------------------|----------------:|--------------:|----------------------:|:---------------------|:---------------------|:-----------|:----------------------------------------------------------|:--------|
| id                    |               0 |         0     |                     0 | —                    | 1.76%                | —          | No imputation needed                                      | —       |
| transaction_amount    |               0 |         0     |                     0 | —                    | 1.76%                | —          | No imputation needed                                      | —       |
| transaction_hour      |               0 |         0     |                     0 | —                    | 1.76%                | —          | No imputation needed                                      | —       |
| merchant_category     |             113 |         0.565 |                     0 | 0.00%                | 1.78%                | 0.2727     | Unknown category                                          | 0.4286  |
| country               |             107 |         0.535 |                     1 | 0.93%                | 1.77%                | 1          | Unknown category                                          | 1       |
| transaction_channel   |             123 |         0.615 |                     4 | 3.25%                | 1.76%                | 0.173      | Unknown category                                          | 0.4286  |
| transactions_last_24h |             159 |         0.795 |                     5 | 3.14%                | 1.75%                | 0.2092     | Train-fitted median or native missing; do not assume zero | 0.4286  |
| spend_last_24h        |             132 |         0.66  |                     4 | 3.03%                | 1.76%                | 0.2996     | Train-fitted median or native missing; do not assume zero | 0.4286  |
| account_age           |             137 |         0.685 |                     3 | 2.19%                | 1.76%                | 0.5228     | Train-fitted median or native missing; do not assume zero | 0.5975  |
| new_device            |             124 |         0.62  |                     0 | 0.00%                | 1.78%                | 0.2856     | Unknown category                                          | 0.4286  |
| transactions_last_1h  |             142 |         0.71  |                     4 | 2.82%                | 1.76%                | 0.3215     | Train-fitted median or native missing; do not assume zero | 0.4286  |
| fraud                 |               0 |         0     |                     0 | —                    | 1.76%                | —          | No imputation needed                                      | —       |

Missing rates range from 0.535% to 0.795% among affected columns. For example, channel-missing rows have 3.25% fraud versus 1.76% when present, but this is only 4 frauds in 123 rows. All eight missingness tests have Benjamini–Hochberg adjusted q-values at least 0.429. Consequently, missingness indicators remain candidates to validate, not proven predictors; zero frauds among missing merchant/device entries should not be interpreted as zero risk.


## 3. Amount analysis
| feature            |   fraud |   count |   mean |   min |   max |   p25 |    p50 |   p75 |    p90 |    p95 |   p99 |
|:-------------------|--------:|--------:|-------:|------:|------:|------:|-------:|------:|-------:|-------:|------:|
| transaction_amount |       0 |   19647 |  246.9 |   2.5 |  9000 | 31.66 |  74.26 | 198.9 |  509.2 |  935.8 |  3727 |
| transaction_amount |       1 |     353 |  570.8 |   3.4 |  5447 | 52.44 | 224.3  | 684.3 | 1657   | 2484   |  4038 |

| amount_bucket   |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:----------------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| [0.0, 10.0)     |      883 |            11 | 1.25%        | 0.71×  | 0.70%      | 2.22%       |
| [10.0, 50.0)    |     6708 |            74 | 1.10%        | 0.63×  | 0.88%      | 1.38%       |
| [50.0, 100.0)   |     4070 |            43 | 1.06%        | 0.60×  | 0.79%      | 1.42%       |
| [100.0, 500.0)  |     6210 |           104 | 1.67%        | 0.95×  | 1.38%      | 2.03%       |
| [500.0, 1000.0) |     1146 |            62 | 5.41%        | 3.07×  | 4.24%      | 6.88%       |
| [1000.0, inf)   |      983 |            59 | 6.00%        | 3.40×  | 4.68%      | 7.67%       |

Fraudulent transactions have median amount **224.26**, compared with **74.26** for non-fraud; means are 570.81 versus 246.95. The rate rises from around 1.1% in the 10–100 range to 5.41% at 500–1,000 and 6.00% at 1,000+. Amounts of 1,000+ represent 983 transactions and 59 frauds, or 16.71% of all frauds. Amounts below 10 account for 11 frauds (3.12% of frauds), with a below-baseline 1.25% fraud rate.

Integer amounts contain only 1 fraud in 198 rows (0.51%); multiples of 10 contain 0 frauds in just 24 rows. These small samples do not support a round-amount fraud feature. Preserve outliers: the maximum non-fraud amount is 9,000, whereas the maximum fraud amount is 5,446.58, so unusually large amounts are not deterministic fraud labels. Test log1p(amount) alongside raw amount.

Plots: amount_distributions.png; amount_bucket_account_age.png (within this bundle).


## 4. Velocity and relative-amount hypotheses
| feature               |   fraud |   count |      mean |   min |   max |    p25 |   p50 |    p75 |    p90 |   p95 |     p99 |
|:----------------------|--------:|--------:|----------:|------:|------:|-------:|------:|-------:|-------:|------:|--------:|
| transactions_last_24h |       0 |   19493 |    4.585  |     1 |    27 |   3    |   4   |    6   |    7   |     8 |   12    |
| transactions_last_24h |       1 |     348 |    7.606  |     1 |    27 |   4    |   5   |    8   |   19.3 |    23 |   26.06 |
| spend_last_24h        |       0 |   19519 |  539.2    |     0 | 15670 |  93.32 | 214.5 |  562.8 | 1355   |  2121 | 4755    |
| spend_last_24h        |       1 |     349 | 1451      |     0 | 13230 | 187.4  | 662.8 | 1898   | 4363   |  5491 | 7569    |
| transactions_last_1h  |       0 |   19509 |    0.8547 |     0 |     9 |   0    |   1   |    1   |    2   |     3 |    4    |
| transactions_last_1h  |       1 |     349 |    1.911  |     0 |     9 |   0    |   1   |    3   |    5.2 |     7 |    9    |

### transactions_last_1h buckets
| bucket   |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:---------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| 0        |     8659 |           119 | 1.37%        | 0.78×  | 1.15%      | 1.64%       |
| 1        |     7353 |            96 | 1.31%        | 0.74×  | 1.07%      | 1.59%       |
| 2–3      |     3380 |            59 | 1.75%        | 0.99×  | 1.36%      | 2.24%       |
| 4–5      |      299 |            40 | 13.38%       | 7.58×  | 9.98%      | 17.70%      |
| 6–10     |      167 |            35 | 20.96%       | 11.87× | 15.47%     | 27.75%      |
| Unknown  |      142 |             4 | 2.82%        | 1.60×  | 1.10%      | 7.02%       |

### transactions_last_24h buckets
| bucket   |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:---------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| 1        |      790 |             8 | 1.01%        | 0.57×  | 0.51%      | 1.99%       |
| 2–3      |     5802 |            72 | 1.24%        | 0.70×  | 0.99%      | 1.56%       |
| 4–5      |     7789 |           110 | 1.41%        | 0.80×  | 1.17%      | 1.70%       |
| 6–10     |     5090 |            85 | 1.67%        | 0.95×  | 1.35%      | 2.06%       |
| 11+      |      370 |            73 | 19.73%       | 11.18× | 15.99%     | 24.09%      |
| Unknown  |      159 |             5 | 3.14%        | 1.78×  | 1.35%      | 7.15%       |

### spend_last_24h buckets
| bucket    |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:----------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| 0         |      557 |             6 | 1.08%        | 0.61×  | 0.49%      | 2.33%       |
| >0–100    |     4754 |            49 | 1.03%        | 0.58×  | 0.78%      | 1.36%       |
| 100–500   |     9001 |            94 | 1.04%        | 0.59×  | 0.85%      | 1.28%       |
| 500–1000  |     2570 |            58 | 2.26%        | 1.28×  | 1.75%      | 2.91%       |
| 1000–2500 |     2174 |            81 | 3.73%        | 2.11×  | 3.01%      | 4.61%       |
| 2500+     |      812 |            61 | 7.51%        | 4.26×  | 5.89%      | 9.53%       |
| Unknown   |      132 |             4 | 3.03%        | 1.72×  | 1.18%      | 7.53%       |

Transactions with **4+ in the last hour** have **16.09% fraud (75/466)**, **9.12×** the overall rate; this group captures **21.25%** of all frauds while covering 2.33% of transactions. The 24h count of 11+ has **19.73% fraud (73/370)** and **11.18×** baseline risk. Recent spend of 2,500+ has 7.51% fraud (61/812).

Relative features do not all help equally: burst ratio >=0.5 has 2.19% fraud, while current amount exceeding recent spend has 1.82% versus 1.74% otherwise. The latter is not a strong standalone discriminator here. Retain ratios as experimental features, with missing and zero-denominator flags; flooring denominators at 1 changes interpretation at small/zero values. All ratio distributions and quantile risk tables are exported.


## 5. Categorical risk
Rates include Wilson 95% intervals and baseline lift. Low-volume categories and combinations require caution.

### merchant_category
| merchant_category   |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:--------------------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| luxury              |      586 |            31 | 5.29%        | 3.00×  | 3.75%      | 7.41%       |
| cash_transfer       |     1498 |            71 | 4.74%        | 2.69×  | 3.77%      | 5.94%       |
| electronics         |     1524 |            60 | 3.94%        | 2.23×  | 3.07%      | 5.03%       |
| travel              |     1094 |            26 | 2.38%        | 1.35×  | 1.63%      | 3.46%       |
| food_delivery       |     2304 |            38 | 1.65%        | 0.93×  | 1.20%      | 2.26%       |
| gaming              |     1339 |            21 | 1.57%        | 0.89×  | 1.03%      | 2.39%       |
| utilities           |     1581 |            19 | 1.20%        | 0.68×  | 0.77%      | 1.87%       |
| entertainment       |     1407 |            14 | 1.00%        | 0.56×  | 0.59%      | 1.66%       |
| retail              |     2601 |            25 | 0.96%        | 0.54×  | 0.65%      | 1.42%       |
| transport           |     2387 |            21 | 0.88%        | 0.50×  | 0.58%      | 1.34%       |
| grocery             |     3566 |            27 | 0.76%        | 0.43×  | 0.52%      | 1.10%       |
| Unknown             |      113 |             0 | 0.00%        | 0.00×  | 0.00%      | 3.29%       |

### country
| country   |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:----------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| ID        |      840 |            34 | 4.05%        | 2.29×  | 2.91%      | 5.60%       |
| VN        |      508 |            19 | 3.74%        | 2.12×  | 2.41%      | 5.77%       |
| PH        |      581 |            20 | 3.44%        | 1.95×  | 2.24%      | 5.26%       |
| JP        |      382 |            13 | 3.40%        | 1.93×  | 2.00%      | 5.73%       |
| US        |      693 |            22 | 3.17%        | 1.80×  | 2.11%      | 4.76%       |
| GB        |      505 |            16 | 3.17%        | 1.80×  | 1.96%      | 5.08%       |
| AU        |      584 |            17 | 2.91%        | 1.65×  | 1.83%      | 4.61%       |
| TH        |      644 |            11 | 1.71%        | 0.97×  | 0.96%      | 3.03%       |
| MY        |     1225 |            18 | 1.47%        | 0.83×  | 0.93%      | 2.31%       |
| SG        |    13931 |           182 | 1.31%        | 0.74×  | 1.13%      | 1.51%       |
| Unknown   |      107 |             1 | 0.93%        | 0.53×  | 0.17%      | 5.11%       |

### transaction_channel
| transaction_channel   |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:----------------------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| Unknown               |      123 |             4 | 3.25%        | 1.84×  | 1.27%      | 8.06%       |
| bank_transfer         |     2009 |            57 | 2.84%        | 1.61×  | 2.20%      | 3.66%       |
| ecommerce             |     5507 |           132 | 2.40%        | 1.36×  | 2.02%      | 2.84%       |
| mobile_app            |     5172 |            87 | 1.68%        | 0.95×  | 1.37%      | 2.07%       |
| card_present          |     7189 |            73 | 1.02%        | 0.58×  | 0.81%      | 1.27%       |

### new_device
| new_device   |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:-------------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| 1.0          |     1800 |           135 | 7.50%        | 4.25×  | 6.37%      | 8.81%       |
| 0.0          |    18076 |           218 | 1.21%        | 0.68×  | 1.06%      | 1.38%       |
| Unknown      |      124 |             0 | 0.00%        | 0.00×  | 0.00%      | 3.00%       |

New devices have **7.50% fraud (135/1,800)** versus **1.21% (218/18,076)** for known existing devices: **6.22×** the existing-device rate and **4.25×** the overall rate. Luxury (5.29%), cash transfer (4.74%), and electronics (3.94%) have the highest merchant rates. Indonesia has the highest observed country rate (4.05%); Singapore dominates volume (13,931 rows) and contributes 182 frauds despite its lower 1.31% rate. Bank transfers have 2.84% fraud versus 1.02% for card-present.

These are low-cardinality categories (11 merchants, 10 countries, 4 channels), so one-hot or native categorical handling is a sensible first comparison. High-cardinality target encoding is not necessary for the main effects.


## 6. Time of day
|   transaction_hour |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|-------------------:|---------:|--------------:|:-------------|:-------|:-----------|:------------|
|                  0 |      842 |            20 | 2.38%        | 1.35×  | 1.54%      | 3.64%       |
|                  1 |      841 |            25 | 2.97%        | 1.68×  | 2.02%      | 4.35%       |
|                  2 |      804 |            27 | 3.36%        | 1.90×  | 2.32%      | 4.84%       |
|                  3 |      923 |            22 | 2.38%        | 1.35×  | 1.58%      | 3.58%       |
|                  4 |      859 |            17 | 1.98%        | 1.12×  | 1.24%      | 3.15%       |
|                  5 |      821 |             6 | 0.73%        | 0.41×  | 0.34%      | 1.59%       |
|                  6 |      765 |            15 | 1.96%        | 1.11×  | 1.19%      | 3.21%       |
|                  7 |      809 |            17 | 2.10%        | 1.19×  | 1.32%      | 3.34%       |
|                  8 |      768 |             9 | 1.17%        | 0.66×  | 0.62%      | 2.21%       |
|                  9 |      830 |             6 | 0.72%        | 0.41×  | 0.33%      | 1.57%       |
|                 10 |      883 |            14 | 1.59%        | 0.90×  | 0.95%      | 2.64%       |
|                 11 |      871 |            12 | 1.38%        | 0.78×  | 0.79%      | 2.39%       |
|                 12 |      839 |            17 | 2.03%        | 1.15×  | 1.27%      | 3.22%       |
|                 13 |      880 |            17 | 1.93%        | 1.09×  | 1.21%      | 3.07%       |
|                 14 |      819 |            11 | 1.34%        | 0.76×  | 0.75%      | 2.39%       |
|                 15 |      850 |             9 | 1.06%        | 0.60×  | 0.56%      | 2.00%       |
|                 16 |      861 |            16 | 1.86%        | 1.05×  | 1.15%      | 3.00%       |
|                 17 |      863 |            16 | 1.85%        | 1.05×  | 1.14%      | 2.99%       |
|                 18 |      828 |            12 | 1.45%        | 0.82×  | 0.83%      | 2.52%       |
|                 19 |      785 |             8 | 1.02%        | 0.58×  | 0.52%      | 2.00%       |
|                 20 |      859 |            18 | 2.10%        | 1.19×  | 1.33%      | 3.29%       |
|                 21 |      831 |             4 | 0.48%        | 0.27×  | 0.19%      | 1.23%       |
|                 22 |      785 |            11 | 1.40%        | 0.79×  | 0.78%      | 2.49%       |
|                 23 |      784 |            24 | 3.06%        | 1.73×  | 2.07%      | 4.51%       |

| period      |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:------------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| Night 23–04 |     5053 |           135 | 2.67%        | 1.51×  | 2.26%      | 3.15%       |
| Other hours |    14947 |           218 | 1.46%        | 0.83×  | 1.28%      | 1.66%       |

Night is explicitly defined as **23:00–04:59**. Its fraud rate is **2.67% (135/5,053)** versus **1.46% (218/14,947)** at other hours, a **1.83×** relative risk. Hour 02 has the highest hourly rate (3.36%, 27 frauds), while hour 21 has the lowest (0.48%, 4 frauds). Volumes range from 765 to 923 per hour. Use cyclic hour encoding and test the night flag; do not mistake hour-of-day variation for changes across dates.


## 7. Rationale-driven interactions
The high-amount threshold is the full-data descriptive P95, **977.963**; in a pipeline, estimate thresholds from training folds only. Burst means 4+ transactions in the last hour. Missing device/velocity values are kept separate.

### high_amount_new_device
| high_amount_new_device   |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:-------------------------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| False | 0.0              |    17162 |           190 | 1.11%        | 0.63×  | 0.96%      | 1.27%       |
| False | 1.0              |     1718 |           104 | 6.05%        | 3.43×  | 5.02%      | 7.28%       |
| False | nan              |      120 |             0 | 0.00%        | 0.00×  | 0.00%      | 3.10%       |
| True | 0.0               |      914 |            28 | 3.06%        | 1.74×  | 2.13%      | 4.39%       |
| True | 1.0               |       82 |            31 | 37.80%       | 21.42× | 28.08%     | 48.62%      |
| True | nan               |        4 |             0 | 0.00%        | 0.00×  | 0.00%      | 48.99%      |

### burst_high_amount
| burst_high_amount   |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:--------------------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| False | False       |    18461 |           240 | 1.30%        | 0.74×  | 1.15%      | 1.47%       |
| False | True        |      931 |            34 | 3.65%        | 2.07×  | 2.62%      | 5.06%       |
| True | False        |      406 |            51 | 12.56%       | 7.12×  | 9.68%      | 16.14%      |
| True | True         |       60 |            24 | 40.00%       | 22.66× | 28.57%     | 52.63%      |
| Unknown | False     |      133 |             3 | 2.26%        | 1.28×  | 0.77%      | 6.42%       |
| Unknown | True      |        9 |             1 | 11.11%       | 6.30×  | 1.99%      | 43.50%      |

### night_new_device
| night_new_device   |   volume |   fraud_count | fraud_rate   | lift   | ci95_low   | ci95_high   |
|:-------------------|---------:|--------------:|:-------------|:-------|:-----------|:------------|
| False | 0.0        |    13452 |           150 | 1.12%        | 0.63×  | 0.95%      | 1.31%       |
| False | 1.0        |     1403 |            68 | 4.85%        | 2.75×  | 3.84%      | 6.10%       |
| False | nan        |       92 |             0 | 0.00%        | 0.00×  | 0.00%      | 4.01%       |
| True | 0.0         |     4624 |            68 | 1.47%        | 0.83×  | 1.16%      | 1.86%       |
| True | 1.0         |      397 |            67 | 16.88%       | 9.56×  | 13.51%     | 20.87%      |
| True | nan         |       32 |             0 | 0.00%        | 0.00×  | 0.00%      | 10.72%      |

- High amount plus a new device: **37.80% fraud (31/82)**, **21.42×** baseline; Wilson interval **28.08–48.62%**. This group contains 8.78% of all frauds.
- High amount plus a burst: **40.00% (24/60)**, **22.66×** baseline; interval **28.57–52.63%**. This contains 6.80% of frauds.
- Night plus new device: **16.88% (67/397)**, **9.56×** baseline.
- Pairwise country×merchant and channel×merchant tables are included. Treat their sparse cells as exploratory, not deployable risk estimates. Use main effects first; if using risk encodings, fit smoothed encodings strictly out-of-fold.

These concentrations justify testing interactions but do not establish causal mechanisms or incremental predictive value beyond the main effects. Subgroups overlap, so their fraud capture cannot be added.


## 8. Statistical checks, correlations and leakage
| hypothesis         |   group_count |   fraud_count | group_rate   | comparator_rate   |   risk_ratio |   fisher_p |     fdr_q |
|:-------------------|--------------:|--------------:|:-------------|:------------------|-------------:|-----------:|----------:|
| New device         |          1800 |           135 | 7.50%        | 1.21%             |        6.219 |  2.833e-51 | 8.5e-51   |
| 1h count >=4       |           466 |            75 | 16.09%       | 1.41%             |       11.39  |  2.176e-50 | 4.351e-50 |
| 24h count >=11     |           370 |            73 | 19.73%       | 1.41%             |       13.97  |  1.151e-55 | 6.904e-55 |
| Night 23–04        |          5053 |           135 | 2.67%        | 1.46%             |        1.832 |  6.033e-08 | 7.239e-08 |
| Top 5% amount      |          1000 |            59 | 5.90%        | 1.55%             |        3.813 |  2.323e-16 | 3.485e-16 |
| Young account <=30 |           749 |            32 | 4.27%        | 1.66%             |        2.568 |  5.137e-06 | 5.137e-06 |

| feature             |   cramers_v_bias_corrected |   chi_square_p |
|:--------------------|---------------------------:|---------------:|
| merchant_category   |                    0.1007  |      9.819e-40 |
| country             |                    0.06248 |      1.295e-14 |
| transaction_channel |                    0.04852 |      2.154e-10 |
| new_device          |                    0.1368  |      1.984e-82 |

Fisher tests compare each stated group with its valid complement; FDR adjustment is within this six-test family. They do not correct every exploratory bucket or interaction examined here. Wilson intervals are pointwise rather than simultaneous, and inference assumes independent transactions, which cannot be verified without entity identifiers.

Numeric Pearson and Spearman matrices are exported. Largest absolute numeric Spearman correlation with fraud is new_device (0.137), followed by spend_last_24h (0.082), amount (0.070), and account_age (-0.059). No inspected numeric or categorical main effect is almost perfectly predictive. This is not proof that leakage is absent: nonlinear combinations and undocumented feature construction can still leak.

Confirm with the data provider that count/spend windows are calculated only from information available at transaction scoring, not future transactions; clarify whether they include the current transaction. Verify device flags and account age are also available at scoring time. Exclude ID from the predictor set. Feature-only duplicate and label-conflict checks passed, but repeated account/customer observations cannot be audited without an entity key.

## 9. Account age: additional schema-driven finding
Account age median is **335.5 for fraud versus 796 for non-fraud**. The <=30 group has **4.27% fraud (32/749)** versus **1.66%** for larger observed ages. Use raw/log age and cautiously test a young-account flag; do not interpret the unit as days without confirmation.

## 10. Recommended preprocessing, features and evaluation
1. **Splits:** with current fields, use a reproducible stratified split, such as 60/20/20, retaining about 212/71/70 frauds; exact counts depend on the splitter. This is a provisional non-temporal estimate. If timestamps/entities become available, prioritize chronological evaluation and assess entity overlap. Hour and ID suffix must not be used as invented chronology.
2. **Missingness:** numeric median fitted on training folds, or model-native missing handling; categorical Unknown; device 0/1/Unknown. Avoid velocity/spend zero imputation without semantics. Evaluate indicators by held-out ablation rather than relying on noisy missing-group rates.
3. **Core features:** amount/raw and log1p; 1h/24h counts and spend; device; merchant/country/channel; account age; sine/cosine hour and night. Prioritize amount×new-device, burst×amount, and night×new-device for validation. Do not prioritize round-amount or amount/spend flags based on these results alone.
4. **Model comparisons:** an interpretable, regularized logistic baseline versus a nonlinear tree-boosting model. These are proposed experiments, not fitted models. Compare raw versus transformed amount/age, relative features, interactions and missingness indicators via ablations.
5. **Metrics:** average precision/PR-AUC (state the implementation), recall, precision, alert rate, and recall at a fixed alert budget. Use ROC-AUC secondarily; report uncertainty and calibration. Choose thresholds on validation data against review capacity/costs, not on the final test set. Overall prevalence 1.765% gives a descriptive random-ranking precision baseline; use each evaluation set's own prevalence when reporting results.
6. **Imbalance:** compare class weighting with an unweighted baseline and check calibration. Any resampling must occur inside training folds only. Do not impute, select thresholds, encode targets, or select features using validation/test labels.
7. **Deployment caveats:** no transaction dates, account identifiers, loss/cost labels, currency metadata or provenance were supplied. Time drift, account-level leakage, business-optimal thresholds, causal claims and operational performance remain unverified.

## Deliverables and scope
The bundle provides all six required outputs: prevalence, missingness, amount, velocity, categorical risk, and hourly risk, plus targeted interactions, correlations, hypothesis tests, a reproducible analysis script and charts. No predictive model was trained and no held-out performance is claimed. Analyses use only the supplied dataset; external references in the guidelines were not needed to calculate these findings.
