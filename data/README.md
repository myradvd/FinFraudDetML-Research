# Data

> **The competition data is not included in this repository.** It belongs to the datathon organisers. To run
> anything that trains or scores, download the files from the organisers and place them as below. Everything you
> can *read* (figures, executed notebooks, reports, results) works without them.

## Getting the data

```
data/Track 2 Training Dataset.csv
data/Track 2 Testing Dataset.csv
format/sample_submission.csv        # optional: only used to check the submission's id order
```

Or keep them anywhere and point the pipeline at the folder: `FRAUDLAB_DATA_DIR=/path/to/csvs python -m fraudlab all`
(the final notebook takes `DATATHON_INPUT_PATH` instead). Check that you have the exact versions this work used:

```bash
md5sum data/*.csv format/*.csv
# 7792dcf73cf814c34566c8c32e05632b  data/Track 2 Training Dataset.csv
# d3ac85298c39862021593532f53896ad  data/Track 2 Testing Dataset.csv
# 9275e283e76b510f632e7ca8f8dec0f2  format/sample_submission.csv
```

Without the files, `fraudlab` and the notebooks stop with a message pointing here. The unit tests that need data are
skipped, and CI runs only the data-free tests.

## Files

| File | Rows | Columns | Notes |
|---|---|---|---|
| `Track 2 Training Dataset.csv` | 20,000 | 12 | includes the label `fraud` (353 frauds, 1.77%) |
| `Track 2 Testing Dataset.csv` | 12,000 | 11 | no label; ids continue after training (`FR020001`…) |
| `../format/sample_submission.csv` | 12,000 | 2 | `id, prediction`, ids in test order |

## Columns

| Column | Type | Description |
|---|---|---|
| `id` | id | transaction id. Carries no signal (Spearman with fraud 0.001), never a feature |
| `transaction_amount` | float | amount; heavy right tail, max 9,000 |
| `transaction_hour` | int 0–23 | hour of day |
| `merchant_category` | 11 levels | retail, gaming, entertainment, utilities, food_delivery, electronics, grocery, travel, transport, cash_transfer, luxury |
| `country` | 10 levels | SG (70% of rows), GB, TH, ID, PH, AU, US, VN, MY, JP |
| `transaction_channel` | 4 levels | ecommerce, mobile_app, card_present, bank_transfer |
| `transactions_last_24h` | int | account's transactions in the last 24 hours |
| `spend_last_24h` | float | account's spend in the last 24 hours (semantics ambiguous: amount > spend in 3,904 rows) |
| `account_age` | int | account age in days, capped at 4,000 |
| `new_device` | 0/1 | transaction from a device not seen before |
| `transactions_last_1h` | int | account's transactions in the last hour |
| `fraud` | 0/1 | label (train only) |

There is no timestamp and no account id, so only stratified random cross-validation is possible.

## Data quality (`python -m fraudlab eda` → `results/eda/`)

- No duplicate ids, no duplicate rows, no overlap between train and test.
- Blanks: 0.5–0.8% per column in train, **2.0–2.4% in test** (4.9% → 13.9% of rows have at least one blank).
  Blanks are not predictive of fraud (Fisher p = 0.45).
- **The test set is shifted:** new devices 9% → 15%, ≥ 4 transactions/hour 2.3% → 4.4%, amount P90 530 → 1,707, while
  Singapore's distributions are unchanged at every quantile. A train-vs-test classifier reaches AUC 0.67. The full
  analysis is in [`../docs/eda/`](../docs/eda/) and [`../docs/TECHNICAL_RECORD.md` §5](../docs/TECHNICAL_RECORD.md).
