# Data

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
