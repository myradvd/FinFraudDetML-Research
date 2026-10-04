"""Paths, schema and the frozen settings of the final model.

Everything here matches `final_notebook/predict_catboost.ipynb` and the bundle in `final_notebook/model.pkl`.
Changing a value here changes the model, so `python -m fraudlab verify` checks them against that bundle.
"""
import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FRAUDLAB_DATA_DIR", ROOT / "data"))   # the competition CSVs are not in the repo
TRAIN_CSV = DATA_DIR / "Track 2 Training Dataset.csv"
TEST_CSV = DATA_DIR / "Track 2 Testing Dataset.csv"
SAMPLE_SUBMISSION = ROOT / "format" / "sample_submission.csv"

RESULTS_DIR = ROOT / "results"
SELECTION_DIR = RESULTS_DIR / "model_selection"     # output of `fraudlab select`
FINAL_DIR = RESULTS_DIR / "final"                   # output of `fraudlab train` / `predict`
FIGURES_DIR = ROOT / "docs" / "figures"             # output of `fraudlab eda` (embedded in the READMEs)
LEADERBOARD_CSV = RESULTS_DIR / "leaderboard.csv"   # every public-leaderboard upload and its score
SHIPPED_MODEL = ROOT / "final_notebook" / "model.pkl"

# --- Schema ------------------------------------------------------------------------------------------------------
TARGET = "fraud"
ID = "id"
CATS = ["merchant_category", "country", "transaction_channel"]
RAW_COLS = ["transaction_amount", "transaction_hour", "merchant_category", "country", "transaction_channel",
            "transactions_last_24h", "spend_last_24h", "account_age", "new_device", "transactions_last_1h"]
NUMERIC_RAW = [c for c in RAW_COLS if c not in CATS]
NON_NEGATIVE = ["transaction_amount", "spend_last_24h", "transactions_last_1h", "transactions_last_24h", "account_age"]
LEVELS = {                                          # category levels seen in training
    "merchant_category": ["retail", "gaming", "entertainment", "utilities", "food_delivery", "electronics",
                          "grocery", "travel", "transport", "cash_transfer", "luxury"],
    "country": ["SG", "GB", "TH", "ID", "PH", "AU", "US", "VN", "MY", "JP"],
    "transaction_channel": ["ecommerce", "mobile_app", "card_present", "bank_transfer"],
}
CAT_LEVELS = {c: {v.lower(): v for v in LEVELS[c]} for c in CATS}    # trimmed lower-case spelling -> training spelling
CATEGORIES = {c: sorted(LEVELS[c] + ["Unknown"]) for c in CATS}      # category lists the model was trained with

# The 24 model features, in the exact order of the 0.19586 submission (CatBoost can depend on column order)
FEATURES = ["transaction_amount", "transaction_hour", "merchant_category", "country", "transaction_channel",
            "transactions_last_24h", "spend_last_24h", "account_age", "new_device", "transactions_last_1h",
            "log_amt", "log_spend", "log_age", "hour_sin", "hour_cos", "night", "burst", "high24", "young",
            "amt_to_avg_ticket", "share_1h", "risky_merchant", "night_x_device", "n_flags"]
FLAG_COLS = ["burst", "high24", "young", "new_device", "night", "risky_merchant"]

# --- Final model -------------------------------------------------------------------------------------------------
FINAL_MODEL = "catboost"
CATBOOST_PARAMS = dict(iterations=800, depth=4, learning_rate=0.03, l2_leaf_reg=10)
THREADS = 3               # CatBoost's fitted trees depend on the thread count; 3 produced the 0.19586 file
FINAL_SEEDS = [0, 1, 2]   # the final model averages one CatBoost per seed
CV_SEEDS = [0, 1]         # two seeds per fold keep cross-validation short
N_SPLITS = 5
CV_RANDOM_STATE = 0

# Threshold search grid: share of rows flagged, 0.2%-6% in 0.05% steps
SHARE_GRID = (0.002, 0.0601, 0.0005)

# MD5 of the submission that scored 0.19586 on the public leaderboard
EXPECTED_SUBMISSION_MD5 = "d666afc9c1454c2933b2aca8e21933fb"
