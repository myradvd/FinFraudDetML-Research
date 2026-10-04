"""Feature engineering: 24 per-row features, each traced to an EDA finding.

Every feature uses only the row's own values (no labels, no other rows), so nothing can leak across folds.
"""
import numpy as np
import pandas as pd

from . import config as C
from .data import sanitise

# Extra smooth velocity columns used by the linear / distance-based challengers (not by the final CatBoost)
EXTRA = ["log_tx_1h", "log_tx_24h"]


def engineer(d):
    d = sanitise(d)
    d["log_amt"] = np.log1p(d.transaction_amount)
    d["log_spend"] = np.log1p(d.spend_last_24h)
    d["log_age"] = np.log1p(d.account_age)
    d["hour_sin"] = np.sin(2 * np.pi * d.transaction_hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d.transaction_hour / 24)
    d["night"] = d.transaction_hour.isin([23, 0, 1, 2, 3, 4]).astype(int)
    d["burst"] = (d.transactions_last_1h >= 4).astype(float)
    d["high24"] = (d.transactions_last_24h >= 11).astype(float)
    d["young"] = (d.account_age <= 30).astype(float)
    d["amt_to_avg_ticket"] = d.log_amt - np.log1p(d.spend_last_24h / d.transactions_last_24h.clip(lower=1))
    d["share_1h"] = d.transactions_last_1h / d.transactions_last_24h.clip(lower=1)
    d["risky_merchant"] = d.merchant_category.isin(["luxury", "cash_transfer", "electronics"]).astype(int)
    d["night_x_device"] = d.night * d.new_device
    d["n_flags"] = d[C.FLAG_COLS].fillna(0).sum(axis=1)
    d["log_tx_1h"] = np.log1p(d.transactions_last_1h)
    d["log_tx_24h"] = np.log1p(d.transactions_last_24h)
    for c in C.CATS:
        d[c] = d[c].fillna("Unknown").astype("category")
    return d


def make_features(frame, extra=False):
    """Feature table for any raw frame (with or without `id`/`fraud`), using the training category lists."""
    X = engineer(frame.drop(columns=[C.ID, C.TARGET], errors="ignore"))
    for c in C.CATS:
        X[c] = pd.Categorical(X[c].astype(str), categories=C.CATEGORIES[c])
    return X[C.FEATURES + (EXTRA if extra else [])]


def as_catboost(X):
    """CatBoost takes categoricals as text."""
    return X[C.FEATURES].astype({c: str for c in C.CATS})
