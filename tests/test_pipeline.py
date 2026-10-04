"""Fast checks of the pipeline's contracts (no model training; < 10 s).

Tests marked `needs_data` are skipped when the competition CSVs are absent (e.g. in CI)."""
import numpy as np
import pandas as pd
import pytest

from fraudlab import config as C
from fraudlab.data import data_available, load_test, load_train, sanitise
from fraudlab.evaluate import pick_share, to_submission_score
from fraudlab.features import make_features


needs_data = pytest.mark.skipif(not data_available(), reason="competition CSVs not in data/ (see data/README.md)")


@pytest.fixture(scope="module")
def train():
    return load_train()


@needs_data
def test_schema(train):
    assert train.shape == (20000, 12) and train[C.TARGET].sum() == 353
    assert load_test().shape == (12000, 11)


@needs_data
def test_features_shape_and_order(train):
    X = make_features(train)
    assert list(X.columns) == C.FEATURES and len(X) == len(train)
    for c in C.CATS:
        assert list(X[c].cat.categories) == C.CATEGORIES[c] and X[c].notna().all()


@needs_data
def test_sanitise_is_identity_on_competition_files(train):
    clean = sanitise(train)
    assert clean.isna().sum().sum() == train[C.RAW_COLS].isna().sum().sum()


def test_sanitise_edge_cases():
    messy = pd.DataFrame({
        "transaction_amount": [-5, "abc", np.inf], "transaction_hour": [25, 3, 12],
        "merchant_category": [" Luxury ", "pets", None], "country": ["sg", "XX", "JP"],
        "transaction_channel": ["ECOMMERCE", "card_present", "fax"], "transactions_last_24h": [2, 3, 4],
        "spend_last_24h": [10, 20, 30], "account_age": [-1, 100, 200], "new_device": [2, 1, 0],
        "transactions_last_1h": [5, 1, 1], "unexpected_column": [1, 2, 3],
    }).drop(columns="transaction_channel")                  # a missing column must not crash
    d = sanitise(messy)
    assert list(d.columns) == C.RAW_COLS
    assert d.transaction_amount.isna().all()                # negative, text, infinity -> blank
    assert d.transaction_hour.tolist()[1:] == [3, 12] and np.isnan(d.transaction_hour[0])
    assert d.merchant_category.tolist()[0] == "luxury" and pd.isna(d.merchant_category[1])
    assert d.country.tolist()[0] == "SG" and pd.isna(d.country[1])
    assert d.transaction_channel.isna().all()
    assert d.transactions_last_1h[0] == 2                   # capped at the 24h count
    assert np.isnan(d.new_device[0]) and np.isnan(d.account_age[0])
    X = make_features(messy)                                # and features still build
    assert X.shape == (3, len(C.FEATURES))


def test_submission_map_keeps_order_and_puts_cutoff_at_half():
    rng = np.random.default_rng(0)
    score = rng.random(1000) ** 3
    thr = np.quantile(score, 1 - 0.0165)
    mapped = to_submission_score(score, thr)
    assert ((mapped >= 0) & (mapped <= 1)).all()
    assert (np.argsort(score, kind="stable") == np.argsort(mapped, kind="stable")).all()
    assert ((mapped >= 0.5) == (score >= thr)).all()


def test_pick_share_on_perfect_ranker():
    y = np.r_[np.ones(20), np.zeros(980)].astype(int)
    best = pick_share(y, -np.arange(1000.0))
    assert best["F1"] == pytest.approx(1.0, abs=0.05)
