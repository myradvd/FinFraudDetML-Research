"""Loading and defensive cleaning (`sanitise`) of the raw transaction tables."""
import numpy as np
import pandas as pd

from . import config as C


def load_train(path=C.TRAIN_CSV):
    return pd.read_csv(path)


def load_test(path=C.TEST_CSV):
    return pd.read_csv(path)


def sanitise(d):
    """Harden a raw table for unseen, messier data. It changes nothing on the competition files.

    - Columns are selected by name: a missing column becomes blank, extra columns are ignored.
    - Text, +-infinity and impossible values (negative amounts/counts/ages, hour outside 0-23,
      `new_device` other than 0/1) become blank.
    - The 1-hour count is capped at the 24-hour count.
    - Category spelling is normalised to training's (" sg" -> "SG"); unseen levels become blank.

    Blanks are not zero-filled: CatBoost handles missing values natively.
    """
    d = d.reindex(columns=C.RAW_COLS).copy()
    for c in C.NUMERIC_RAW:
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in C.NON_NEGATIVE:
        d[c] = d[c].where(d[c] >= 0)
    d["transaction_hour"] = d.transaction_hour.where(d.transaction_hour.between(0, 23))
    d["new_device"] = d.new_device.where(d.new_device.isin([0, 1]))
    over = d.transactions_last_1h > d.transactions_last_24h
    d.loc[over, "transactions_last_1h"] = d.loc[over, "transactions_last_24h"]
    for c in C.CATS:
        d[c] = d[c].astype("string").str.strip().str.lower().map(C.CAT_LEVELS[c]).astype(object)
    return d
