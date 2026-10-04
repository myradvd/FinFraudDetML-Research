"""Stages 3-5 - train the selected model on all rows, predict the test file, verify reproducibility.

For the final CatBoost, the bundle written here has exactly the layout `final_notebook/predict_catboost.ipynb`
loads, so `results/final/model.pkl` can replace `final_notebook/model.pkl`.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data import load_test, load_train
from .evaluate import to_submission_score
from .features import make_features
from .models import CANDIDATES, combine, make, SEEDED

MODEL_PATH = C.FINAL_DIR / "model.pkl"
SUBMISSION_PATH = C.FINAL_DIR / "submission.csv"


def selected():
    path = C.SELECTION_DIR / "selected_model.json"
    if path.exists():
        return json.loads(path.read_text())
    print(f"  {path.relative_to(C.ROOT)} not found: using the frozen final model '{C.FINAL_MODEL}' (run `select` first to re-derive it)")
    return {"selected": C.FINAL_MODEL, "flag_share": None}


def train():
    sel = selected()
    cand = next(c for c in CANDIDATES if c.name == sel["selected"])
    share = sel["flag_share"]
    if share is None:                                 # frozen final model without a selection run: the shipped share
        share = joblib.load(C.SHIPPED_MODEL)["share"]
    df = load_train()
    X, y = make_features(df, extra=True), df[C.TARGET].values
    fitted = {m: [make(m, s).fit(X, y) for s in (C.FINAL_SEEDS if m in SEEDED else [0])] for m in cand.members}
    if cand.name == "catboost":                       # the final notebook's bundle layout
        bundle = {"models": [m.estimator for m in fitted["catboost"]], "features": C.FEATURES, "cats": C.CATS,
                  "raw_cols": C.RAW_COLS, "non_negative": C.NON_NEGATIVE, "cat_levels": C.CAT_LEVELS,
                  "categories": C.CATEGORIES, "share": share}
    else:
        bundle = {"candidate": cand.name, "fitted": fitted, "share": share}
        print("  note: final_notebook/ only loads a CatBoost bundle; this one is for `fraudlab predict`")
    C.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"  trained '{cand.name}' on {len(y):,} rows, flag share {share:.2%} -> {MODEL_PATH.relative_to(C.ROOT)}")


def score_bundle(bundle, X):
    if "models" in bundle:                            # CatBoost bundle
        Xc = X[C.FEATURES].astype({c: str for c in C.CATS})
        return np.mean([m.predict_proba(Xc)[:, 1] for m in bundle["models"]], axis=0)
    cand = next(c for c in CANDIDATES if c.name == bundle["candidate"])
    return combine(cand, {m: np.mean([f.score(X) for f in fs], axis=0) for m, fs in bundle["fitted"].items()})


def predict(model_path=MODEL_PATH, test_path=C.TEST_CSV, out_path=SUBMISSION_PATH):
    bundle = joblib.load(model_path)
    test = load_test(test_path)
    score = score_bundle(bundle, make_features(test, extra=True))
    score = np.where(np.isfinite(score), score, np.nanmedian(score))   # defensive: a missing score gets the median
    sub = pd.DataFrame({"id": test[C.ID].values,
                        "prediction": to_submission_score(score, np.quantile(score, 1 - bundle["share"]))})
    assert len(sub) == len(test) and sub["id"].is_unique and sub["prediction"].between(0, 1).all()
    if C.SAMPLE_SUBMISSION.exists() and len(sub) == len(pd.read_csv(C.SAMPLE_SUBMISSION)):
        assert (pd.read_csv(C.SAMPLE_SUBMISSION)["id"].values == sub["id"].values).all(), "ids not in sample order"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"  {len(sub):,} predictions -> {out_path.relative_to(C.ROOT)}; flagged (>= 0.5): {(sub.prediction >= 0.5).sum()}")
    return out_path


def verify():
    """Reproducibility checks: the rebuilt model must match the shipped one and the 0.19586 upload, byte for byte."""
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    shipped = joblib.load(C.SHIPPED_MODEL)
    check("shipped bundle constants match fraudlab.config",
          (shipped["features"], shipped["raw_cols"], shipped["non_negative"], shipped["cat_levels"], shipped["categories"])
          == (C.FEATURES, C.RAW_COLS, C.NON_NEGATIVE, C.CAT_LEVELS, C.CATEGORIES))
    check("shipped model hyper-parameters", all(
        {k: m.get_params()[k] for k in C.CATBOOST_PARAMS} == C.CATBOOST_PARAMS for m in shipped["models"])
        and [m.get_params()["random_seed"] for m in shipped["models"]] == C.FINAL_SEEDS)

    shipped_sub = predict(C.SHIPPED_MODEL, out_path=C.FINAL_DIR / "submission_from_shipped_model.csv")
    md5 = hashlib.md5(shipped_sub.read_bytes()).hexdigest()
    check("shipped model reproduces the 0.19586 upload", md5 == C.EXPECTED_SUBMISSION_MD5, f"md5 {md5}")

    if MODEL_PATH.exists():
        rebuilt = joblib.load(MODEL_PATH)
        if "models" in rebuilt:
            X = make_features(load_test(), extra=True)
            diff = np.abs(score_bundle(rebuilt, X) - score_bundle(shipped, X)).max()
            check("retrained model == shipped model (test scores)", diff == 0, f"max |diff| {diff:.2e}")
            check("retrained flag share == shipped", np.isclose(rebuilt["share"], shipped["share"]),
                  f"{rebuilt['share']:.4f} vs {shipped['share']:.4f}")
    if SUBMISSION_PATH.exists():
        md5 = hashlib.md5(SUBMISSION_PATH.read_bytes()).hexdigest()
        check("pipeline submission == 0.19586 upload", md5 == C.EXPECTED_SUBMISSION_MD5, f"md5 {md5}")
    print("  ALL CHECKS PASS" if ok else "  SOME CHECKS FAILED")
    return ok
