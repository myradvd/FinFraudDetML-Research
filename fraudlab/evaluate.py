"""Cross-validation, the alert-share threshold, metrics and bootstrap uncertainty."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import config as C


def folds(y):
    """5-fold stratified split (seeded): every fold keeps the 1.77% fraud rate."""
    return list(StratifiedKFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.CV_RANDOM_STATE).split(np.zeros(len(y)), y))


def pick_share(y, score):
    """Share of rows to flag: best F1 on the out-of-fold scores, smoothed over 5 neighbouring shares."""
    rows = []
    for share in np.arange(*C.SHARE_GRID):
        flag = (score >= np.quantile(score, 1 - share)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, flag).ravel()
        rows.append({"share": share, "TP": tp, "FP": fp, "FN": fn,
                     "precision": tp / max(tp + fp, 1), "recall": tp / (tp + fn), "F1": 2 * tp / (2 * tp + fp + fn)})
    cm = pd.DataFrame(rows)
    cm["F1_smooth"] = cm["F1"].rolling(5, center=True, min_periods=1).mean()   # don't pick a one-off noisy spike
    return cm.loc[cm["F1_smooth"].idxmax()]


def metrics(y, score, probabilities=True):
    """Ranking metrics (PR-AUC, ROC-AUC, precision of the top 100 alerts) and F1/recall at the chosen share."""
    best = pick_share(y, score)
    top100 = np.argsort(-score, kind="stable")[:100]
    out = {"PR-AUC": average_precision_score(y, score), "ROC-AUC": roc_auc_score(y, score),
           "top-100 precision": y[top100].mean(), "flag share": best["share"],
           "precision": best["precision"], "recall": best["recall"], "F1": best["F1"]}
    out["Brier"] = brier_score_loss(y, score) if probabilities else np.nan
    return out


def bootstrap_ap(y, scores, n=1000, seed=0):
    """Bootstrap PR-AUC for several score vectors on the *same* resampled rows (so differences are paired)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    keep = y[idx].sum(axis=1) > 0
    return {name: np.array([average_precision_score(y[i], s[i]) for i in idx[keep]]) for name, s in scores.items()}


def to_submission_score(score, thr):
    """Monotone map to [0, 1] that puts the flag cut-off at 0.5 (order, and so PR-AUC, unchanged)."""
    score = np.asarray(score, dtype=float)
    lo, hi = score.min(), score.max()
    below = 0.49 * (score - lo) / max(thr - lo, 1e-12)
    above = 0.51 + 0.49 * (score - thr) / max(hi - thr, 1e-12)
    return np.clip(np.where(score >= thr, above, below), 0, 1)
