"""Model zoo: every model family the project tried, behind one interface.

Each base learner is built by `make(name, seed)` and exposes `.fit(X, y)` and `.score(X)` (fraud probability).
`X` is the frame from `features.make_features(..., extra=True)`. Blends are rank averages of base learners,
and are defined in `CANDIDATES`.

Settings are the ones tuned in the experiment notebooks (see notebooks/README.md for where each came from).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C
from .features import as_catboost

NUMERIC_FEATURES = [c for c in C.FEATURES if c not in C.CATS]
LOGIT_COLS = ["transactions_last_24h", "new_device", "log_amt", "log_age", "night", "burst", "high24", "night_x_device"]
NN_COLS = ["log_amt", "log_spend", "log_age", "log_tx_1h", "log_tx_24h", "new_device", "night", "night_x_device",
           "hour_sin", "hour_cos", "amt_to_avg_ticket", "share_1h"]


def _p1(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


class SklearnModel:
    """Wraps a scikit-learn estimator (pipeline) that takes the feature frame directly."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def score(self, X):
        return _p1(self.estimator, X)


class CatBoostModel:
    """The final model's learner: depth-4 CatBoost on the 24 features, categoricals as text."""

    def __init__(self, seed=0, **params):
        from catboost import CatBoostClassifier
        self.estimator = CatBoostClassifier(**{**C.CATBOOST_PARAMS, **params}, cat_features=C.CATS, random_seed=seed,
                                            verbose=0, allow_writing_files=False, thread_count=C.THREADS)

    def fit(self, X, y):
        self.estimator.fit(as_catboost(X), y)
        return self

    def score(self, X):
        return self.estimator.predict_proba(as_catboost(X))[:, 1]


class LightGBMModel(SklearnModel):
    """Tiny-leaf LightGBM tuned by a 70-config random search (5x2 CV), unweighted."""

    def __init__(self, seed=0):
        import lightgbm as lgb
        super().__init__(lgb.LGBMClassifier(
            n_estimators=900, learning_rate=0.03, num_leaves=4, min_child_samples=150, colsample_bytree=0.85,
            reg_lambda=10, reg_alpha=0.5, min_split_gain=0.01, cat_smooth=100, min_data_per_group=50,
            random_state=seed, verbose=-1, n_jobs=C.THREADS))

    def fit(self, X, y):
        self.estimator.fit(X[C.FEATURES], y)
        return self

    def score(self, X):
        return _p1(self.estimator, X[C.FEATURES])


class XGBoostModel(SklearnModel):
    """Shallow XGBoost (depth 3) with native categoricals, unweighted (exact class weighting cost 0.02 PR-AUC)."""

    def __init__(self, seed=0):
        import xgboost as xgb
        super().__init__(xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=3, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1, reg_lambda=5, objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            enable_categorical=True, random_state=seed, n_jobs=C.THREADS))

    def fit(self, X, y):
        self.estimator.fit(X[C.FEATURES], y)
        return self

    def score(self, X):
        return _p1(self.estimator, X[C.FEATURES])


def logistic(seed=0):
    """L2 logistic regression on 8 lean numeric features + 3 one-hot categoricals, C=0.3, unweighted."""
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), LOGIT_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), C.CATS)])
    return SklearnModel(make_pipeline(pre, LogisticRegression(C=0.3, max_iter=3000)))


def random_forest(seed=0):
    """Bagged trees with large leaves (>= 25 rows) and 35% feature sampling."""
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), C.CATS)])
    return SklearnModel(make_pipeline(pre, RandomForestClassifier(
        n_estimators=600, min_samples_leaf=25, max_features=0.35, n_jobs=C.THREADS, random_state=seed)))


def _nn_preprocessor():
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler(clip=True), StandardScaler()), NN_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CATS)])


class InContextKNN:
    """TabPFN-like in-context scoring: smoothed fraud share of the k nearest training rows,
    in a space weighted by logistic-regression coefficient size."""

    def __init__(self, seed=0, k=150, prior=20):
        self.k, self.prior = k, prior

    def fit(self, X, y):
        self.pre = _nn_preprocessor().fit(X)
        Z = self.pre.transform(X)
        coef = LogisticRegression(C=0.3, max_iter=3000).fit(Z, y).coef_[0]
        self.w = np.abs(coef) / np.abs(coef).mean()
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(Z * self.w)
        self.y, self.base = np.asarray(y), float(np.mean(y))
        return self

    def score(self, X):
        _, idx = self.nn.kneighbors(self.pre.transform(X) * self.w)
        return (self.y[idx].sum(axis=1) + self.prior * self.base) / (self.k + self.prior)


# name -> (factory(seed), family, one-line description)
BASE_MODELS = {
    "logistic": (logistic, "linear", "L2 logistic regression, 8 lean features + one-hot categoricals"),
    "random_forest": (random_forest, "bagged trees", "Random forest, 600 trees, min leaf 25"),
    "lightgbm": (LightGBMModel, "boosted trees", "LightGBM, 4 leaves, min 150 rows per leaf"),
    "xgboost": (XGBoostModel, "boosted trees", "XGBoost, depth 3, native categoricals"),
    "knn": (InContextKNN, "instance-based", "In-context kNN (k=150), TabPFN-like"),
    "catboost": (CatBoostModel, "boosted trees", "CatBoost, depth 4, lr 0.03 x 800, L2 10 (final spec)"),
}
SEEDED = {"catboost"}     # learners averaged over several seeds (the others are fitted once)


def make(name, seed=0):
    return BASE_MODELS[name][0](seed)


@dataclass
class Candidate:
    """A submission candidate: one base learner, or a rank-average blend of several."""
    name: str
    weights: dict
    public_name: str = None     # row of results/leaderboard.csv with this candidate's public score, if uploaded
    note: str = ""
    members: list = field(init=False)

    def __post_init__(self):
        self.members = list(self.weights)

    @property
    def is_blend(self):
        return len(self.members) > 1


CANDIDATES = [
    Candidate("logistic", {"logistic": 1}),
    Candidate("random_forest", {"random_forest": 1}),
    Candidate("lightgbm", {"lightgbm": 1}, public_name="LightGBM"),
    Candidate("xgboost", {"xgboost": 1}),
    Candidate("knn", {"knn": 1}),
    Candidate("catboost", {"catboost": 1}, public_name="Plain CatBoost, 3 seeds (final)"),
    Candidate("catboost+logistic 50/50", {"catboost": .5, "logistic": .5}, public_name="CatBoost + logistic 50/50"),
    Candidate("catboost+lightgbm+logistic", {"catboost": 1 / 3, "lightgbm": 1 / 3, "logistic": 1 / 3}),
    Candidate("catboost+random_forest 60/40", {"catboost": .6, "random_forest": .4}),
    Candidate("catboost+knn 70/30", {"catboost": .7, "knn": .3}),
]


def rank(s):
    return pd.Series(np.asarray(s, dtype=float)).rank(pct=True).values


def combine(candidate, scores):
    """Score of a candidate from its members' scores (rank average for blends, raw probability otherwise)."""
    if not candidate.is_blend:
        return np.asarray(scores[candidate.members[0]], dtype=float)
    return sum(w * rank(scores[m]) for m, w in candidate.weights.items())


def fit_score(name, X_fit, y_fit, X_score, seeds):
    """Fit one base learner (averaged over `seeds` if it is seeded) and score `X_score`."""
    seeds = seeds if name in SEEDED else [0]
    return np.mean([make(name, s).fit(X_fit, y_fit).score(X_score) for s in seeds], axis=0)
