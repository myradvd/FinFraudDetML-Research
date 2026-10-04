"""fraudlab: the reproducible pipeline behind the Track 2 fraud-detection submission.

    python -m fraudlab all      # eda -> select -> train -> predict -> verify
"""
import os as _os

# Cap BLAS / OpenMP threads before numpy loads. Small matrices gain nothing from many threads, and oversubscription
# next to CatBoost's own pool slows the sklearn models down badly. CatBoost's thread_count is set explicitly and is unaffected.
for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    _os.environ.setdefault(_var, "3")

from .data import load_test, load_train, sanitise
from .features import as_catboost, engineer, make_features

__all__ = ["load_train", "load_test", "sanitise", "engineer", "make_features", "as_catboost"]
