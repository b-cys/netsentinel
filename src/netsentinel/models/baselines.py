"""Classical ML baselines. A deep model is only worth it if it beats these."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def random_forest(seed: int = 42, **kw) -> RandomForestClassifier:
    params = dict(n_estimators=200, n_jobs=-1, class_weight="balanced_subsample",
                  random_state=seed)
    params.update(kw)
    return RandomForestClassifier(**params)


def xgboost(y_train: np.ndarray, seed: int = 42, **kw) -> XGBClassifier:
    pos = max(int(y_train.sum()), 1)
    neg = len(y_train) - pos
    params = dict(n_estimators=300, max_depth=8, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                  scale_pos_weight=neg / pos, eval_metric="logloss",
                  n_jobs=-1, random_state=seed)
    params.update(kw)
    return XGBClassifier(**params)
