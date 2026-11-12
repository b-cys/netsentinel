"""Deployable hybrid detector: supervised classifier OR benign-only autoencoder.

The classifier is precise on attack types it was trained on but close to blind on
new ones; the autoencoder is the opposite. Running both, each calibrated to half
of the alert budget, covers known and unknown attacks at a fixed false-positive rate.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .alerts import build_alert
from .evaluate import threshold_at_fpr
from .features import build_preprocessor
from .models import AutoencoderDetector, xgboost


class HybridDetector:
    def __init__(self, target_fpr: float = 0.01, ae_params: dict | None = None,
                 seed: int = 42, verbose: bool = False):
        self.target_fpr, self.seed, self.verbose = target_fpr, seed, verbose
        self.ae_params = dict(ae_params or {})
        self.ae_params.pop("threshold_percentile", None)

    def fit(self, X_train: pd.DataFrame, y_train, fam_train, X_val: pd.DataFrame, y_val):
        y_train, y_val, fam_train = map(np.asarray, (y_train, y_val, fam_train))
        self.feature_names_ = list(X_train.columns)
        ben_tr, ben_va = y_train == 0, y_val == 0

        # 1) supervised binary classifier
        self.pre_ = build_preprocessor().fit(X_train)
        Xtr = self.pre_.transform(X_train)
        self.clf_ = xgboost(y_train, self.seed).fit(Xtr, y_train)

        # 2) attack-family classifier, for triage only
        atk = y_train == 1
        self.fam_enc_ = LabelEncoder().fit(fam_train[atk])
        self.fam_clf_ = None
        if len(self.fam_enc_.classes_) > 1:
            self.fam_clf_ = XGBClassifier(n_estimators=150, max_depth=6, tree_method="hist",
                                          n_jobs=-1, random_state=self.seed)
            self.fam_clf_.fit(Xtr[atk], self.fam_enc_.transform(fam_train[atk]))

        # 3) autoencoder on benign traffic only (own scaler: never sees attack statistics)
        self.pre_ae_ = build_preprocessor().fit(X_train[ben_tr])
        self.ae_ = AutoencoderDetector(**self.ae_params, seed=self.seed, verbose=self.verbose)
        self.ae_.fit(self.pre_ae_.transform(X_train[ben_tr]),
                     self.pre_ae_.transform(X_val[ben_va]))

        # 4) each detector gets half of the alert budget, measured on benign validation flows
        half = self.target_fpr / 2
        p_val, a_val = self._raw_scores(X_val[ben_va])
        self.clf_threshold_ = threshold_at_fpr(p_val, half)
        self.ae_threshold_ = threshold_at_fpr(a_val, half)
        self._explainer = None
        return self

    def _raw_scores(self, X: pd.DataFrame):
        X = X[self.feature_names_]
        p = self.clf_.predict_proba(self.pre_.transform(X))[:, 1]
        a = self.ae_.score(self.pre_ae_.transform(X))
        return p, a

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        p, a = self._raw_scores(X)
        by_clf, by_ae = p > self.clf_threshold_, a > self.ae_threshold_
        return pd.DataFrame({"p_attack": p, "anomaly_ratio": a / self.ae_threshold_,
                             "by_classifier": by_clf, "by_anomaly": by_ae,
                             "alert": by_clf | by_ae}, index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.score(X)["alert"].to_numpy().astype(int)

    def alerts(self, X: pd.DataFrame, top_k: int = 3) -> list[dict]:
        """Explained alert records for every flagged flow in ``X``."""
        s = self.score(X)
        hits = s[s["alert"]]
        if hits.empty:
            return []
        Xt = self.pre_.transform(X.loc[hits.index, self.feature_names_])
        fams = [None] * len(hits)
        if self.fam_clf_ is not None:
            fams = self.fam_enc_.inverse_transform(self.fam_clf_.predict(Xt))
        elif len(self.fam_enc_.classes_) == 1:
            fams = [self.fam_enc_.classes_[0]] * len(hits)
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.clf_)
        sv = np.asarray(self._explainer.shap_values(Xt))
        raw = X.loc[hits.index, self.feature_names_].to_numpy()
        out = []
        for i, (idx, row) in enumerate(hits.iterrows()):
            top = np.argsort(-np.abs(sv[i]))[:top_k]
            reasons = [{"feature": self.feature_names_[j], "value": round(float(raw[i, j]), 3),
                        "shap": round(float(sv[i, j]), 3)} for j in top]
            out.append(build_alert(idx if not hasattr(idx, "item") else idx.item(),
                                   row.p_attack, row.anomaly_ratio, bool(row.by_classifier),
                                   bool(row.by_anomaly), fams[i], reasons))
        return out

    def save(self, path: str | Path):
        self._explainer = None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "HybridDetector":
        return joblib.load(path)
