"""Feature preprocessing and reduction (fit on train only — no leakage)."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _signed_expm1(x):
    return np.sign(x) * np.expm1(np.abs(x))


def build_preprocessor(reduction: str | None = None,
                       n_components: int | float | None = None) -> Pipeline:
    """Signed log1p (flow features are heavy-tailed) -> standardize -> optional reduction.

    reduction: None | "pca" | "svd". For PCA, n_components may be a float in (0, 1)
    meaning "keep this fraction of explained variance".
    """
    steps = [("log", FunctionTransformer(_signed_log1p, inverse_func=_signed_expm1,
                                         check_inverse=False, feature_names_out="one-to-one")),
             ("scale", StandardScaler())]
    if reduction == "pca":
        steps.append(("reduce", PCA(n_components=n_components, random_state=42)))
    elif reduction == "svd":
        steps.append(("reduce", TruncatedSVD(n_components=int(n_components), random_state=42)))
    elif reduction is not None:
        raise ValueError(f"Unknown reduction: {reduction}")
    return Pipeline(steps)
