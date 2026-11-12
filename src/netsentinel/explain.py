"""SHAP explanations: *why* was this flow flagged as an attack?"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def explain_tree_model(model, X: pd.DataFrame, out_dir: Path,
                       max_samples: int = 2000, top_k: int = 15) -> pd.Series:
    """Global SHAP importance for a tree model (XGBoost / RF). Saves a summary plot."""
    X = X.sample(min(len(X), max_samples), random_state=42)
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):  # older SHAP returns one array per class
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    importance = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    importance = importance.sort_values(ascending=False)

    shap.summary_plot(sv, X, max_display=top_k, show=False)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    return importance.head(top_k).round(5)
