"""Metrics, per-family detection rates and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def binary_metrics(y_true, y_pred, y_score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        # False-positive rate matters most in a SOC: every FP is an analyst's time.
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        m["roc_auc"] = roc_auc_score(y_true, y_score)
    return {k: round(float(v), 4) for k, v in m.items()}


def per_family_detection(y_pred, families) -> dict:
    """Share of each family flagged as attack (for Benign this equals the FPR)."""
    df = pd.DataFrame({"pred": y_pred, "family": families})
    return df.groupby("family")["pred"].mean().round(4).to_dict()


def plot_confusion(y_true, y_pred, title: str, path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Benign", "Attack"])
    ax.set_yticks([0, 1], ["Benign", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    cols = ["accuracy", "precision", "recall", "f1", "f1_macro", "fpr", "roc_auc", "n_features"]
    df = pd.DataFrame(results).T
    return df[[c for c in cols if c in df.columns]]


def threshold_at_fpr(benign_scores, target_fpr: float) -> float:
    """Score threshold that lets through at most ``target_fpr`` of benign flows.

    A SOC has a fixed alert budget, so models are compared at the same
    false-positive rate instead of the arbitrary 0.5 cut-off.
    """
    return float(np.quantile(np.asarray(benign_scores), 1 - target_fpr))


def detection_at_fpr(y_true, scores, benign_val_scores, target_fpr: float) -> dict:
    """Recall and realised FPR on test when the threshold is fixed on validation."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    thr = threshold_at_fpr(benign_val_scores, target_fpr)
    pred = scores > thr
    return {"threshold": thr,
            "recall": round(float(pred[y_true == 1].mean()), 4) if (y_true == 1).any() else None,
            "fpr": round(float(pred[y_true == 0].mean()), 4) if (y_true == 0).any() else None}


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of one feature (>0.25 = significant drift)."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))
