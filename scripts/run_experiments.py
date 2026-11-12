"""Run the full NetSentinel benchmark.

    python scripts/run_experiments.py --config configs/default.yaml
    python scripts/run_experiments.py --synthetic          # quick demo, no download

Experiments
  1. Baselines (RF, XGBoost) on all features
  2. FFNN on all features
  3. FFNN on PCA-reduced features
  4. Autoencoder anomaly detector (trained on benign only)
  5. Zero-day: hold out whole attack families from supervised training and
     measure how often each model still catches them
  6. SHAP explanation of the XGBoost model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netsentinel.data import load_cicids2017, make_synthetic, split  # noqa: E402
from netsentinel.evaluate import (binary_metrics, per_family_detection,  # noqa: E402
                                  plot_confusion, results_table)
from netsentinel.explain import explain_tree_model  # noqa: E402
from netsentinel.features import build_preprocessor  # noqa: E402
from netsentinel.models import (AutoencoderDetector, FFNNClassifier,  # noqa: E402
                                random_forest, xgboost)


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def evaluate(name, model, Xte, s, out, results, n_features):
    t0 = time.perf_counter()
    y_pred = model.predict(Xte)
    infer_ms = (time.perf_counter() - t0) / len(Xte) * 1e3
    y_score = model.predict_proba(Xte)[:, 1]
    m = binary_metrics(s.y_test, y_pred, y_score)
    m["n_features"] = n_features
    m["infer_us_per_flow"] = round(infer_ms * 1e3, 2)
    m["per_family_detection"] = per_family_detection(y_pred, s.fam_test)
    results[name] = m
    plot_confusion(s.y_test, y_pred, name, out / f"cm_{name}.png")
    log(f"{name:<14} F1={m['f1']:.4f}  recall={m['recall']:.4f}  FPR={m['fpr']:.4f}")
    return y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic demo data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true", help="few epochs (for CI / smoke tests)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["seed"]
    if args.fast:
        for k in ("ffnn", "autoencoder"):
            cfg[k]["epochs"] = 3

    run = "synthetic" if args.synthetic else cfg["data"]["source"]
    out = Path(args.out) / f"{run}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    log("Loading data…")
    if args.synthetic or cfg["data"]["source"] == "synthetic":
        ds = make_synthetic(seed=seed)
    else:
        ds = load_cicids2017(cfg["data"]["raw_dir"], cfg["data"]["sample_frac"], seed)
    log(f"{len(ds.X):,} flows, {ds.X.shape[1]} features")
    fam_counts = pd.Series(ds.y_family).value_counts()
    print(fam_counts.to_string())
    s = split(ds, cfg["data"]["test_size"], cfg["data"]["val_size"], seed)

    results: dict[str, dict] = {}

    # ---------------- full features ----------------
    pre = build_preprocessor()
    Xtr, Xva, Xte = pre.fit_transform(s.X_train), pre.transform(s.X_val), pre.transform(s.X_test)
    n_full = Xtr.shape[1]

    log("Random Forest (all features)")
    rf = random_forest(seed).fit(Xtr, s.y_train)
    evaluate("rf_full", rf, Xte, s, out, results, n_full)

    log("XGBoost (all features)")
    xgb = xgboost(s.y_train, seed).fit(Xtr, s.y_train)
    evaluate("xgb_full", xgb, Xte, s, out, results, n_full)

    log("FFNN (all features)")
    ffnn = FFNNClassifier(**cfg["ffnn"], seed=seed).fit(Xtr, s.y_train, Xva, s.y_val)
    evaluate("ffnn_full", ffnn, Xte, s, out, results, n_full)

    # ---------------- PCA-reduced ----------------
    pre_pca = build_preprocessor("pca", cfg["reduction"]["pca_variance"])
    Ptr = pre_pca.fit_transform(s.X_train)
    Pva, Pte = pre_pca.transform(s.X_val), pre_pca.transform(s.X_test)
    n_pca = Ptr.shape[1]
    log(f"PCA kept {n_pca}/{n_full} components "
        f"({cfg['reduction']['pca_variance']:.0%} variance)")

    log("FFNN (PCA features)")
    ffnn_pca = FFNNClassifier(**cfg["ffnn"], seed=seed).fit(Ptr, s.y_train, Pva, s.y_val)
    evaluate("ffnn_pca", ffnn_pca, Pte, s, out, results, n_pca)

    # ---------------- autoencoder (benign-only) ----------------
    log("Autoencoder (trained on benign only)")
    ben_tr, ben_va = s.y_train == 0, s.y_val == 0
    pre_ae = build_preprocessor().fit(s.X_train[ben_tr])  # never sees attack stats
    ae = AutoencoderDetector(**cfg["autoencoder"], seed=seed).fit(
        pre_ae.transform(s.X_train[ben_tr]), pre_ae.transform(s.X_val[ben_va]))
    evaluate("autoencoder", ae, pre_ae.transform(s.X_test), s, out, results, n_full)

    # ---------------- zero-day simulation ----------------
    holdout = [f for f in cfg["zero_day"]["holdout_families"] if f in set(s.fam_train)]
    zero_day = {}
    if holdout:
        log(f"Zero-day: retraining supervised models WITHOUT {holdout}")
        keep_tr = ~np.isin(s.fam_train, holdout)
        keep_va = ~np.isin(s.fam_val, holdout)
        pre_z = build_preprocessor().fit(s.X_train[keep_tr])
        Ztr, Zva = pre_z.transform(s.X_train[keep_tr]), pre_z.transform(s.X_val[keep_va])
        Zte = pre_z.transform(s.X_test)
        xgb_z = xgboost(s.y_train[keep_tr], seed).fit(Ztr, s.y_train[keep_tr])
        ffnn_z = FFNNClassifier(**cfg["ffnn"], seed=seed, verbose=False).fit(
            Ztr, s.y_train[keep_tr], Zva, s.y_val[keep_va])
        preds = {"xgb (unseen)": xgb_z.predict(Zte),
                 "ffnn (unseen)": ffnn_z.predict(Zte),
                 "autoencoder": ae.predict(pre_ae.transform(s.X_test)),
                 "xgb (seen, reference)": xgb.predict(Xte)}
        for name, p in preds.items():
            det = per_family_detection(p, s.fam_test)
            zero_day[name] = {f: det.get(f) for f in holdout + ["Benign"]}
        zd = pd.DataFrame(zero_day).T
        zd.columns = [f"{c} detection" if c != "Benign" else "Benign FPR" for c in zd.columns]
        print(zd.to_string())

    # ---------------- explainability ----------------
    log("SHAP explanation of XGBoost")
    feat_names = list(s.X_train.columns)
    shap_top = explain_tree_model(xgb, pd.DataFrame(Xte, columns=feat_names), out,
                                  cfg["explain"]["shap_samples"])

    # ---------------- report ----------------
    table = results_table(results)
    table.to_csv(out / "results.csv")
    (out / "metrics.json").write_text(json.dumps(
        {"results": results, "zero_day": zero_day, "shap_top": shap_top.to_dict(),
         "class_counts": fam_counts.to_dict(), "config": cfg}, indent=2, default=str))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    table[["precision", "recall", "f1"]].astype(float).plot.bar(ax=ax, rot=0)
    ax.set_ylim(max(0, float(table[["precision", "recall", "f1"]].min().min()) - 0.05), 1.0)
    ax.set_title("Model comparison (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)

    md = [f"# NetSentinel results — {run}", "",
          f"{len(ds.X):,} flows · {n_full} features · PCA → {n_pca} components", "",
          "## Model comparison", "", table.to_markdown(), ""]
    if zero_day:
        md += ["## Zero-day simulation", "",
               f"Families held out of supervised training: {', '.join(holdout)}", "",
               zd.to_markdown(), ""]
    md += ["## Top SHAP features (XGBoost)", "", shap_top.to_markdown(), ""]
    (out / "results.md").write_text("\n".join(md))
    log(f"Done → {out}")


if __name__ == "__main__":
    main()
