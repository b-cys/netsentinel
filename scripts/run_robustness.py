"""Phase 2: how does the detector hold up outside the comfortable benchmark setting?

    python scripts/run_robustness.py --config configs/default.yaml
    python scripts/run_robustness.py --synthetic            # quick demo, no download

Experiments
  A. Hybrid detector at a fixed alert budget, with attack families held out
     (classifier alone vs autoencoder alone vs both, all at the same target FPR)
  B. Evasion: constrained PGD against the FFNN, transfer to XGBoost and the
     autoencoder, and adversarial training as a defence
  C. Time: train on the first days, test on later days (no shuffling), with a
     feature-drift report (PSI)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netsentinel.adversarial import evade, mutable_mask  # noqa: E402
from netsentinel.data import (load_cicids2017, make_synthetic, split,  # noqa: E402
                              temporal_split)
from netsentinel.evaluate import (per_family_detection, psi,  # noqa: E402
                                  threshold_at_fpr)
from netsentinel.features import build_preprocessor  # noqa: E402
from netsentinel.models import FFNNClassifier, xgboost  # noqa: E402
from netsentinel.pipeline import HybridDetector  # noqa: E402


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def hybrid_experiment(s, cfg, seed) -> pd.DataFrame:
    """A. Same alert budget for everyone; some families never seen in training."""
    fpr = cfg["operating_point"]["target_fpr"]
    holdout = [f for f in cfg["zero_day"]["holdout_families"] if f in set(s.fam_train)]
    keep_tr, keep_va = ~np.isin(s.fam_train, holdout), ~np.isin(s.fam_val, holdout)
    det = HybridDetector(fpr, cfg["autoencoder"], seed).fit(
        s.X_train[keep_tr], s.y_train[keep_tr], s.fam_train[keep_tr],
        s.X_val[keep_va], s.y_val[keep_va])

    # single detectors get the whole budget, the hybrid splits it in half
    ben_va = s.X_val[keep_va][s.y_val[keep_va] == 0]
    p_val, a_val = det._raw_scores(ben_va)
    p, a = det._raw_scores(s.X_test)
    preds = {"classifier only": p > threshold_at_fpr(p_val, fpr),
             "autoencoder only": a > threshold_at_fpr(a_val, fpr),
             "hybrid (both)": det.predict(s.X_test).astype(bool)}
    rows = {}
    for name, pred in preds.items():
        d = per_family_detection(pred.astype(int), s.fam_test)
        seen = [f for f in d if f != "Benign" and f not in holdout]
        seen_mask = np.isin(s.fam_test, seen)
        rows[name] = {"Benign FPR": d.get("Benign"),
                      "seen attacks": round(float(pred[seen_mask].mean()), 4),
                      **{f"{f} (unseen)": d[f] for f in holdout}}
    return pd.DataFrame(rows).T, det


def adversarial_experiment(s, cfg, seed, det) -> tuple[pd.DataFrame, int]:
    """B. Attack flows try to look benign by changing attacker-controlled features."""
    a = cfg["adversarial"]
    mask = mutable_mask(list(s.X_train.columns), a["mutable_keywords"])
    pre = build_preprocessor().fit(s.X_train)
    Xtr, Xva, Xte = pre.transform(s.X_train), pre.transform(s.X_val), pre.transform(s.X_test)
    atk = Xte[s.y_test == 1]

    ffnn = FFNNClassifier(**cfg["ffnn"], seed=seed, verbose=False).fit(Xtr, s.y_train, Xva, s.y_val)
    robust = FFNNClassifier(**cfg["ffnn"], seed=seed, verbose=False,
                            adv={"mask": mask, "eps": a["eps"], "steps": 5}
                            ).fit(Xtr, s.y_train, Xva, s.y_val)
    xgb = xgboost(s.y_train, seed).fit(Xtr, s.y_train)

    adv = evade(ffnn, atk, mask, a["eps"], a["steps"])          # white-box vs FFNN
    adv_r = evade(robust, atk, mask, a["eps"], a["steps"])      # white-box vs robust FFNN
    # Back to raw feature space so detectors with their own scaler can score the same flows
    adv_raw = pd.DataFrame(pre.inverse_transform(adv), columns=s.X_train.columns)
    ae_flags = det.score(adv_raw)["by_anomaly"].to_numpy()
    clean_raw = s.X_test[s.y_test == 1]

    ben = Xte[s.y_test == 0]
    rows = {
        "FFNN": {"clean recall": ffnn.predict(atk).mean(),
                 "recall under attack": ffnn.predict(adv).mean(),
                 "attack": "white-box PGD", "Benign FPR": ffnn.predict(ben).mean()},
        "FFNN + adversarial training": {
            "clean recall": robust.predict(atk).mean(),
            "recall under attack": robust.predict(adv_r).mean(),
            "attack": "white-box PGD", "Benign FPR": robust.predict(ben).mean()},
        "XGBoost": {"clean recall": xgb.predict(atk).mean(),
                    "recall under attack": xgb.predict(adv).mean(),
                    "attack": "transfer from FFNN", "Benign FPR": xgb.predict(ben).mean()},
        "Autoencoder (in hybrid)": {
            "clean recall": det.score(clean_raw)["by_anomaly"].mean(),
            "recall under attack": ae_flags.mean(),
            "attack": "transfer from FFNN",
            "Benign FPR": det.score(s.X_test[s.y_test == 0])["by_anomaly"].mean()},
    }
    df = pd.DataFrame(rows).T
    num = ["clean recall", "recall under attack", "Benign FPR"]
    df[num] = df[num].astype(float).round(4)
    return df, int(mask.sum())


def temporal_experiment(ds, cfg, seed) -> tuple[pd.DataFrame, pd.Series, dict]:
    """C. Train on the past, test on the future."""
    days = cfg["temporal"]["train_days"]
    s = temporal_split(ds, days, seed=seed)
    fpr = cfg["operating_point"]["target_fpr"]
    det = HybridDetector(fpr, cfg["autoencoder"], seed).fit(
        s.X_train, s.y_train, s.fam_train, s.X_val, s.y_val)
    sc = det.score(s.X_test)
    seen = set(s.fam_train)
    rows = {}
    for name, col in (("classifier", "by_classifier"), ("autoencoder", "by_anomaly"),
                      ("hybrid", "alert")):
        rows[name] = per_family_detection(sc[col].to_numpy().astype(int), s.fam_test)
    df = pd.DataFrame(rows)
    df.index = [f if f == "Benign" else f"{f} ({'seen' if f in seen else 'NEW'})"
                for f in df.index]
    df = df.rename(index={"Benign": "Benign FPR"})
    df.insert(0, "test flows", pd.Series(s.fam_test).value_counts().rename(
        lambda f: "Benign FPR" if f == "Benign" else f"{f} ({'seen' if f in seen else 'NEW'})"))

    ben_tr, ben_te = s.X_train[s.y_train == 0], s.X_test[s.y_test == 0]
    drift = pd.Series({c: psi(ben_tr[c].to_numpy(), ben_te[c].to_numpy())
                       for c in s.X_train.columns}).sort_values(ascending=False)
    info = {"train_flows": len(s.X_train), "test_flows": len(s.X_test),
            "target_fpr": fpr, "drifted_features": int((drift > 0.25).sum())}
    return df, drift.head(10).round(3), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["seed"]
    if args.fast:
        for k in ("ffnn", "autoencoder"):
            cfg[k]["epochs"] = 3
    run = "synthetic" if args.synthetic else cfg["data"]["source"]
    out = Path(args.out) / f"robustness_{run}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    log("Loading data…")
    if args.synthetic or cfg["data"]["source"] == "synthetic":
        ds = make_synthetic(seed=seed)
    else:
        ds = load_cicids2017(cfg["data"]["raw_dir"], cfg["data"]["sample_frac"], seed)
    s = split(ds, cfg["data"]["test_size"], cfg["data"]["val_size"], seed)
    fpr = cfg["operating_point"]["target_fpr"]

    log("A. Hybrid detector at a fixed alert budget")
    hyb, det = hybrid_experiment(s, cfg, seed)
    print(hyb.to_string())

    log("B. Evasion attacks and adversarial training")
    adv, n_mut = adversarial_experiment(s, cfg, seed, det)
    print(adv.to_string())

    log("C. Train on the past, test on the future")
    tmp, drift, info = temporal_experiment(ds, cfg, seed)
    print(tmp.to_string())

    a = cfg["adversarial"]
    md = [f"# NetSentinel robustness — {run}", "",
          f"## A. Hybrid detector at {fpr:.1%} target FPR", "",
          f"Held out of training: {', '.join(cfg['zero_day']['holdout_families'])}. "
          "Values are the share of flows flagged.", "", hyb.to_markdown(), "",
          "## B. Evasion", "",
          f"Constrained PGD: {n_mut} attacker-controlled features, increase-only, "
          f"eps={a['eps']} (standardised units), {a['steps']} steps.", "",
          adv.to_markdown(), "",
          "## C. Train on the past, test on the future", "",
          f"Train days {cfg['temporal']['train_days']} ({info['train_flows']:,} flows) → "
          f"later days ({info['test_flows']:,} flows). Thresholds fixed on validation at "
          f"{fpr:.1%} target FPR.", "", tmp.to_markdown(), "",
          f"Benign feature drift (PSI, train days vs later days); "
          f"{info['drifted_features']} features above 0.25:", "",
          drift.rename("PSI").to_markdown(), ""]
    (out / "robustness.md").write_text("\n".join(md))
    (out / "robustness.json").write_text(json.dumps(
        {"hybrid": hyb.to_dict(), "adversarial": adv.to_dict(), "temporal": tmp.to_dict(),
         "drift": drift.to_dict(), "info": info}, indent=2, default=str))
    log(f"Done → {out}")


if __name__ == "__main__":
    main()
