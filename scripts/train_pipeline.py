"""Train the deployable hybrid detector and save it for the API / dashboard.

    python scripts/train_pipeline.py --synthetic
    python scripts/train_pipeline.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netsentinel.data import load_cicids2017, make_synthetic, split  # noqa: E402
from netsentinel.evaluate import binary_metrics  # noqa: E402
from netsentinel.pipeline import HybridDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="models/netsentinel.joblib")
    ap.add_argument("--replay", default="models/replay_sample.csv",
                    help="held-out flows saved for the dashboard replay")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.synthetic or cfg["data"]["source"] == "synthetic":
        ds = make_synthetic(seed=cfg["seed"])
    else:
        ds = load_cicids2017(cfg["data"]["raw_dir"], cfg["data"]["sample_frac"], cfg["seed"])
    s = split(ds, cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["seed"])

    det = HybridDetector(cfg["operating_point"]["target_fpr"], cfg["autoencoder"],
                         cfg["seed"], verbose=True)
    det.fit(s.X_train, s.y_train, s.fam_train, s.X_val, s.y_val)
    print("test:", binary_metrics(s.y_test, det.predict(s.X_test)))
    det.save(args.out)

    sample = s.X_test.copy()
    sample["true_family"] = s.fam_test
    sample.sample(min(5000, len(sample)), random_state=cfg["seed"]).to_csv(args.replay, index=False)
    print(f"saved {args.out} and {args.replay}")


if __name__ == "__main__":
    main()
