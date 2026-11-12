"""Inference API.

    uvicorn app.api:app --port 8000
    curl -X POST localhost:8000/predict -H 'content-type: application/json' \
         -d '{"flows": [{"<feature>": 1.0, ...}]}'
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netsentinel.pipeline import HybridDetector  # noqa: E402

MODEL_PATH = os.environ.get("NETSENTINEL_MODEL", "models/netsentinel.joblib")
app = FastAPI(title="NetSentinel", version="0.2.0")
_det: HybridDetector | None = None


def detector() -> HybridDetector:
    global _det
    if _det is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(503, f"No model at {MODEL_PATH}. Run scripts/train_pipeline.py")
        _det = HybridDetector.load(MODEL_PATH)
    return _det


class Flows(BaseModel):
    flows: list[dict[str, float]]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": Path(MODEL_PATH).exists()}


@app.get("/features")
def features():
    return {"features": detector().feature_names_}


@app.post("/predict")
def predict(body: Flows):
    det = detector()
    if not body.flows:
        return {"n_flows": 0, "n_alerts": 0, "alerts": []}
    X = pd.DataFrame(body.flows)
    missing = [f for f in det.feature_names_ if f not in X.columns]
    if missing:
        raise HTTPException(422, f"Missing features: {missing[:5]} (+{max(len(missing) - 5, 0)} more)")
    alerts = det.alerts(X)
    return {"n_flows": len(X), "n_alerts": len(alerts), "alerts": alerts}
