"""SOC-style dashboard that replays held-out flows through the detector.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netsentinel.pipeline import HybridDetector  # noqa: E402

MODEL, REPLAY = Path("models/netsentinel.joblib"), Path("models/replay_sample.csv")

st.set_page_config(page_title="NetSentinel", page_icon="🛡️", layout="wide")
st.title("🛡️ NetSentinel")
st.caption("Hybrid intrusion detection: supervised classifier + benign-only autoencoder")

if not MODEL.exists() or not REPLAY.exists():
    st.error("No trained model found. Run `python scripts/train_pipeline.py --synthetic` first.")
    st.stop()


@st.cache_resource
def load():
    return HybridDetector.load(MODEL), pd.read_csv(REPLAY)


det, flows = load()
n = st.sidebar.slider("Flows to replay", 100, len(flows), min(1000, len(flows)), step=100)
only = st.sidebar.multiselect("Severity", ["critical", "high", "medium"],
                              default=["critical", "high", "medium"])
batch = flows.head(n)
truth = batch["true_family"]
X = batch.drop(columns=["true_family"])

scores = det.score(X)
alerts = det.alerts(X)
is_attack = truth != "Benign"
flagged = scores["alert"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flows", f"{n:,}")
c2.metric("Alerts", f"{int(flagged.sum()):,}")
c3.metric("Attacks caught", f"{(flagged & is_attack).sum() / max(is_attack.sum(), 1):.1%}")
c4.metric("False-positive rate", f"{(flagged & ~is_attack).sum() / max((~is_attack).sum(), 1):.2%}")

left, right = st.columns(2)
with left:
    st.subheader("Detection rate by true family")
    st.bar_chart(flagged.groupby(truth).mean())
with right:
    st.subheader("Which detector fired")
    st.bar_chart(pd.Series({
        "classifier only": int((scores.by_classifier & ~scores.by_anomaly).sum()),
        "autoencoder only": int((~scores.by_classifier & scores.by_anomaly).sum()),
        "both": int((scores.by_classifier & scores.by_anomaly).sum())}))

st.subheader("Alerts")
table = pd.DataFrame([{
    "flow": a["flow_id"], "severity": a["severity"], "kind": a["kind"],
    "family": a["family"] or "unknown",
    "ATT&CK": a["mitre_attack"]["technique"] if a["mitre_attack"] else "",
    "p(attack)": a["p_attack"], "anomaly ×": a["anomaly_ratio"],
    "why": ", ".join(f"{r['feature']}={r['value']}" for r in a["top_features"]),
    "truth": truth.loc[a["flow_id"]]} for a in alerts if a["severity"] in only])
st.dataframe(table, use_container_width=True, hide_index=True)
