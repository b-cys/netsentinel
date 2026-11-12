"""Turn a model score into something an analyst can act on."""
from __future__ import annotations

# Attack family -> MITRE ATT&CK technique an analyst would start from.
ATTACK_MAP = {
    "PortScan": ("T1046", "Network Service Discovery", "Discovery"),
    "BruteForce": ("T1110", "Brute Force", "Credential Access"),
    "DoS": ("T1499", "Endpoint Denial of Service", "Impact"),
    "DDoS": ("T1498", "Network Denial of Service", "Impact"),
    "WebAttack": ("T1190", "Exploit Public-Facing Application", "Initial Access"),
    "Heartbleed": ("T1190", "Exploit Public-Facing Application", "Initial Access"),
    "Bot": ("T1071", "Application Layer Protocol (C2)", "Command and Control"),
    "Infiltration": ("T1021", "Remote Services", "Lateral Movement"),
}


def severity(p_supervised: float, anomaly_ratio: float) -> str:
    if p_supervised >= 0.9 and anomaly_ratio >= 1.0:
        return "critical"
    if p_supervised >= 0.9 or anomaly_ratio >= 3.0:
        return "high"
    return "medium"


def build_alert(flow_id, p_supervised: float, anomaly_ratio: float, by_classifier: bool,
                by_anomaly: bool, family: str | None, reasons: list[dict]) -> dict:
    """One alert record (JSON-serialisable, ready for a SIEM)."""
    if by_classifier and family:
        kind = "known-attack"
    else:
        kind, family = "anomaly", None  # nothing the classifier has seen before
    tid = ATTACK_MAP.get(family) if family else None
    return {
        "flow_id": flow_id,
        "kind": kind,
        "severity": severity(p_supervised, anomaly_ratio),
        "family": family,
        "mitre_attack": ({"technique": tid[0], "name": tid[1], "tactic": tid[2]}
                         if tid else None),
        "p_attack": round(float(p_supervised), 4),
        "anomaly_ratio": round(float(anomaly_ratio), 3),
        "detectors": [d for d, on in (("classifier", by_classifier),
                                      ("autoencoder", by_anomaly)) if on],
        "top_features": reasons,
    }
