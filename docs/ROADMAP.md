# Roadmap

## Phase 1: Research core ✅
- Cleaning pipeline for CIC-IDS2017's known data-quality problems
- RF / XGBoost baselines, FFNN, FFNN + PCA, benign-only autoencoder
- Zero-day simulation (whole families withheld from supervised training)
- FPR and per-family detection reporting, SHAP explanations

## Phase 2: Robustness ✅
- `HybridDetector`: classifier OR autoencoder, each calibrated to half of a fixed FPR budget
- Constrained PGD evasion (attacker-controlled features, increase-only, bounded), transfer to XGBoost and the autoencoder, adversarial training
- Train-on-past / test-on-future split by capture day, PSI drift report
- Explained alerts mapped to MITRE ATT&CK
- FastAPI service, Streamlit dashboard, Docker Compose

## Phase 3: Attack lab
- Isolated host-only network: Kali, Metasploitable 2, Ubuntu victim (no internet bridge)
- Generate benign traffic plus nmap scans, Hydra SSH/FTP brute force, slowloris/hping3 DoS, DVWA web attacks
- Capture PCAPs → flows with CICFlowMeter or nfstream → label from attack timestamps
- Run the trained detector on this self-generated traffic and report the gap to the benchmark
- Replace the dashboard replay with live capture; measure flows/sec and latency

## Phase 4: Generalization
- Map CIC-IDS2017 and CSE-CIC-IDS2018 to a shared feature set (both use CICFlowMeter)
- Train on 2017 → test on 2018 and report the drop
- UNSW-NB15 as a harder transfer test (different feature extractor)
- Retrain when PSI drift crosses a threshold; compare against a frozen model

## Phase 5: Context
- Per-host windows of consecutive flows: LSTM / 1D-CNN, then a host-to-host graph model
- Problem-space evasion: replay modified PCAPs through the lab instead of perturbing features
- LLM-written alert summaries from the alert JSON (family, ATT&CK technique, SHAP reasons)
- Ship alerts to Wazuh or ELK; side-by-side with Suricata rules
