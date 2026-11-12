# NetSentinel 🛡️

**Deep-learning network intrusion detection, benchmarked honestly against classical ML and tested on attacks it has never seen.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://github.com/b-cys/netsentinel/actions/workflows/ci.yml/badge.svg)

Most IDS projects train a classifier on one dataset, report 99% accuracy and stop there. NetSentinel asks the questions a SOC team would ask instead:

- **Does deep learning actually beat a well-tuned XGBoost?** Every DL model is benchmarked against strong classical baselines.
- **What happens with a brand-new attack type?** Whole attack families are removed from training to simulate zero-day attacks, and the models are scored on how many they still catch.
- **How many false alarms will analysts drown in?** The false-positive rate is reported next to every model, not just accuracy.
- **Why was this flow flagged?** SHAP explanations show which traffic features drove each decision.
- **Can an attacker talk their way past it?** Attack flows are perturbed with a constrained PGD evasion attack, and adversarial training is tested as a defence.
- **Does it survive time?** Models are trained on the first capture days and tested on later ones, with no shuffling across days.

---

## Pipeline

```
 CIC-IDS2017 flows ──► cleaning ──► signed-log + scaling ──┬──► RF / XGBoost        (baselines)
 (inf/NaN, dupes,       (fit on train only)                ├──► FFNN                (supervised DL)
  constant cols)                                           ├──► PCA ─► FFNN         (feature reduction)
                                                           └──► Autoencoder         (benign-only anomaly detection)
                                                                     │
          metrics · per-family detection · zero-day test · SHAP ◄────┘
```

| Component | Why it's there |
|---|---|
| **Signed log1p + StandardScaler** | Flow features (bytes, durations) are extremely heavy-tailed. |
| **Stratified split by attack family** | Rare families (Heartbleed, Infiltration) still show up in the test set. |
| **Class-weighted losses** | Benign traffic is ~80% of CIC-IDS2017. |
| **PCA (95% variance)** | Measures how much the input can be compressed without losing detection performance. |
| **Autoencoder** | Trained only on benign traffic, so it can flag attack types it has never seen. The threshold is set to a ~1% false-positive budget. |
| **Zero-day simulation** | PortScan and WebAttack are removed from supervised training, then each model's detection rate on them is measured. |

## The hybrid detector

A supervised classifier is precise on attacks it has seen and close to blind on new ones. A benign-only autoencoder is the opposite. `HybridDetector` runs both, gives each half of a fixed alert budget (1% FPR by default, calibrated on benign validation traffic), and raises an alert if either fires.

Every alert is a JSON record an analyst can act on:

```json
{"kind": "known-attack", "severity": "critical", "family": "PortScan",
 "mitre_attack": {"technique": "T1046", "name": "Network Service Discovery", "tactic": "Discovery"},
 "p_attack": 0.9989, "anomaly_ratio": 1.018, "detectors": ["classifier", "autoencoder"],
 "top_features": [{"feature": "...", "value": 24.72, "shap": 3.1}]}
```

Flows that only the autoencoder flags are reported as `"kind": "anomaly"` with no family: the honest answer for something the classifier has never seen.

## Quickstart

```bash
git clone https://github.com/b-cys/netsentinel.git
cd netsentinel
pip install -r requirements.txt

# 1) smoke-test everything on synthetic data (no download, ~30s)
python scripts/run_experiments.py --synthetic

# 2) run the real benchmark (see "Dataset" below)
python scripts/run_experiments.py --config configs/default.yaml

# 3) robustness: hybrid at fixed FPR, evasion attacks, train-on-past/test-on-future
python scripts/run_robustness.py --synthetic

# 4) train the deployable detector, then serve it
python scripts/train_pipeline.py --synthetic
uvicorn app.api:app --port 8000          # POST /predict
streamlit run app/dashboard.py           # alert dashboard (replays held-out flows)
# or: docker compose up

pytest -q
```

Each run writes to `results/<run>_<timestamp>/`:
`results.md` · `results.csv` · `metrics.json` · `comparison.png` · `cm_*.png` · `shap_summary.png` · `robustness.md`

## Dataset

**CIC-IDS2017** from the Canadian Institute for Cybersecurity: 5 days of realistic traffic with labelled DoS/DDoS, PortScan, brute-force, web attacks, botnet, infiltration and Heartbleed.

1. Download `MachineLearningCSV.zip` from <https://www.unb.ca/cic/datasets/ids-2017.html>
2. Extract the 8 CSVs into `data/raw/CIC-IDS2017/`
3. `sample_frac` in `configs/default.yaml` controls how much is used (0.3 by default to keep it laptop-friendly)

**Use the corrected labels if you can.** The original CSVs have documented labelling and flow-construction errors (Engelen et al., *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*, 2021; Liu et al., *Error Prevalence in NIDS Datasets*, 2022; Lanvin et al., 2022). The authors of the first two published a regenerated version of the dataset. The loader treats its `"<attack> - Attempted"` labels as benign.

Known data-quality issues handled in `data.py`: `inf`/`NaN` values in rate columns, duplicate rows, constant columns, a duplicated `Fwd Header Length.1` column, whitespace in column names, and a broken character in the "Web Attack" labels.

## Results

> Fill in after running on CIC-IDS2017. Copy the tables from `results/<run>/results.md`.

| Model | Features | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Random Forest | all | – | – | – | – |
| XGBoost | all | – | – | – | – |
| FFNN | all | – | – | – | – |
| FFNN | PCA | – | – | – | – |
| Autoencoder | all | – | – | – | – |

**Zero-day detection (families never seen in training):** – 

**Key findings:** –

### Robustness

> Copy from `results/robustness_<run>/robustness.md` after running on CIC-IDS2017.

| Question | Result |
|---|---|
| Hybrid vs classifier alone on unseen families (same FPR) | – |
| FFNN recall under constrained PGD, before / after adversarial training | – |
| Train Mon–Wed → test Thu–Fri: detection of new families, benign FPR | – |

**Limits worth stating:** the evasion attack works in feature space (bounded, increase-only changes to attacker-controlled features), so it is a lower bound on robustness, not a packet-level attack. CIC-IDS2017 is lab traffic from 2017; results on it say little about a production network until Phase 3.

## Project structure

```
netsentinel/
├── configs/default.yaml        # all hyper-parameters in one place
├── scripts/
│   ├── run_experiments.py      # benchmark: baselines vs DL, zero-day, SHAP
│   ├── run_robustness.py       # hybrid @ fixed FPR, evasion, temporal split
│   └── train_pipeline.py       # train + save the deployable detector
├── app/
│   ├── api.py                  # FastAPI inference service
│   └── dashboard.py            # Streamlit alert dashboard
├── src/netsentinel/
│   ├── data.py                 # loading, cleaning, family mapping, splitting
│   ├── features.py             # preprocessing + PCA/SVD
│   ├── models/
│   │   ├── baselines.py        # Random Forest, XGBoost
│   │   ├── ffnn.py             # PyTorch FFNN, early stopping
│   │   └── autoencoder.py      # benign-only anomaly detector
│   ├── pipeline.py             # HybridDetector (classifier OR autoencoder)
│   ├── adversarial.py          # constrained PGD evasion
│   ├── alerts.py               # alert records + MITRE ATT&CK mapping
│   ├── evaluate.py             # metrics, FPR budgets, per-family detection, PSI drift
│   └── explain.py              # SHAP
├── tests/                      # pytest suite (runs in CI)
└── docs/ROADMAP.md
```

## Roadmap

- [x] **Phase 1: Research core.** Baselines, FFNN, PCA, autoencoder, zero-day simulation, SHAP
- [x] **Phase 2: Robustness.** Hybrid detector at a fixed alert budget, constrained evasion attacks + adversarial training, temporal evaluation + drift report, ATT&CK-mapped alerts, API + dashboard + Docker
- [ ] **Phase 3: Attack lab.** Isolated VM lab (Kali → Metasploitable), self-generated attack PCAPs, live capture with nfstream, evaluation on that traffic
- [ ] **Phase 4: Generalization.** Cross-dataset evaluation (CIC-IDS2017 → CSE-CIC-IDS2018 / UNSW-NB15), drift-triggered retraining
- [ ] **Phase 5: Context.** Sequence/graph models over flows per host, LLM-written alert summaries for analysts, alert export to Wazuh / ELK

See [docs/ROADMAP.md](docs/ROADMAP.md) for details.

## License

MIT
