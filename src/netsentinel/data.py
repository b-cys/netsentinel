"""Dataset loading, cleaning and splitting.

Supports CIC-IDS2017 (MachineLearningCVE CSVs) plus a synthetic generator
so the full pipeline can be smoke-tested without downloading ~1 GB of data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BENIGN = "BENIGN"

# Map raw CIC-IDS2017 labels to attack families.
FAMILY_MAP = {
    "BENIGN": "Benign",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
}


def to_family(label: str) -> str:
    label = label.strip()
    # The corrected CIC-IDS2017 release (Engelen et al. / Liu et al.) marks attack
    # flows that carried no malicious payload as "<attack> - Attempted".
    if "Attempted" in label:
        return "Benign"
    if label.startswith("Web Attack"):  # raw labels contain a broken dash char
        return "WebAttack"
    return FAMILY_MAP.get(label, "Other")


@dataclass
class Dataset:
    X: pd.DataFrame
    y_binary: np.ndarray  # 0 = benign, 1 = attack
    y_family: np.ndarray  # string family names
    day: np.ndarray | None = None  # capture-day index (0 = first day), for temporal splits

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def clean(df: pd.DataFrame, label_col: str = "Label",
          meta_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Fix the well-known CIC-IDS2017 data-quality problems."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()
    feats = df.drop(columns=[label_col, *meta_cols])
    constant = feats.columns[feats.nunique() <= 1]
    df = df.drop(columns=list(constant))
    # 'Fwd Header Length.1' is an exact duplicate column in the raw CSVs
    df = df.drop(columns=[c for c in df.columns if c.endswith(".1")], errors="ignore")
    return df


DAY_COL = "__day"
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]


def _day_from_filename(name: str) -> int:
    """CIC-IDS2017 ships one or more CSVs per capture day (Monday…Friday)."""
    low = name.lower()
    for i, d in enumerate(_DAYS):
        if d in low:
            return i
    return -1


def load_cicids2017(raw_dir: str | Path, sample_frac: float | None = None,
                    seed: int = 42) -> Dataset:
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSVs in {raw_dir}. See README > Dataset for download steps.")
    frames = []
    for f in files:
        part = pd.read_csv(f, low_memory=False, encoding="latin-1")
        part[DAY_COL] = _day_from_filename(f.name)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=seed)
    df = clean(df, meta_cols=(DAY_COL,))
    families = df["Label"].map(to_family).to_numpy()
    day = df[DAY_COL].to_numpy()
    X = df.drop(columns=["Label", DAY_COL]).select_dtypes("number").astype(np.float32)
    return Dataset(X=X, y_binary=(families != "Benign").astype(int), y_family=families,
                   day=day)


def make_synthetic(n: int = 20_000, n_features: int = 30, seed: int = 42) -> Dataset:
    """Flow-like synthetic data (heavy-tailed, imbalanced). For tests/demo ONLY."""
    rng = np.random.default_rng(seed)
    families = ["Benign", "DoS", "PortScan", "BruteForce", "WebAttack"]
    probs = [0.75, 0.10, 0.08, 0.05, 0.02]
    fam = rng.choice(families, size=n, p=probs)
    X = rng.lognormal(mean=1.0, sigma=0.6, size=(n, n_features))
    for i, f in enumerate(families[1:], start=1):
        mask = fam == f
        cols = rng.choice(n_features, size=5, replace=False)
        X[np.ix_(mask, cols)] *= 2.0 + i
    # Capture days: traffic drifts slowly, and (like the real dataset) some attack
    # families only appear late in the week.
    day = rng.integers(0, 5, size=n)
    late = np.isin(fam, ["PortScan", "WebAttack"])
    day[late] = rng.integers(3, 5, size=late.sum())
    drift_cols = rng.choice(n_features, size=6, replace=False)
    X[:, drift_cols] *= (1 + 0.12 * day)[:, None]
    cols = [f"feat_{i:02d}" for i in range(n_features)]
    return Dataset(X=pd.DataFrame(X.astype(np.float32), columns=cols),
                   y_binary=(fam != "Benign").astype(int), y_family=fam, day=day)


@dataclass
class Split:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    fam_train: np.ndarray
    fam_val: np.ndarray
    fam_test: np.ndarray


def split(ds: Dataset, test_size: float = 0.2, val_size: float = 0.1,
          seed: int = 42) -> Split:
    """Stratified train/val/test split on attack family."""
    idx = np.arange(len(ds.X))
    idx_trv, idx_te = train_test_split(idx, test_size=test_size,
                                       stratify=ds.y_family, random_state=seed)
    rel_val = val_size / (1 - test_size)
    idx_tr, idx_va = train_test_split(idx_trv, test_size=rel_val,
                                      stratify=ds.y_family[idx_trv], random_state=seed)
    X = ds.X.reset_index(drop=True)
    return Split(X.iloc[idx_tr], X.iloc[idx_va], X.iloc[idx_te],
                 ds.y_binary[idx_tr], ds.y_binary[idx_va], ds.y_binary[idx_te],
                 ds.y_family[idx_tr], ds.y_family[idx_va], ds.y_family[idx_te])


def temporal_split(ds: Dataset, train_days: list[int], val_size: float = 0.15,
                   seed: int = 42) -> Split:
    """Train on the past, test on the future (no shuffling across days).

    Random splits leak: flows from the same attack session end up on both sides.
    Here every flow captured on ``train_days`` is train/val and everything
    captured later is test.
    """
    if ds.day is None:
        raise ValueError("Dataset has no capture-day information")
    X = ds.X.reset_index(drop=True)
    idx_past = np.flatnonzero(np.isin(ds.day, train_days))
    idx_te = np.flatnonzero(ds.day > max(train_days))
    strat = ds.y_binary[idx_past]
    idx_tr, idx_va = train_test_split(idx_past, test_size=val_size, stratify=strat,
                                      random_state=seed)
    return Split(X.iloc[idx_tr], X.iloc[idx_va], X.iloc[idx_te],
                 ds.y_binary[idx_tr], ds.y_binary[idx_va], ds.y_binary[idx_te],
                 ds.y_family[idx_tr], ds.y_family[idx_va], ds.y_family[idx_te])
