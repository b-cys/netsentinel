import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netsentinel.data import clean, make_synthetic, split, to_family  # noqa: E402
from netsentinel.evaluate import binary_metrics  # noqa: E402
from netsentinel.features import build_preprocessor  # noqa: E402
from netsentinel.models import AutoencoderDetector, FFNNClassifier  # noqa: E402


def test_family_mapping():
    assert to_family(" BENIGN") == "Benign"
    assert to_family("Web Attack \x96 Brute Force") == "WebAttack"
    assert to_family("DoS Hulk") == "DoS"


def test_clean_removes_inf_nan_dupes_constants():
    df = pd.DataFrame({" A": [1, 1, np.inf, 3, np.nan], " B": [5, 5, 5, 5, 5],
                       " Label": ["BENIGN"] * 5})
    out = clean(df)
    assert "B" not in out.columns and "A" in out.columns
    assert len(out) == 2  # inf + nan rows dropped, duplicate removed


def test_split_has_no_overlap_and_is_stratified():
    ds = make_synthetic(n=3000)
    s = split(ds)
    idx = [set(s.X_train.index), set(s.X_val.index), set(s.X_test.index)]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    assert abs(s.y_train.mean() - s.y_test.mean()) < 0.02


def test_ffnn_and_autoencoder_learn():
    ds = make_synthetic(n=4000)
    s = split(ds)
    pre = build_preprocessor("pca", 0.95)
    Xtr, Xva, Xte = pre.fit_transform(s.X_train), pre.transform(s.X_val), pre.transform(s.X_test)
    clf = FFNNClassifier(epochs=5, verbose=False).fit(Xtr, s.y_train, Xva, s.y_val)
    assert binary_metrics(s.y_test, clf.predict(Xte))["f1"] > 0.7

    b = s.y_train == 0
    ae = AutoencoderDetector(epochs=5, verbose=False).fit(Xtr[b], Xva[s.y_val == 0])
    scores = ae.score(Xte)
    assert scores.shape == (len(Xte),)
    # attacks should reconstruct worse than benign traffic
    assert scores[s.y_test == 1].mean() > scores[s.y_test == 0].mean()


def test_temporal_split_never_trains_on_the_future():
    from netsentinel.data import temporal_split
    ds = make_synthetic(n=3000)
    s = temporal_split(ds, [0, 1, 2])
    assert ds.day[s.X_train.index].max() <= 2 and ds.day[s.X_test.index].min() >= 3
    assert "PortScan" not in set(s.fam_train) and "PortScan" in set(s.fam_test)


def test_pgd_respects_attacker_constraints():
    from netsentinel.adversarial import evade, mutable_mask
    ds = make_synthetic(n=3000)
    s = split(ds)
    pre = build_preprocessor().fit(s.X_train)
    Xtr, Xva, Xte = pre.transform(s.X_train), pre.transform(s.X_val), pre.transform(s.X_test)
    clf = FFNNClassifier(epochs=3, verbose=False).fit(Xtr, s.y_train, Xva, s.y_val)
    mask = mutable_mask(list(s.X_train.columns), ["feat_0"])
    assert mask.sum() == 10
    atk = Xte[s.y_test == 1]
    delta = evade(clf, atk, mask, eps=0.5, steps=5) - atk
    assert np.abs(delta[:, ~mask]).max() == 0          # immutable features untouched
    assert delta.min() >= 0 and delta.max() <= 0.5 + 1e-5  # increase-only, bounded
    # the attack lowers the model's attack score on average
    free = evade(clf, atk, np.ones_like(mask), eps=1.5, steps=10, increase_only=False)
    assert clf.predict(free).mean() < clf.predict(atk).mean()


def test_hybrid_detector_alerts_and_api(tmp_path, monkeypatch):
    from netsentinel.pipeline import HybridDetector
    ds = make_synthetic(n=4000)
    s = split(ds)
    det = HybridDetector(0.02, {"epochs": 3}).fit(s.X_train, s.y_train, s.fam_train,
                                                  s.X_val, s.y_val)
    ben = det.predict(s.X_test[s.y_test == 0]).mean()
    assert ben < 0.08 and det.predict(s.X_test[s.y_test == 1]).mean() > 0.7
    alerts = det.alerts(s.X_test.head(200))
    assert alerts and {"severity", "mitre_attack", "top_features"} <= set(alerts[0])

    path = tmp_path / "m.joblib"
    det.save(path)
    monkeypatch.setenv("NETSENTINEL_MODEL", str(path))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib
    api = importlib.import_module("app.api")
    api.MODEL_PATH, api._det = str(path), None
    from fastapi.testclient import TestClient
    client = TestClient(api.app)
    flows = s.X_test.head(50).astype(float).to_dict(orient="records")
    r = client.post("/predict", json={"flows": flows})
    assert r.status_code == 200 and r.json()["n_flows"] == 50
    assert client.post("/predict", json={"flows": [{"x": 1.0}]}).status_code == 422
