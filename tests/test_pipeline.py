"""Pipeline contract and leakage tests."""
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stride.windowing import official_window, resample_uniform  # noqa: E402


def test_splits_disjoint_and_counts():
    df = pd.read_csv(ROOT / "configs" / "splits" / "captures.csv")
    assert (df.groupby("capture_id").split.nunique() == 1).all()
    val = df[df.split == "validation"]
    assert (val.groupby(["room", "label"]).size() == 25).all()           # challenge §9
    assert not df[(df.split == "validation") & (df.dev_holdout.astype(str) == "True")].shape[0]


def test_roomB_validation_only_documented_patterns():
    v = pd.read_csv(ROOT / "configs" / "splits" / "roomB_validation.csv")
    assert v.source_relpath.str.match(r"^((Empty|Lying|Sitting|Standing)_Amp_\d+|Walking_\d+)\.csv$").all()


def test_window_contract_and_no_padding():
    X = np.arange(60 * 256, dtype=float).reshape(60, 256)
    t = np.sort(np.random.default_rng(0).uniform(0, 4, 60))
    w, prov = official_window(X, t, mode="full")
    assert w.shape == (150, 256) and w.dtype == np.float32
    assert len(np.unique(w, axis=0)) > 140           # interpolation, not repeated padding rows
    assert np.all(w == np.rint(w))


def test_resample_endpoints():
    X = np.array([[0.0], [10.0]])
    out = resample_uniform(X, np.array([0.0, 1.0]), 11)
    assert np.allclose(out[:, 0], np.arange(11))


def test_manifest_ids_reveal_nothing():
    m = pd.read_csv(ROOT / "results" / "windows_manifest.csv")
    assert m.window_id.is_unique
    # ids are a random permutation: room must not be predictable from id order
    order = m.sort_values("window_id").room.values
    runs = np.sum(order[1:] != order[:-1])
    assert runs > len(order) * 0.3


def test_deployment_uses_development_only():
    p = ROOT / "artifacts" / "deployment.pkl"
    if not p.exists():
        pytest.skip("deployment artefact not built")
    dep = pickle.load(open(p, "rb"))
    caps = pd.read_csv(ROOT / "configs" / "splits" / "captures.csv").set_index("capture_id")
    assert (caps.loc[list(dep["dev_cid"]), "split"] == "development").all()


def test_run_episode_smoke(tmp_path):
    if not (ROOT / "artifacts" / "deployment.pkl").exists():
        pytest.skip("deployment artefact not built")
    caps = pd.read_csv(ROOT / "configs" / "splits" / "captures.csv")
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs" / "experiment.yaml"))
    rawA = (ROOT / cfg["paths"]["room_A_raw"]).resolve()
    rawB = (ROOT / cfg["paths"]["room_B_raw"]).resolve()
    if not rawA.exists():
        pytest.skip("raw data not available")
    pick = caps[caps.split == "development"].groupby("room").head(2)
    for r in pick.itertuples():
        src = (rawA if r.room == "A" else rawB) / r.source_relpath
        shutil.copy(src, tmp_path / f"cap{r.Index}{src.suffix}")
    out = tmp_path / "pred.csv"
    subprocess.run([sys.executable, str(ROOT / "run_episode.py"), "--episode", str(tmp_path), "--out", str(out)],
                   check=True, capture_output=True)
    pred = pd.read_csv(out)
    assert list(pred.columns) == ["window_id", "room", "label", "confidence"]
    assert len(pred) == 4 and set(pred.room) <= {"A", "B"}


def test_portable_predictors_match_sklearn():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from stride.portable import LinearSoftmax, PortablePCA
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    for n_cls in (2, 4):
        y = rng.integers(0, n_cls, 200)
        lr = LogisticRegression(max_iter=2000).fit(X, y)
        assert np.allclose(LinearSoftmax.from_sklearn(lr).predict_proba(X), lr.predict_proba(X), atol=1e-8)
    p = PCA(3).fit(X)
    assert np.allclose(PortablePCA(p.mean_, p.components_).transform(X), p.transform(X))
