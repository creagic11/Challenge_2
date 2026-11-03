"""Decoder tests: synthetic bit-layout checks + physical plausibility on real dev data.

The physical tests validate the decoder *against physics*, not against any other
implementation: a wrong bit layout produces spectra that are not smooth across
adjacent subcarriers and lack the 80 MHz guard-band nulls.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stride.cleaning import DATA_BINS, GUARD_BINS, scale_free_log  # noqa: E402
from stride.csi_decode import unpack_4366c0  # noqa: E402
from stride.ingest import read_pcap_capture  # noqa: E402
from stride.utils import load_config  # noqa: E402


def pack(re, im, e):
    """Build a 4366c0 word from signed mantissas and a signed exponent (test helper)."""
    w = (int(re < 0) << 31) | ((abs(re) & 0xFFF) << 19) | (int(im < 0) << 18) | ((abs(im) & 0xFFF) << 6) | (e & 0x3F)
    return np.uint32(w)


@pytest.mark.parametrize("re,im,e", [(1, 0, 0), (0, 1, 0), (-5, 7, 3), (4095, -4095, -12), (-1, -1, -32), (100, 0, 31)])
def test_unpack_synthetic(re, im, e):
    v = unpack_4366c0(np.array([pack(re, im, e)]))[0]
    assert v == pytest.approx(complex(re, im) * 2.0 ** e)


def _dev_captures(n=6):
    cfg = load_config()
    df = pd.read_csv(ROOT / "configs" / "splits" / "captures.csv")
    dev = df[(df.room == "A") & (df.split == "development")]
    picks = dev.groupby("label").head(1).head(n)
    return [Path(cfg["paths"]["room_A_raw"]) / p for p in picks.source_relpath], list(picks.label)


@pytest.fixture(scope="module")
def caps():
    paths, labels = _dev_captures()
    if not paths[0].exists():
        pytest.skip("raw data not available")
    return [read_pcap_capture(p) for p in paths], labels


def test_256_finite(caps):
    for c in caps[0]:
        assert c.amp.shape[1] == 256 and np.isfinite(c.amp).all()


def test_smooth_across_subcarriers(caps):
    """Valid observations vary smoothly across adjacent subcarriers (coherence bandwidth >> spacing)."""
    for c in caps[0]:
        L = scale_free_log(c.amp[c.valid])[:, DATA_BINS]
        rough = np.median(np.abs(np.diff(L, axis=1)))
        shuffled = np.median(np.abs(np.diff(L[:, np.random.default_rng(0).permutation(L.shape[1])], axis=1)))
        assert rough < 0.5 * shuffled


def test_guard_band_nulls(caps):
    """80 MHz VHT guard subcarriers (FFT bins 123..133) are far below the data subcarriers."""
    for c in caps[0]:
        L = scale_free_log(c.amp[c.valid])
        depth = np.median(L[:, GUARD_BINS]) - np.median(L[:, DATA_BINS])
        assert depth < -2.0


def test_motion_increases_temporal_variance(caps):
    """Walking shows larger temporal variation of the gain-normalised amplitude than standing."""
    caps_, labels = caps
    def tv(c):
        L = scale_free_log(c.amp[c.valid])[:, DATA_BINS]
        L = L - np.median(L, axis=1, keepdims=True)
        return np.median(np.std(L, axis=0))
    v = {lab: tv(c) for c, lab in zip(caps_, labels)}
    assert v["walk"] > v["stand"]
