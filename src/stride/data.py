"""Loading helpers for the derived caches and the provenance manifest.

The model-facing accessors return arrays only; room/label lookups go through the
manifest and are used only by training/evaluation code (metadata firewall).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import load_config


def load_capture_cache(path) -> dict:
    z = np.load(path)
    X, T = z["X"], z["t"]          # decompress once (each z[...] access re-reads the file)
    out, off = {}, 0
    for cid, n in zip(z["ids"], z["lengths"]):
        out[str(cid)] = (X[off:off + n], T[off:off + n])
        off += n
    return out


class Dataset:
    """Official windows + manifest, with leakage-safe subset helpers."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        P = self.cfg["paths"]
        self.man = pd.read_csv(Path(P["results"]) / "windows_manifest.csv")
        self.man["dev_holdout"] = self.man.dev_holdout.astype(str) == "True"
        z = np.load(Path(P["derived"]) / "cache" / "official_windows.npz")
        assert (z["window_id"] == self.man.window_id.values).all()
        self.W = z["W"]
        self._captures = None

    @property
    def captures(self) -> dict:
        if self._captures is None:
            self._captures = load_capture_cache(Path(self.cfg["paths"]["derived"]) / "cache" / "captures.npz")
        return self._captures

    def idx(self, room=None, split=None, labels=None, holdout=None) -> np.ndarray:
        m = np.ones(len(self.man), bool)
        if room is not None:
            m &= self.man.room.values == room
        if split is not None:
            m &= self.man.split.values == split
        if labels is not None:
            m &= self.man.label.isin(labels).values
        if holdout is not None:
            m &= self.man.dev_holdout.values == holdout
        return np.where(m)[0]
