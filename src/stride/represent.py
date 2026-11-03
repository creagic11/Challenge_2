"""Representation factory: maps windows (N, 150, 256) to frozen feature vectors."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .features import HandcraftedFeatures
from .models.encoder import Encoder, embed


class Representation:
    def __init__(self, name: str, artifacts: str | Path, seed: int = 0, dev_windows: np.ndarray | None = None):
        self.name, self.seed = name, seed
        art = Path(artifacts)
        if name in ("H", "Hd"):
            self.hf = HandcraftedFeatures(denoise=(name == "Hd"))
            if dev_windows is None:
                raise ValueError("handcrafted features need development windows to fit the static PCA")
            self.hf.fit(dev_windows)
        else:
            kind = {"S": "dyn_dn", "S-raw": "raw", "R": "dyn_dn"}[name]
            ck = torch.load(art / f"encoder_{kind}_s{seed}.pt")
            self.kind, self.scale = kind, ck["scale"]
            self.enc = Encoder()
            if name == "R":
                torch.manual_seed(seed)
                self.enc = Encoder()
            else:
                self.enc.load_state_dict(ck["state_dict"])

    def __call__(self, W: np.ndarray) -> np.ndarray:
        if self.name in ("H", "Hd"):
            return self.hf.transform(W)
        return embed(self.enc, W, self.kind, self.scale)
