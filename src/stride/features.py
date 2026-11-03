"""Handcrafted-feature baseline (Task 2), computed on the same STRIDE parts.

Per window (all parameter-free except the static PCA, which is fitted on dev):
  motion energy      per-subcarrier temporal std of D: 8 sub-band means + 10/50/90 % quantiles
  dynamics shape     explained-variance ratio of the top-3 temporal components of D
  temporal spectrum  of the first component: 6 relative band powers, spectral entropy,
                     dominant normalised frequency
  temporal structure lag-1/5/10 autocorrelation, zero-crossing rate, kurtosis, peak-to-peak
  static profile     10 principal components of s (PCA fitted on development windows)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import PCA

from .cleaning import DATA_BINS
from .transform import decompose, decompose_denoised


def _dyn_feats(D: np.ndarray) -> np.ndarray:
    D = D[:, DATA_BINS]
    sd = D.std(axis=0)
    bands = [b.mean() for b in np.array_split(sd, 8)]
    q = np.quantile(sd, [0.1, 0.5, 0.9])
    Dc = D - D.mean(axis=0)
    u, s, _ = np.linalg.svd(Dc, full_matrices=False)
    ev = s ** 2 / max(np.sum(s ** 2), 1e-12)
    pc = u[:, 0] * s[0]
    P = np.abs(np.fft.rfft(pc - pc.mean())) ** 2
    P = P[1:] / max(P[1:].sum(), 1e-12)
    bp = [b.sum() for b in np.array_split(P, 6)]
    ent = -np.sum(P * np.log(P + 1e-12)) / np.log(len(P))
    dom = (np.argmax(P) + 1) / (2 * len(P))
    def ac(k):
        a, b = pc[:-k] - pc.mean(), pc[k:] - pc.mean()
        return np.sum(a * b) / max(np.sqrt(np.sum(a * a) * np.sum(b * b)), 1e-12)
    zcr = np.mean(np.diff(np.sign(pc - pc.mean())) != 0)
    return np.r_[bands, q, ev[:3], bp, ent, dom, ac(1), ac(5), ac(10), zcr, kurtosis(pc), np.ptp(pc)]


class HandcraftedFeatures:
    def __init__(self, n_static_pcs: int = 10, denoise: bool = True):
        self.pca = PCA(n_static_pcs, random_state=0)
        self.denoise = denoise

    def _parts(self, W):
        s, D = decompose_denoised(W) if self.denoise else decompose(W)
        dyn = np.stack([_dyn_feats(d) for d in D])
        return s[:, DATA_BINS], dyn

    def fit(self, W):
        s, _ = self._parts(W)
        self.pca.fit(s)
        return self

    def transform(self, W, include_static: bool = True):
        s, dyn = self._parts(W)
        out = np.c_[dyn, self.pca.transform(s)] if include_static else dyn
        return np.nan_to_num(out)   # constant windows (captures with a single valid observation)
