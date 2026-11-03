"""Task 1: recorded-environment (room) router on the static multipath profile.

Feature = s, the gain-normalised per-subcarrier median log amplitude of the window
(STRIDE transform), i.e. the frequency-selective fading fingerprint of the
environment. Classifier = standardised L2 logistic regression. It operates only
on the numerical window (no metadata), for any window length L >= 1.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cleaning import DATA_BINS
from .transform import decompose, log_gain_normalise


def static_features(W: np.ndarray, bins=None) -> np.ndarray:
    s, _ = decompose(W)
    return s if bins is None else s[..., bins]


def dynamic_features(W: np.ndarray) -> np.ndarray:
    """Motion-energy profile: per-subcarrier temporal std of the dynamic part (data bins)."""
    _, D = decompose(W)
    return D[..., DATA_BINS].std(axis=-2)


def texture_features(W: np.ndarray) -> np.ndarray:
    """Pipeline-texture probe: lag-1/lag-2 temporal autocorrelation of the dynamic part."""
    _, D = decompose(W)
    D = D[..., DATA_BINS]
    D = D - D.mean(axis=-2, keepdims=True)
    den = (D ** 2).sum(axis=-2) + 1e-9
    a1 = (D[..., 1:, :] * D[..., :-1, :]).sum(axis=-2) / den
    a2 = (D[..., 2:, :] * D[..., :-2, :]).sum(axis=-2) / den
    return np.stack([np.median(a1, -1), np.median(a2, -1), np.mean(a1, -1), np.mean(a2, -1)], -1)


def quantisation_features(W: np.ndarray) -> np.ndarray:
    """Format probe: fraction of integer values, of zeros, and amplitude level."""
    W = np.asarray(W)
    frac_int = (W == np.rint(W)).mean(axis=(-1, -2))
    frac_zero = (W == 0).mean(axis=(-1, -2))
    lvl = np.median(W[..., DATA_BINS], axis=(-1, -2))
    return np.stack([frac_int, frac_zero, np.log1p(lvl)], -1)


def make_router(C: float = 1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def centre_crop(W: np.ndarray, L: int) -> np.ndarray:
    T = W.shape[-2]
    s = (T - L) // 2
    return W[..., s:s + L, :]


class RoomRouter:
    """Fitted router used by Task 3 / run_episode (routes on the static profile)."""

    def __init__(self, C: float = 1.0):
        self.model = make_router(C)

    def fit(self, W, rooms):
        self.model.fit(static_features(W), rooms)
        return self

    def predict_proba(self, W):
        return self.model.predict_proba(static_features(W))

    def predict(self, W):
        return self.model.predict(static_features(W))

    @property
    def classes_(self):
        return self.model.classes_
