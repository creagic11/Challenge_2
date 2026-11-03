"""The STRIDE transform: log -> per-observation gain removal -> static / dynamic split.

For a window A (T x 256) of amplitudes:
  L  = log(A + 1)                       multiplicative effects become additive
  g_t = median_{k in data} L[t, k]      per-observation gain (AGC / tx power / scale)
  L' = L - g_t
  s_k = median_t L'[t, k]               static multipath profile (environment, posture)
  D  = L' - s                           dynamic residual (body motion)
Parameter-free and per-window, so it cannot leak information across captures.
"""
from __future__ import annotations

import numpy as np

from .cleaning import DATA_BINS


def log_gain_normalise(W: np.ndarray) -> np.ndarray:
    """W: (..., T, 256) -> L' (same shape)."""
    L = np.log(np.asarray(W, dtype=np.float32) + 1.0)
    g = np.median(L[..., DATA_BINS], axis=-1, keepdims=True)
    return L - g


def decompose(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (static profile s: (..., 256), dynamic D: (..., T, 256))."""
    Lp = log_gain_normalise(W)
    s = np.median(Lp, axis=-2)
    return s, Lp - s[..., None, :]
