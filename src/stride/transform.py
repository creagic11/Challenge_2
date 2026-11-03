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


def gd_threshold_rank(sv: np.ndarray, shape: tuple) -> int:
    """Number of singular values above the Gavish-Donoho (2014) optimal hard threshold
    for unknown noise level: tau = omega(beta) * median(sv)."""
    m, n = min(shape), max(shape)
    b = m / n
    w = 0.56 * b ** 3 - 0.95 * b ** 2 + 1.82 * b + 1.43
    return int((sv > w * np.median(sv[:m])).sum())


def denoise_dynamic(D: np.ndarray, bins=DATA_BINS) -> np.ndarray:
    """Low-rank (random-matrix) denoising of the dynamic part of ONE window.

    Body motion changes many subcarriers coherently (low rank), whereas measurement noise
    is spread over all singular values. Keeping only components above the Gavish-Donoho
    threshold removes the session-dependent noise floor without any fitted parameter.
    D: (T, 256) -> (T, 256), non-data bins zeroed.
    """
    X = D[:, bins].astype(np.float64)
    mu = X.mean(0)
    u, sv, vt = np.linalg.svd(X - mu, full_matrices=False)
    k = gd_threshold_rank(sv, X.shape)
    out = np.zeros_like(D, dtype=np.float32)
    out[:, bins] = (u[:, :k] * sv[:k]) @ vt[:k]
    return out


def decompose_denoised(W: np.ndarray):
    """(static profile, denoised dynamic part) for a batch of windows (N, T, 256)."""
    s, D = decompose(W)
    return s, np.stack([denoise_dynamic(d) for d in D])
