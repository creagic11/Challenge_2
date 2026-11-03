"""Deterministic (150, 256) window construction from one capture's valid observations.

Facts measured on development data (see docs/decision_log):
  * Room A: ~3.9 s per capture, valid full-band observations ~55-75 per capture
    (never >= 150), irregularly spaced (narrow-band and 0xFFFF reports removed).
  * Room B: 500 rows per capture, 80-100 % valid, no timestamps (row index = time proxy).

Rule (identical code for both rooms; `mode` is a configuration choice):
  'consecutive' : if the capture has >= L valid observations, take the L consecutive
                  valid observations centred in the capture (no interpolation);
                  otherwise fall back to 'full'.
  'full'        : resample the capture's whole valid-observation sequence onto L
                  uniformly spaced time points (linear interpolation per subcarrier,
                  using timestamps when present, else row index).
No observations from different captures are ever mixed, and no padding rows are
created, so capture length cannot leak through repeated rows.
"""
from __future__ import annotations

import numpy as np

L_OFFICIAL = 150


def resample_uniform(X: np.ndarray, t: np.ndarray, n: int, t0: float | None = None,
                     t1: float | None = None) -> np.ndarray:
    """Linear interpolation of X (T, C) sampled at times t onto n uniform points in [t0, t1]."""
    if len(t) == 1:
        return np.repeat(X, n, axis=0)
    t0 = t[0] if t0 is None else t0
    t1 = t[-1] if t1 is None else t1
    grid = np.linspace(t0, t1, n)
    idx = np.searchsorted(t, grid, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 2)
    w = ((grid - t[idx]) / np.maximum(t[idx + 1] - t[idx], 1e-12)).clip(0, 1)[:, None]
    return (1 - w) * X[idx] + w * X[idx + 1]


def official_window(X: np.ndarray, t: np.ndarray, mode: str = "consecutive",
                    length: int = L_OFFICIAL) -> tuple[np.ndarray, dict]:
    """X, t: valid observations only (in temporal order). Returns (window, provenance)."""
    n = len(X)
    if n == 0:
        raise ValueError("capture has no valid observations")
    if mode == "consecutive" and n >= length:
        s = (n - length) // 2
        return X[s:s + length].astype(np.float32), dict(method="consecutive", obs_start=s,
                                                        obs_end=s + length, n_valid=n)
    # round: interpolated values are mapped back onto the integer amplitude grid used by
    # all observations, so "integer vs non-integer" cannot act as a room cue
    W = np.rint(resample_uniform(X, t, length))
    return W.astype(np.float32), dict(method="resampled_full", obs_start=0, obs_end=n, n_valid=n)


def random_crop(X: np.ndarray, t: np.ndarray, rng: np.random.Generator, mode: str = "consecutive",
                length: int = L_OFFICIAL, min_frac: float = 0.6) -> np.ndarray:
    """Training-time crop of one capture (never mixes captures)."""
    n = len(X)
    if mode == "consecutive" and n >= length:
        s = rng.integers(0, n - length + 1)
        return X[s:s + length].astype(np.float32)
    span = t[-1] - t[0]
    frac = rng.uniform(min_frac, 1.0)
    a = t[0] + rng.uniform(0, 1 - frac) * span
    return np.rint(resample_uniform(X, t, length, a, a + frac * span)).astype(np.float32)
