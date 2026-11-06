"""CSI quality control: per-observation validity and per-capture checks.

Rejection reasons (one per observation, '' = valid). Every rule is a physical or
format criterion, applied identically to both rooms where the information exists:

  malformed            truncated / wrong-length packet (Room A reader)
  non_finite           NaN or inf amplitude
  all_zero             every subcarrier is exactly zero
  fixed_pattern        Room A reports carrying sequence number 0xFFFF (strictly alternating
                       with normal reports). Measured on development data, their energy is
                       confined to the lowest 20 MHz quarter, like the narrow-band frames below,
                       so they are not valid 80 MHz observations. (The reason code keeps its
                       original name; the shared band mask is why their profile is nearly identical
                       in every capture, cross-capture corr. 0.99.)
  partial_band         energy is confined to one or two 20 MHz quarters of the
                       80 MHz band (narrow-band frames in the ping/pong exchange).
                       Such rows carry noise on >= half of the 256 subcarriers, so they
                       are not valid 256-subcarrier observations.
  duplicate_obs        exact copy of the previous valid observation
"""
from __future__ import annotations

import numpy as np

# FFT-bin index sets (bins 0..127 -> subcarriers 0..127; 128..255 -> -128..-1)
GUARD_BINS = np.r_[123:134]
DC_BINS = np.array([255, 0, 1])
QUARTERS = {                       # four 20 MHz sub-bands, guards/DC excluded
    "q1(-128..-65)": np.r_[134:192],
    "q2(-64..-1)": np.r_[192:255],
    "q3(0..63)": np.r_[2:64],
    "q4(64..127)": np.r_[64:123],
}
DATA_BINS = np.setdiff1d(np.arange(256), np.r_[GUARD_BINS, DC_BINS])
QUARTER_ACTIVE_DB = 2.0            # log-amplitude units below the strongest quarter
MIN_ACTIVE_QUARTERS = 3


def scale_free_log(amp: np.ndarray) -> np.ndarray:
    """log amplitude relative to each observation's median data-subcarrier level.

    Scale-invariant, so QC behaves identically on decoded PCAP values (arbitrary
    power-of-two scale) and on the CSV integers. The small floor (1e-3 of the
    median, i.e. -60 dB) keeps exact zeros finite.
    """
    med = np.median(amp[:, DATA_BINS], axis=1, keepdims=True)
    med = np.where(med > 0, med, 1.0)
    return np.log(amp / med + 1e-3)


def quarter_levels(amp: np.ndarray) -> np.ndarray:
    """Median log-amplitude in each 20 MHz quarter, shape (T, 4)."""
    L = scale_free_log(amp)
    return np.stack([np.median(L[:, ix], axis=1) for ix in QUARTERS.values()], axis=1)


def active_quarters(amp: np.ndarray, tol: float = QUARTER_ACTIVE_DB) -> np.ndarray:
    q = quarter_levels(amp)
    return (q > q.max(axis=1, keepdims=True) - tol).sum(axis=1)


def observation_validity(amp: np.ndarray, fixed_pattern_mask: np.ndarray | None = None) -> np.ndarray:
    """Return an array of reject reasons ('' = valid) for each row of `amp` (T, 256)."""
    T = amp.shape[0]
    reason = np.full(T, "", dtype=object)
    finite = np.isfinite(amp).all(axis=1)
    reason[~finite] = "non_finite"
    safe = np.where(np.isfinite(amp), amp, 0.0)
    reason[(reason == "") & (safe == 0).all(axis=1)] = "all_zero"
    if fixed_pattern_mask is not None:
        reason[(reason == "") & fixed_pattern_mask] = "fixed_pattern"
    nq = active_quarters(safe)
    reason[(reason == "") & (nq < MIN_ACTIVE_QUARTERS)] = "partial_band"
    # exact duplicates of the previous *valid* observation
    last = None
    for t in range(T):
        if reason[t] != "":
            continue
        if last is not None and np.array_equal(safe[t], safe[last]):
            reason[t] = "duplicate_obs"
        else:
            last = t
    return reason


def null_depth(amp: np.ndarray) -> np.ndarray:
    """Guard-band minus data-band median log amplitude (diagnostic only)."""
    L = scale_free_log(amp)
    return np.median(L[:, GUARD_BINS], axis=1) - np.median(L[:, DATA_BINS], axis=1)
