"""Complex CSI -> amplitude, and cross-room quantisation harmonisation.

Room B arrives as integer amplitudes (the dataset's CSV export). Room A is decoded
here to exact floating-point values with an arbitrary per-packet power-of-two
scale. If left as is, "integer vs non-integer" and the absolute scale would be
trivial, non-physical room cues (symmetry principle). We therefore map every
Room-A observation to Room B's convention: scale each observation so that its
median data-subcarrier amplitude equals a reference level measured on Room-B
*development* rows, then round to integers. The per-observation gain is removed
again later (log + gain removal), so this changes only the quantisation
granularity, which now matches across rooms.
"""
from __future__ import annotations

import numpy as np

from .cleaning import DATA_BINS


def amplitude(csi: np.ndarray) -> np.ndarray:
    return np.abs(csi)


def harmonise_to_reference(amp: np.ndarray, ref_median: float) -> np.ndarray:
    med = np.median(amp[:, DATA_BINS], axis=1, keepdims=True)
    med = np.where(med > 0, med, 1.0)
    return np.rint(amp / med * ref_median)
