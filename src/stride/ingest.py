"""Read one capture (PCAP or CSV) into a common in-memory form.

The returned record contains only numbers plus provenance. The source *format*
is used solely to choose the reader; it is never passed to any model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .amplitude import amplitude
from .cleaning import observation_validity
from .csi_decode import decode_payloads
from .pcap_reader import read_pcap


@dataclass
class Capture:
    amp: np.ndarray                 # (T_all, 256) float64 amplitude, every decodable observation, file order
    time: np.ndarray                # (T_all,) seconds if known, else row index (float)
    has_timestamps: bool
    reasons: np.ndarray             # (T_all,) reject reason per observation ('' = valid)
    n_packets_in_file: int
    file_level_rejects: dict = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return self.reasons == ""


def read_pcap_capture(path: str | Path) -> Capture:
    cap = read_pcap(str(path))
    pk, dec_reasons = decode_payloads(cap.payloads)
    file_rej = {}
    for r in list(cap.reject_reason) + list(dec_reasons):
        if r:
            file_rej[r] = file_rej.get(r, 0) + 1
    amp = amplitude(pk.csi)
    t = cap.timestamps[pk.packet_index]
    reasons = observation_validity(amp, fixed_pattern_mask=(pk.seq == 0xFFFF))
    return Capture(amp, t, True, reasons, len(cap.payloads), file_rej)


def read_csv_capture(path: str | Path) -> Capture:
    df = pd.read_csv(path, header=0)
    if df.shape[1] != 256:
        raise ValueError(f"{path}: expected 256 columns, found {df.shape[1]}")
    amp = df.to_numpy(dtype=np.float64)
    if np.nanmin(amp) < 0:
        raise ValueError(f"{path}: negative amplitudes")
    reasons = observation_validity(amp)
    return Capture(amp, np.arange(len(amp), dtype=float), False, reasons, len(amp), {})


def read_capture(path: str | Path) -> Capture:
    p = str(path).lower()
    if p.endswith(".pcap"):
        return read_pcap_capture(path)
    if p.endswith(".csv"):
        return read_csv_capture(path)
    raise ValueError(f"unsupported capture format: {path}")
