"""Reproduce the data-inspection statistics quoted in docs/data_findings.md and the report.

    python scripts/01_data_inspection.py        (development captures only)

Writes results/inspection_stats.csv (statistic, room, label, value).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stride.cleaning import DATA_BINS, scale_free_log  # noqa: E402
from stride.csi_decode import decode_payloads  # noqa: E402
from stride.data import Dataset  # noqa: E402
from stride.pcap_reader import read_pcap  # noqa: E402
from stride.transform import decompose, denoise_dynamic  # noqa: E402
from stride.utils import load_config  # noqa: E402


def main():
    cfg = load_config()
    P = cfg["paths"]
    caps = pd.read_csv(Path(P["splits"]) / "captures.csv")
    dev = caps[caps.split == "development"]
    rows = []
    add = lambda s, room, label, v: rows.append(dict(statistic=s, room=room, label=label, value=float(v)))

    # 1) 0xFFFF reports vs normal reports (Room A, every 10th development capture)
    pf, pn, ratio, timing, band = [], [], [], [], []
    for r in dev[dev.room == "A"].iloc[::10].itertuples():
        cap = read_pcap(str(Path(P["room_A_raw"]) / r.source_relpath))
        pk, _ = decode_payloads(cap.payloads)
        ff = pk.seq == 0xFFFF
        if ff.sum() < 5 or (~ff).sum() < 5:
            continue
        A = np.abs(pk.csi)
        pf.append(np.median(scale_free_log(A[ff]), 0)[DATA_BINS])
        prof = np.fft.fftshift(np.median(scale_free_log(A[ff]), 0))   # subcarrier order -128..127
        band.append(np.median(prof[8:58]) - np.median(prof[138:228]))  # lowest 20 MHz quarter vs rest
        pn.append(np.median(scale_free_log(A[~ff]), 0)[DATA_BINS])
        ratio.append(np.median(A[~ff][:, DATA_BINS]) / max(np.median(A[ff][:, DATA_BINS]), 1e-30))
        t = cap.timestamps[pk.packet_index]
        timing.append((np.median(np.diff(t)), t[-1] - t[0]))
    cc = lambda M: np.corrcoef(np.array(M))[np.triu_indices(len(M), 1)].mean()
    add("cross_capture_profile_corr_0xFFFF", "A", "all", cc(pf))
    add("cross_capture_profile_corr_normal", "A", "all", cc(pn))
    add("median_amplitude_ratio_normal_over_0xFFFF", "A", "all", np.median(ratio))
    add("median_capture_duration_s", "A", "all", np.median([d for _, d in timing]))
    add("0xFFFF_lowest_quarter_minus_rest_log_amp", "A", "all", np.median(band))

    # 2) narrow-band share per room/class (from the QC table, all development captures)
    qc = pd.read_csv(Path(P["results"]) / "qc_captures.csv", keep_default_na=False)
    qc = qc[(qc.split == "development") & (qc.excluded.astype(str) != "True")]
    for (room, lab), g in qc.groupby(["room", "label"]):
        base = (g.n_decoded - g.rej_fixed_pattern).sum()
        add("narrowband_share_of_non_fixed_rows", room, lab, g.rej_partial_band.sum() / base)
        add("median_valid_observations", room, lab, g.n_valid.median())

    # 3) motion energy per class: naive (std of D) vs denoised (rms of D-hat), development windows
    ds = Dataset(cfg)
    idx = ds.idx(split="development")
    _, D = decompose(ds.W[idx])
    me_raw = np.median(D[..., DATA_BINS].std(1), -1)
    me_dn = np.array([np.sqrt((denoise_dynamic(d)[:, DATA_BINS] ** 2).mean()) for d in D])
    m = ds.man.iloc[idx]
    for (room, lab), g in pd.DataFrame(dict(room=m.room.values, label=m.label.values, raw=me_raw,
                                            dn=me_dn)).groupby(["room", "label"]):
        add("motion_energy_naive_median", room, lab, g.raw.median())
        add("motion_energy_denoised_median", room, lab, g.dn.median())

    # 4) sampling-rate evidence: lag-1 autocorrelation of top-3 motion components, walking captures
    def pc_lag1(X, k=3):
        L = np.log1p(X[:, DATA_BINS])
        L = L - np.median(L, 1, keepdims=True)
        Dd = L - np.median(L, 0)
        Dd = Dd - Dd.mean(0)
        u, s, _ = np.linalg.svd(Dd, full_matrices=False)
        Y = u[:, :k] * s[:k]
        return np.sum(Y[1:] * Y[:-1]) / np.sqrt(np.sum(Y[1:] ** 2) * np.sum(Y[:-1] ** 2))
    c = ds.captures
    walkA = [pc_lag1(c[i][0]) for i in dev[(dev.room == "A") & (dev.label == "walk")].capture_id if len(c[i][0]) >= 30]
    add("walk_pc_lag1_native", "A", "walk", np.median(walkA))
    for d in [1, 2, 4, 8, 16]:
        v = [pc_lag1(c[i][0][::d]) for i in dev[(dev.room == "B") & (dev.label == "walk")].capture_id]
        add(f"walk_pc_lag1_decimated_x{d}", "B", "walk", np.median(v))

    out = pd.DataFrame(rows)
    out.to_csv(Path(P["results"]) / "inspection_stats.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
