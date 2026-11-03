"""Stage 1: raw captures -> QC -> common 256-column amplitude form.

    python scripts/10_build_intermediate.py [--no-csv]

Outputs
  derived/intermediate_csv/room_{A,B}/<capture>.csv   valid observations, subcarrier_001..256
  derived/cache/captures.npz                         fast cache (valid obs + time per capture)
  results/qc_captures.csv                            per-capture QC summary (committed)
  results/qc_summary.csv                             reject counts per room/reason (committed)
  artifacts/harmonisation.json                       reference level fitted on Room-B dev rows
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stride.amplitude import harmonise_to_reference  # noqa: E402
from stride.cleaning import DATA_BINS  # noqa: E402
from stride.ingest import read_capture  # noqa: E402
from stride.utils import ensure_dir, load_config, sha1_file  # noqa: E402

COLS = [f"subcarrier_{i:03d}" for i in range(1, 257)]


def canonical_rank(relpath: str) -> tuple:
    """Prefer names without copy suffixes like '(1)' when de-duplicating."""
    return (bool(re.search(r"\(\d+\)", relpath)), len(relpath), relpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-csv", action="store_true", help="skip writing intermediate CSVs")
    args = ap.parse_args()
    cfg = load_config()
    P = cfg["paths"]
    caps = pd.read_csv(Path(P["splits"]) / "captures.csv")
    derived = Path(P["derived"])
    t0 = time.time()

    # 1) read + QC every capture
    recs, data = [], {}
    for r in caps.itertuples():
        raw = Path(P["room_A_raw"] if r.room == "A" else P["room_B_raw"]) / r.source_relpath
        c = read_capture(raw)
        counts = pd.Series(c.reasons).value_counts().to_dict()
        rec = dict(capture_id=r.capture_id, room=r.room, label=r.label, split=r.split,
                   dev_holdout=r.dev_holdout, source_relpath=r.source_relpath,
                   source_sha1=sha1_file(raw), n_packets_in_file=c.n_packets_in_file,
                   n_decoded=len(c.reasons), n_valid=int(c.valid.sum()))
        for k in ["fixed_pattern", "partial_band", "duplicate_obs", "non_finite", "all_zero"]:
            rec[f"rej_{k}"] = int(counts.get(k, 0))
        for k, v in c.file_level_rejects.items():
            rec[f"rej_file_{k}"] = v
        tv = c.time[c.valid]
        rec["valid_span"] = float(tv[-1] - tv[0]) if len(tv) > 1 else 0.0
        recs.append(rec)
        data[r.capture_id] = (c.amp[c.valid], tv)
    qc = pd.DataFrame(recs).fillna(0)
    print(f"read {len(qc)} captures in {time.time() - t0:.0f}s")

    # 2) exact duplicate captures (same raw bytes) -> keep the canonical name only
    qc["duplicate_of"] = ""
    for h, g in qc.groupby("source_sha1"):
        if len(g) > 1:
            keep = min(g.source_relpath, key=canonical_rank)
            kid = g[g.source_relpath == keep].capture_id.iloc[0]
            if g.split.nunique() > 1:
                print("WARNING: duplicate capture across splits:", list(g.capture_id))
            for i in g.index:
                if qc.at[i, "capture_id"] != kid:
                    qc.at[i, "duplicate_of"] = kid
    qc["excluded"] = (qc.duplicate_of != "") | (qc.n_valid == 0)
    print("excluded captures:", qc[qc.excluded][["capture_id", "duplicate_of", "n_valid"]].to_dict("records"))

    # 3) harmonisation reference from Room-B *development* valid rows only
    bdev = qc[(qc.room == "B") & (qc.split == "development") & ~qc.excluded].capture_id
    ref = float(np.median(np.concatenate([np.median(data[c][0][:, DATA_BINS], axis=1) for c in bdev])))
    ensure_dir(P["artifacts"])
    with open(Path(P["artifacts"]) / "harmonisation.json", "w") as fh:
        json.dump({"ref_median_data_amplitude": ref, "fitted_on": "room B development valid rows"}, fh, indent=1)
    print(f"harmonisation reference median amplitude = {ref:.1f}")

    # 4) harmonise Room A, write intermediate CSVs + cache
    cache = {}
    for r in qc.itertuples():
        X, t = data[r.capture_id]
        if r.room == "A" and len(X):
            X = harmonise_to_reference(X, ref)
        X = X.astype(np.float32)
        cache[r.capture_id] = (X, t.astype(np.float64))
        if not args.no_csv and len(X):
            out = ensure_dir(derived / "intermediate_csv" / f"room_{r.room}")
            name = Path(r.source_relpath).stem
            sub = Path(r.source_relpath).parent.as_posix().replace("/", "__")
            fname = f"{sub}__{name}.csv" if r.room == "A" else f"{name}.csv"
            pd.DataFrame(X.astype(np.int32), columns=COLS).to_csv(out / fname, index=False)
    ensure_dir(derived / "cache")
    ids = list(cache)
    np.savez_compressed(derived / "cache" / "captures.npz",
                        ids=np.array(ids),
                        lengths=np.array([len(cache[i][0]) for i in ids]),
                        X=np.concatenate([cache[i][0] for i in ids]),
                        t=np.concatenate([cache[i][1] for i in ids]))
    ensure_dir(P["results"])
    qc.to_csv(Path(P["results"]) / "qc_captures.csv", index=False)
    rej = [c for c in qc.columns if c.startswith("rej_")]
    summ = qc.groupby("room")[["n_decoded", "n_valid"] + rej].sum().T
    summ.to_csv(Path(P["results"]) / "qc_summary.csv")
    print(summ)
    print(qc.groupby(["room", "label"]).n_valid.describe()[["min", "50%", "max"]])
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
