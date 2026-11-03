"""Stage 2: official (150, 256) float32 windows, one per capture, plus provenance.

    python scripts/11_build_windows.py [--no-npy]

Outputs
  derived/windows/{development,local_validation}/window_XXXXX.npy  (challenge contract)
  derived/cache/official_windows.npz                               (all windows, fast loading)
  results/windows_manifest.csv                                     (provenance; committed)
Window ids are a seeded random permutation, so an id never reveals room or label.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stride.data import load_capture_cache  # noqa: E402
from stride.utils import ensure_dir, git_hash, load_config  # noqa: E402
from stride.windowing import official_window  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-npy", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    P = cfg["paths"]
    qc = pd.read_csv(Path(P["results"]) / "qc_captures.csv", keep_default_na=False)
    qc = qc[qc.excluded.astype(str) != "True"].reset_index(drop=True)
    cache = load_capture_cache(Path(P["derived"]) / "cache" / "captures.npz")
    perm = np.random.default_rng(cfg["seeds"]["global"]).permutation(len(qc))
    mode = cfg["window"].get("mode", "consecutive")
    rows, W = [], np.zeros((len(qc), 150, 256), np.float32)
    for i, r in enumerate(qc.itertuples()):
        X, t = cache[r.capture_id]
        w, prov = official_window(X, t, mode=mode)
        W[i] = w
        wid = f"window_{perm[i]:05d}"
        rows.append(dict(window_id=wid, capture_id=r.capture_id, room=r.room, label=r.label,
                         split=r.split, dev_holdout=r.dev_holdout, source_relpath=r.source_relpath,
                         source_sha1=r.source_sha1, **prov, window_mode=mode,
                         config_hash=cfg["_config_hash"], git_hash=git_hash()))
        if not args.no_npy:
            sub = "development" if r.split == "development" else "local_validation"
            np.save(ensure_dir(Path(P["derived"]) / "windows" / sub) / f"{wid}.npy", w)
    man = pd.DataFrame(rows)
    man.to_csv(Path(P["results"]) / "windows_manifest.csv", index=False)
    np.savez(Path(P["derived"]) / "cache" / "official_windows.npz", W=W,
             window_id=np.array(man.window_id.tolist(), dtype="U16"))
    print(man.groupby(["room", "split", "method"]).size())


if __name__ == "__main__":
    main()
