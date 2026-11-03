"""End-to-end episode runner (challenge §8/§10/§14).

    python run_episode.py --episode <dir> [--support support.csv] [--out predictions.csv]

<dir> may contain raw Room-A PCAP files, raw Room-B CSV files and/or pre-made
window_*.npy files (150, 256), in any sub-folder structure. Every raw file goes
through the SAME pipeline as development/validation data:
    read -> QC (valid full-band observations) -> harmonise -> full-capture (150, 256) window
The file format only selects the reader; it is discarded before routing, so the
room router sees numbers only.

Optional support.csv (columns: file,label) supplies labelled target captures for the
target-exclusive classes; the adaptation model of the matching room is then refit
with these supports instead of the stored development supports.

Output CSV columns: window_id, room, label, confidence
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stride.amplitude import harmonise_to_reference  # noqa: E402
from stride.fewshot import EXCLUSIVE  # noqa: E402
from stride.ingest import read_capture  # noqa: E402
from stride.represent import Representation  # noqa: E402
from stride.windowing import official_window  # noqa: E402


def load_windows(episode: Path, ref_median: float):
    ids, W, notes = [], [], []
    files = sorted(p for p in episode.rglob("*") if p.suffix.lower() in (".pcap", ".csv", ".npy")
                   and p.name != "support.csv")
    for f in files:
        try:
            if f.suffix.lower() == ".npy":
                w = np.load(f).astype(np.float32)
                if w.shape != (150, 256):
                    raise ValueError(f"window shape {w.shape}")
            else:
                c = read_capture(f)
                X, t = c.amp[c.valid], c.time[c.valid]
                if len(X) == 0:                       # no valid observation: fall back to all rows
                    X, t = c.amp, c.time
                if f.suffix.lower() == ".pcap":
                    X = harmonise_to_reference(X, ref_median)
                w, _ = official_window(X, t, mode="full")
            ids.append(f.stem)
            W.append(w)
            notes.append(str(f.relative_to(episode)))
        except Exception as e:                        # report, never crash the whole episode
            print(f"skipping {f}: {e}", file=sys.stderr)
    return ids, np.stack(W) if W else np.zeros((0, 150, 256), np.float32), notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--support", default=None)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = ap.parse_args()
    t0 = time.time()
    art = Path(args.artifacts)
    try:
        with open(art / "deployment.pkl", "rb") as fh:
            dep = pickle.load(fh)
    except Exception as e:  # pickles are tied to library versions
        sys.exit(f"could not load artifacts/deployment.pkl ({e!r}).\n"
                 "Rebuild it for this Python environment with:  python scripts/prepare_deployment.py")
    ref = json.loads(dep["harmonisation"])["ref_median_data_amplitude"]
    episode = Path(args.episode)
    ids, W, rel = load_windows(episode, ref)
    if not len(W):
        sys.exit("no readable captures/windows in the episode")
    rep = dep["rep_obj"] or Representation(dep["rep"], art, dep["seed"])
    rooms = dep["router"].predict(W)                  # routing on the numbers only
    Z = rep(W)
    models = dict(dep["models"])
    if args.support:
        from importlib import import_module
        fit_target_model = import_module("45_build_deployment").fit_target_model
        sup = pd.read_csv(args.support)
        stem = {Path(f).stem: i for i, f in enumerate(ids)}
        for tgt in "AB":
            s = sup[sup.label.isin(EXCLUSIVE[tgt])]
            if len(s):
                si = np.array([stem[Path(f).stem] for f in s.file])
                local = np.arange(len(dep["dev_Z"]))
                models[tgt] = fit_target_model(dep["dev_Z"], dep["dev_y"], dep["dev_room"], dep["dev_cid"], local,
                                               tgt, dep["variant_kwargs"], Zsup=Z[si], ysup=s.label.values)
                print(f"refit target-{tgt} model with {len(s)} episode supports")
    labels = np.empty(len(W), dtype=object)
    conf = np.zeros(len(W))
    for r in np.unique(rooms):
        m = rooms == r
        labels[m], conf[m] = models[r].predict(Z[m], np.full(m.sum(), r))
    out = pd.DataFrame(dict(window_id=ids, room=rooms, label=labels, confidence=np.round(conf, 4)))
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} predictions to {args.out} in {time.time() - t0:.1f}s")
    print(out.groupby(["room", "label"]).size().to_string())


if __name__ == "__main__":
    main()
