"""Build the deployment artefact used by run_episode.py.

    python scripts/45_build_deployment.py --rep S --seed 0 --variant SAPT-OT+LR --K 20

Stores: router, representation spec, development features/labels/rooms (needed to
refit with episode-provided supports), and one fitted adaptation model per target
room (default supports: K development captures per exclusive class, episode 0).
Validation data is not used.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from stride.data import Dataset  # noqa: E402
from stride.fewshot import EXCLUSIVE, SAPT, SAPTConfig, sample_supports  # noqa: E402
from stride.portable import PortableRouter, make_portable_features, make_portable_sapt  # noqa: E402
from stride.represent import Representation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

VARIANTS = import_module("40_task3_fewshot").VARIANTS


def fit_target_model(Z, y, room, cid, dev_idx, tgt, cfg_kwargs, sup_ids=None, K=20, Zsup=None, ysup=None):
    src = "B" if tgt == "A" else "A"
    s_idx = dev_idx[room[dev_idx] == src]
    t_idx = dev_idx[room[dev_idx] == tgt]
    if Zsup is None:
        lab = {cid[i]: y[i] for i in t_idx}
        sup = sample_supports(lab, EXCLUSIVE[tgt], [K], seed=0 if tgt == "A" else 1)[K]
        pos = {cid[i]: i for i in t_idx}
        si = np.array([pos[c] for c in sup])
        Zsup, ysup = Z[si], y[si]
    c = SAPTConfig(**cfg_kwargs)
    return SAPT(c).fit(Z[dev_idx], Z[s_idx], y[s_idx], src, Zsup, ysup, tgt, Zpool=Z[t_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="S")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="SAPT-OT+LR")
    ap.add_argument("--K", type=int, default=20)
    args = ap.parse_args()
    ds = Dataset()
    P = ds.cfg["paths"]
    dev = ds.idx(split="development")
    rep = Representation(args.rep, P["artifacts"], args.seed, dev_windows=ds.W[dev])
    Z = np.zeros((len(ds.man), 1))
    Zd = rep(ds.W[dev])
    # plain NumPy string arrays (version-robust pickling; no pandas extension arrays)
    y = np.array(ds.man.label.tolist(), dtype=object)[dev]
    room = np.array(ds.man.room.tolist(), dtype=object)[dev]
    cid = np.array(ds.man.capture_id.tolist(), dtype=object)[dev]
    local = np.arange(len(dev))
    models = {t: fit_target_model(Zd, y, room, cid, local, t, VARIANTS[args.variant], K=args.K) for t in "AB"}
    with open(Path(P["artifacts"]) / "router.pkl", "rb") as fh:
        router = pickle.load(fh)
    # export to NumPy-only predictors so the artefact loads under any scikit-learn version
    models = {t: make_portable_sapt(m) for t, m in models.items()}
    if args.rep in ("H", "Hd"):
        make_portable_features(rep.hf)
    dep = dict(rep=args.rep, seed=args.seed, variant=args.variant, variant_kwargs=VARIANTS[args.variant],
               K=args.K, router=PortableRouter(router), models=models, dev_Z=Zd, dev_y=y, dev_room=room, dev_cid=cid,
               rep_obj=rep if args.rep in ("H", "Hd") else None,
               harmonisation=(Path(P["artifacts"]) / "harmonisation.json").read_text())
    with open(Path(P["artifacts"]) / "deployment.pkl", "wb") as fh:
        pickle.dump(dep, fh)
    print("saved artifacts/deployment.pkl", {k: dep[k] for k in ["rep", "seed", "variant", "K"]})
    del Z


if __name__ == "__main__":
    main()
