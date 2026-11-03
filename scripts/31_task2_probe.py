"""Task 2 evaluation: frozen representation + linear probe, scored on validation.

    python scripts/31_task2_probe.py

Representations: handcrafted (H; Hd = on denoised dynamics), random-weight encoder (R),
SSL on raw log amplitude (S-raw), STRIDE-SSL on the denoised dynamic part (S). Probes are logistic
regressions fitted on development windows (C chosen on the internal dev
hold-out, then refit on all dev). Also reported, as diagnostics:
  * cross-room shared-class transfer (probe on SIT/STAND/WALK of one room,
    tested on the other room's validation windows)
  * room-leakage probe (linear room accuracy from the embedding)
Embeddings are cached to derived/cache/emb_*.npz for Task 3.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from stride.data import Dataset  # noqa: E402
from stride.features import HandcraftedFeatures  # noqa: E402
from stride.metrics import ResultLog, activity_metrics  # noqa: E402
from stride.models.encoder import Encoder, embed  # noqa: E402
from stride.utils import ensure_dir  # noqa: E402

SHARED = ["sit", "stand", "walk"]
CS = [0.01, 0.1, 1.0, 10.0]


def probe(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fit_probe(F, y, tr, ho, dev):
    best = max(CS, key=lambda C: activity_metrics(y[ho], probe(C).fit(F[tr], y[tr]).predict(F[ho]),
                                                   sorted(set(y[tr])))["macro_f1"])
    return probe(best).fit(F[dev], y[dev]), best


def representations(ds):
    """Yield (name, seed, features for all windows)."""
    W = ds.W
    dev = ds.idx(split="development")
    for name, dn in [("H", False), ("Hd", True)]:
        hf = HandcraftedFeatures(denoise=dn).fit(W[dev])
        yield name, 0, hf.transform(W)
    art = Path(ds.cfg["paths"]["artifacts"])
    for seed in ds.cfg["seeds"]["ssl"]:
        ck = art / f"encoder_dyn_dn_s{seed}.pt"
        if not ck.exists():
            continue
        meta = torch.load(ck)
        torch.manual_seed(seed)
        yield "R", seed, embed(Encoder(), W, "dyn_dn", meta["scale"])
        for kind, name in [("raw", "S-raw"), ("dyn_dn", "S")]:
            ck = art / f"encoder_{kind}_s{seed}.pt"
            if ck.exists():
                m = torch.load(ck)
                enc = Encoder()
                enc.load_state_dict(m["state_dict"])
                yield name, seed, embed(enc, W, kind, m["scale"])


def main():
    torch.set_num_threads(2)
    ds = Dataset()
    P = ds.cfg["paths"]
    log = ResultLog(Path(P["results"]) / "experiments.csv", ds.cfg["_config_hash"])
    man = ds.man
    y, room = man.label.values, man.room.values
    rows = []
    cache = ensure_dir(Path(P["derived"]) / "cache")
    for name, seed, F in representations(ds):
        np.savez(cache / f"emb_{name}_s{seed}.npz", F=F)
        res = {"rep": name, "seed": seed}
        # per-room probes
        for r in "AB":
            tr = ds.idx(room=r, split="development", holdout=False)
            ho = ds.idx(room=r, split="development", holdout=True)
            dev, val = ds.idx(room=r, split="development"), ds.idx(room=r, split="validation")
            m, C = fit_probe(F, y, tr, ho, dev)
            met = activity_metrics(y[val], m.predict(F[val]), sorted(set(y[dev])))
            for k in ["macro_f1", "balanced_accuracy"]:
                res[f"{r}_{k}"] = met[k]
                log.add("task2_linear_probe", r, "NA", k, met[k], representation=name, seed=seed, C=C)
        # joint 8-class probe over both rooms (shared labels are shared)
        tr, ho = ds.idx(split="development", holdout=False), ds.idx(split="development", holdout=True)
        dev, val = ds.idx(split="development"), ds.idx(split="validation")
        m, C = fit_probe(F, y, tr, ho, dev)
        met = activity_metrics(y[val], m.predict(F[val]), sorted(set(y[dev])))
        for k in ["macro_f1", "balanced_accuracy"]:
            res[f"joint_{k}"] = met[k]
            log.add("task2_linear_probe", "both", "NA", k, met[k], representation=name, seed=seed, C=C)
        # diagnostic: cross-room shared-class transfer (zero target labels)
        for src, tgt in [("A", "B"), ("B", "A")]:
            sd = ds.idx(room=src, split="development", labels=SHARED)
            tv = ds.idx(room=tgt, split="validation", labels=SHARED)
            mm = probe(1.0).fit(F[sd], y[sd])
            f1 = activity_metrics(y[tv], mm.predict(F[tv]), SHARED)["macro_f1"]
            res[f"xfer_{src}to{tgt}_shared_f1"] = f1
            log.add("task2_xroom_shared_transfer", tgt, "NA", "macro_f1", f1, representation=name,
                    seed=seed, source=src)
        # diagnostic: room leakage
        mr = probe(1.0).fit(F[dev], room[dev])
        res["room_leak_acc"] = accuracy_score(room[val], mr.predict(F[val]))
        log.add("task2_room_leakage", "both", "NA", "accuracy", res["room_leak_acc"], representation=name,
                seed=seed)
        rows.append(res)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(P["results"]) / "task2_probe_by_seed.csv", index=False)
    summ = df.drop(columns="seed").groupby("rep").agg(["mean", "std"])
    summ.to_csv(Path(P["results"]) / "task2_probe_summary.csv")
    print(summ.round(3).T)
    log.save()


if __name__ == "__main__":
    main()
