"""Task 2: self-supervised pretraining on development captures of BOTH rooms.

    python scripts/30_task2_ssl_train.py --kind dynamic --seeds 0 1 2 [--epochs 100]

No activity labels and no room labels are used: the sampler draws development
captures uniformly from the pooled list, and the loss only needs capture identity.
Validation captures are never loaded here.
Saves artifacts/encoder_{kind}_s{seed}.pt (+ input scale) and a training log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402

from stride.data import Dataset  # noqa: E402
from stride.ssl import fit_input_scale, pretrain  # noqa: E402
from stride.utils import ensure_dir, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="dyn_dn", choices=["dyn_dn", "dynamic", "raw"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ds = Dataset()
    P = ds.cfg["paths"]
    dev = ds.idx(split="development")                 # labels/rooms are NOT read below
    caps = [ds.captures[c] for c in ds.man.capture_id.values[dev]]
    scale = fit_input_scale(ds.W[dev], args.kind)
    out = ensure_dir(P["artifacts"])
    for seed in args.seeds:
        set_seed(seed)
        logf = open(out / f"ssl_{args.kind}_s{seed}.log", "w")
        def log(msg):
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()
        log(f"kind={args.kind} seed={seed} captures={len(caps)} scale={scale:.4f}")
        enc = pretrain(caps, args.kind, scale, epochs=args.epochs, seed=seed, log=log)
        torch.save({"state_dict": enc.state_dict(), "kind": args.kind, "scale": scale, "seed": seed,
                    "epochs": args.epochs}, out / f"encoder_{args.kind}_s{seed}.pt")
        logf.close()
    with open(out / f"ssl_{args.kind}_meta.json", "w") as fh:
        json.dump({"kind": args.kind, "seeds": args.seeds, "epochs": args.epochs, "scale": scale,
                   "n_dev_captures": len(caps)}, fh, indent=1)


if __name__ == "__main__":
    main()
