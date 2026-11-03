"""Create the capture-level split manifests (run once; results are committed).

    python scripts/split_exp3.py [--materialize]

Writes configs/splits/captures.csv (both rooms, with dev hold-out flags) and
configs/splits/roomB_validation.csv. With --materialize, also copies Room B
into the challenge §9 folder layout under derived/raw/experiment3/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from stride.splits import add_dev_holdout, assert_disjoint, list_room_a, split_room_b  # noqa: E402
from stride.utils import ensure_dir, load_config  # noqa: E402

FOLDER = {"empty": "Empty", "lying": "Lying", "sit": "Sitting", "stand": "Standing", "walk": "Walking"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialize", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    a = list_room_a(cfg["paths"]["room_A_raw"])
    b = split_room_b(cfg["paths"]["room_B_raw"], cfg["data"]["room_B_patterns"],
                     cfg["data"]["room_B_validation_per_class"], cfg["seeds"]["roomB_validation"])
    df = pd.concat([a, b], ignore_index=True)
    df = add_dev_holdout(df, cfg["data"]["dev_holdout_fraction"], cfg["seeds"]["dev_holdout"])
    assert_disjoint(df)
    out = ensure_dir(cfg["paths"]["splits"])
    df.to_csv(out / "captures.csv", index=False)
    bv = b[b.split == "validation"][["label", "source_relpath"]].sort_values(["label", "source_relpath"])
    bv.to_csv(out / "roomB_validation.csv", index=False)
    print(df.groupby(["room", "split", "label"]).size().unstack(0).fillna(0).astype(int))
    print("dev hold-out captures:", int(df.dev_holdout.sum()))
    if args.materialize:
        base = ensure_dir(Path(cfg["paths"]["derived"]) / "raw" / "experiment3")
        src = Path(cfg["paths"]["room_B_raw"])
        for r in b.itertuples():
            dst = base / "development" if r.split == "development" else base / "validation" / FOLDER[r.label]
            ensure_dir(dst)
            shutil.copy2(src / r.source_relpath, dst / r.source_relpath)
        print("materialised Room-B split under", base)


if __name__ == "__main__":
    main()
