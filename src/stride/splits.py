"""Capture-level development/validation splits (challenge §9).

Room A: the supplied `validation/` folder is the validation set; everything
        else is development. Files are never moved.
Room B: per activity, candidates = files matching the documented pattern,
        sorted with Python's sorted() (code-point order), then
        random.Random(42).sample(candidates, 25) with a fresh generator per
        activity. All remaining files (including files that do not match the
        documented patterns) form development.
A further *internal development hold-out* (stratified, seed 0) is used for all
tuning so that validation is touched only for reported results.
"""
from __future__ import annotations

import fnmatch
import os
import random
from pathlib import Path

import pandas as pd

ROOM_A_CLASSES = ["fall", "sit", "sitdown", "stand", "standup", "walk"]
ROOM_B_PREFIX = {"Empty": "empty", "Lying": "lying", "Sitting": "sit",
                 "Standing": "stand", "Walking": "walk"}


def room_b_label(fname: str) -> str:
    return ROOM_B_PREFIX[fname.split("_")[0]]


def list_room_a(raw_dir: str) -> pd.DataFrame:
    rows = []
    for cls in ROOM_A_CLASSES:
        for split, d in (("development", Path(raw_dir) / cls),
                         ("validation", Path(raw_dir) / "validation" / cls)):
            for f in sorted(os.listdir(d)):
                if f.endswith(".pcap"):
                    rel = os.path.relpath(d / f, raw_dir).replace(os.sep, "/")
                    rows.append(dict(room="A", label=cls, split=split, source_relpath=rel,
                                     capture_id=f"A/{rel[:-5]}"))
    return pd.DataFrame(rows)


def split_room_b(raw_dir: str, patterns: dict, n_val: int = 25, seed: int = 42) -> pd.DataFrame:
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".csv"))
    val = set()
    for label, pat in patterns.items():
        cands = sorted(f for f in files if fnmatch.fnmatchcase(f, pat))
        val.update(random.Random(seed).sample(cands, n_val))
    rows = [dict(room="B", label=room_b_label(f), split="validation" if f in val else "development",
                 source_relpath=f, capture_id=f"B/{f[:-4]}",
                 pattern_match=any(fnmatch.fnmatchcase(f, p) for p in patterns.values()))
            for f in files]
    return pd.DataFrame(rows)


def add_dev_holdout(df: pd.DataFrame, frac: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """Mark a stratified (room, label) fraction of development captures as internal hold-out."""
    df = df.copy()
    df["dev_holdout"] = False
    dev = df[df.split == "development"]
    for (_, _), g in dev.groupby(["room", "label"]):
        ids = sorted(g.capture_id)
        k = int(round(frac * len(ids)))
        chosen = set(random.Random(seed).sample(ids, k))
        df.loc[df.capture_id.isin(chosen), "dev_holdout"] = True
    return df


def assert_disjoint(df: pd.DataFrame) -> None:
    dup = df.groupby("capture_id").split.nunique()
    assert (dup == 1).all(), f"captures in more than one split: {list(dup[dup > 1].index)}"
    assert not df[(df.split == "validation") & df.get("dev_holdout", False)].shape[0], \
        "validation capture marked as dev hold-out"
