"""Task 1: room identification + window-length curve + shortcut audit.

    python scripts/20_task1_router.py

C is chosen on the internal dev hold-out; the final router is refit on all
development windows and scored on the fixed validation windows.
Outputs: results/experiments.csv (task1_* rows), results/task1_*.csv,
artifacts/router.pkl, figures/task1_window_length.png
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from stride.cleaning import DATA_BINS  # noqa: E402
from stride.data import Dataset  # noqa: E402
from stride.metrics import ResultLog  # noqa: E402
from stride.router import (RoomRouter, centre_crop, dynamic_features, make_router,  # noqa: E402
                           quantisation_features, static_features, texture_features)
from stride.utils import ensure_dir  # noqa: E402

LENGTHS = [1, 2, 5, 10, 25, 50, 100, 150]


def main():
    ds = Dataset()
    cfg = ds.cfg
    P = cfg["paths"]
    log = ResultLog(Path(P["results"]) / "experiments.csv", cfg["_config_hash"])
    tr, ho = ds.idx(split="development", holdout=False), ds.idx(split="development", holdout=True)
    dev, val = ds.idx(split="development"), ds.idx(split="validation")
    y = ds.man.room.values
    W = ds.W

    # --- choose C on the internal dev hold-out (static features, L=150)
    Fs = static_features(W)
    best = max([0.01, 0.1, 1.0, 10.0],
               key=lambda C: accuracy_score(y[ho], make_router(C).fit(Fs[tr], y[tr]).predict(Fs[ho])))
    print("chosen C:", best)

    # --- headline: full router at L=150, refit on all dev, scored on validation
    router = RoomRouter(best).fit(W[dev], y[dev])
    pv = router.predict(W[val])
    acc = accuracy_score(y[val], pv)
    log.add("task1_room_id", "both", "NA", "accuracy", acc, variant="static_profile_L150", C=best)
    for room in "AB":
        m = y[val] == room
        log.add("task1_room_id", room, "NA", "accuracy", accuracy_score(y[val][m], pv[m]),
                variant="static_profile_L150")
    print(f"Task 1 validation accuracy (L=150): {acc:.4f}")
    ensure_dir(P["artifacts"])
    with open(Path(P["artifacts"]) / "router.pkl", "wb") as fh:
        pickle.dump(router, fh)
    pd.DataFrame(dict(window_id=ds.man.window_id.values[val], room=pv,
                      label="", confidence=router.predict_proba(W[val]).max(1))).to_csv(
        Path(P["results"]) / "task1_predictions.csv", index=False)

    # --- audit: which information separates the rooms? (all at L=150)
    feats = {
        "static_profile_all256": static_features(W),
        "static_profile_data_bins": static_features(W, DATA_BINS),
        "dynamic_motion_energy": dynamic_features(W),
        "texture_autocorr_probe": texture_features(W),
        "quantisation_probe": quantisation_features(W),
    }
    audit = []
    for name, F in feats.items():
        m = make_router(best).fit(F[dev], y[dev])
        a = accuracy_score(y[val], m.predict(F[val]))
        audit.append(dict(feature_set=name, n_features=F.shape[1], val_accuracy=a))
        log.add("task1_audit", "both", "NA", "accuracy", a, variant=name)
    audit = pd.DataFrame(audit)
    audit.to_csv(Path(P["results"]) / "task1_audit.csv", index=False)
    print(audit)

    # --- window-length curve (same validation capture pool, centred sub-windows)
    curve = []
    for L in LENGTHS:
        Wd, Wv = centre_crop(W[dev], L), centre_crop(W[val], L)
        for name, fn in [("static", static_features), ("dynamic", dynamic_features)]:
            if name == "dynamic" and L < 3:
                continue
            m = make_router(best).fit(fn(Wd), y[dev])
            a = accuracy_score(y[val], m.predict(fn(Wv)))
            curve.append(dict(L=L, features=name, val_accuracy=a))
            log.add("task1_window_length", "both", "NA", "accuracy", a, variant=f"{name}_L{L}", L=L)
    curve = pd.DataFrame(curve)
    curve.to_csv(Path(P["results"]) / "task1_window_length.csv", index=False)
    print(curve.pivot(index="L", columns="features", values="val_accuracy"))
    log.save()

    ensure_dir(P["figures"])
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for name, mk in [("static", "o-"), ("dynamic", "s--")]:
        c = curve[curve.features == name]
        ax.plot(c.L, c.val_accuracy, mk, label=f"{name} profile", ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("window length L (observations)")
    ax.set_ylabel("room accuracy (validation)")
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(P["figures"]) / "task1_window_length.png", dpi=200)


if __name__ == "__main__":
    main()
