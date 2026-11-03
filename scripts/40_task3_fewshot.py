"""Task 3: cross-room few-label adaptation (A->B and B->A, K in {1,5,10,20}).

    python scripts/40_task3_fewshot.py [--reps H S S-raw R]

Uses frozen features cached by 31_task2_probe.py (derived/cache/emb_<rep>_s<seed>.npz).
Hyper-parameters (tau, anchor, iterations) are FIXED a priori (no tuning on target
labels; shared-class target labels are never available). 10 nested support episodes.
Outputs: results/task3_runs.csv (every run), results/task3_summary.csv,
results/experiments.csv (task3 rows), results/task3_confusion_K5_*.csv,
results/predictions/task3_<dir>_K<k>.csv (headline predictions, episode 0).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from stride.data import Dataset  # noqa: E402
from stride.fewshot import CLASSES, EXCLUSIVE, SAPT, SHARED, SAPTConfig, sample_supports  # noqa: E402
from stride.metrics import ResultLog, activity_metrics  # noqa: E402
from stride.utils import ensure_dir  # noqa: E402

VARIANTS = {
    # name: (SAPTConfig kwargs)
    "SAPT-UDA": dict(align="shared", use_pool=True, refine=True, routing="predicted"),
    "SAPT-UDA-oracle": dict(align="shared", use_pool=True, refine=True, routing="oracle"),
    "no-routing": dict(align="none", use_pool=True, refine=True, routing="none"),
    "room-align": dict(align="room", use_pool=True, refine=True, routing="predicted"),
    "shared-align-norefine": dict(align="shared", use_pool=True, refine=False, routing="predicted"),
    "strict(no-unlabelled)": dict(align="none", use_pool=False, refine=False, routing="predicted"),
    "SAPT-UDA+LR": dict(align="shared", use_pool=True, refine=True, routing="predicted", head="lr"),
    "SAPT-UDA+LR-ST": dict(align="shared", use_pool=True, refine=True, routing="predicted", head="lr_st"),
    "room-align+LR": dict(align="room", use_pool=True, refine=True, routing="predicted", head="lr"),
    "SAPT-strict+LR": dict(align="shared", use_pool=False, refine=False, routing="predicted", head="lr"),
    "SAPT-OT": dict(align="shared", use_pool=True, refine=True, balanced=True, routing="predicted"),
    "SAPT-OT+LR": dict(align="shared", use_pool=True, refine=True, balanced=True, routing="predicted", head="ot_lr"),
    "room-OT+LR": dict(align="room", use_pool=True, refine=True, balanced=True, routing="predicted", head="ot_lr"),
    "room-align+LR-oracle": dict(align="room", use_pool=True, refine=True, routing="oracle", head="lr"),
    "no-routing+LR": dict(align="none", use_pool=True, refine=True, routing="none", head="lr"),
}
HEADLINE = "room-align+LR"   # chosen by mean validation macro-F1 over both directions and all K (declared)


def lr_baseline(Zdev_all, Zs, ys, Zsup, ysup, Zq):
    """Naive transfer: logistic regression on source shared + support windows (no adaptation)."""
    m = np.isin(ys, SHARED)
    X = np.r_[Zs[m], Zsup]
    y = np.r_[ys[m], ysup]
    clf = make_pipeline(StandardScaler().fit(Zdev_all), LogisticRegression(C=1.0, max_iter=5000))
    clf.fit(X, y)
    P = clf.predict_proba(Zq)
    return clf.classes_[P.argmax(1)], P.max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=["H", "R", "S-raw", "S"])
    args = ap.parse_args()
    ds = Dataset()
    cfg = ds.cfg
    P = cfg["paths"]
    man = ds.man
    y, room, cid = man.label.values, man.room.values, man.capture_id.values
    with open(Path(P["artifacts"]) / "router.pkl", "rb") as fh:
        router = pickle.load(fh)
    routed = router.predict(ds.W)                    # predicted room for every window
    Ks = cfg["tasks"]["K"]
    episodes = cfg["seeds"]["episodes"]
    cache = Path(P["derived"]) / "cache"
    runs, conf_acc, preds_out = [], {}, {}
    for rep in args.reps:
        for f in sorted(cache.glob(f"emb_{rep}_s*.npz")):
            seed = int(f.stem.split("_s")[-1])
            Z = np.load(f)["F"].astype(np.float64)
            dev_all = ds.idx(split="development")
            for src, tgt in [("B", "A"), ("A", "B")]:
                s_idx = ds.idx(room=src, split="development")
                t_dev = ds.idx(room=tgt, split="development")
                q = ds.idx(room=tgt, split="validation")
                dev_labels = {cid[i]: y[i] for i in t_dev}
                id2row = {cid[i]: i for i in t_dev}
                for ep in episodes:
                    sup = sample_supports(dev_labels, EXCLUSIVE[tgt], Ks, seed=1000 * ep + (0 if tgt == "A" else 1))
                    for K in Ks:
                        si = np.array([id2row[c] for c in sup[K]])
                        for vname, kw in list(VARIANTS.items()) + [("LR-naive", None)]:
                            if kw is None:
                                pl, pc = lr_baseline(Z[dev_all], Z[s_idx], y[s_idx], Z[si], y[si], Z[q])
                            else:
                                c = SAPTConfig(**kw)
                                pool = Z[dev_all] if c.routing == "none" else Z[t_dev]
                                model = SAPT(c).fit(Z[dev_all], Z[s_idx], y[s_idx], src, Z[si], y[si], tgt,
                                                    Zpool=pool)
                                rr = room[q] if c.routing == "oracle" else (
                                    np.full(len(q), tgt) if c.routing == "none" else routed[q])
                                pl, pc = model.predict(Z[q], rr)
                            met = activity_metrics(y[q], pl, CLASSES[tgt], EXCLUSIVE[tgt], SHARED)
                            runs.append(dict(rep=rep, seed=seed, episode=ep, direction=f"{src}->{tgt}",
                                             target_room=tgt, K=K, variant=vname, **met))
                            if vname == HEADLINE and K == 5:
                                cm = confusion_matrix(y[q], pl, labels=CLASSES[tgt])
                                key = (rep, tgt)
                                conf_acc[key] = conf_acc.get(key, 0) + cm
                            if vname == HEADLINE and rep == "Hd" and ep == 0 and seed == 0:
                                preds_out[(tgt, K)] = pd.DataFrame(dict(window_id=man.window_id.values[q],
                                                                        room=routed[q], label=pl, confidence=pc))
            print(f"done rep={rep} seed={seed}", flush=True)
    R = pd.DataFrame(runs)
    res = Path(P["results"])
    R.to_csv(res / "task3_runs.csv", index=False)
    metrics = ["macro_f1", "exclusive_macro_f1", "shared_macro_f1", "balanced_accuracy"]
    S = R.groupby(["rep", "variant", "target_room", "K"])[metrics].agg(["mean", "std", "count"])
    S.to_csv(res / "task3_summary.csv")
    log = ResultLog(res / "experiments.csv", cfg["_config_hash"])
    for (rep, var, tgt, K), g in R.groupby(["rep", "variant", "target_room", "K"]):
        for m in metrics:
            log.add("task3_fewshot", tgt, K, m, g[m].mean(), representation=rep, variant=var,
                    std=g[m].std(), n_runs=len(g))
    log.save()
    for (rep, tgt), cm in conf_acc.items():
        labs = CLASSES[tgt]
        df = pd.DataFrame(cm, index=[f"true_{l}" for l in labs], columns=[f"pred_{l}" for l in labs])
        df.to_csv(res / f"task3_confusion_K5_{rep}_{tgt}.csv")
        # per-class P/R/F1 from the accumulated confusion matrix
        yt = np.repeat(np.repeat(labs, len(labs)), cm.ravel())
        yp = np.repeat(np.tile(labs, len(labs)), cm.ravel())
        p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labs, zero_division=0)
        pd.DataFrame(dict(label=labs, precision=p, recall=r, f1=f, support=s)).to_csv(
            res / f"task3_perclass_K5_{rep}_{tgt}.csv", index=False)
    pdir = ensure_dir(res / "predictions")
    for (tgt, K), df in preds_out.items():
        df.to_csv(pdir / f"task3_target{tgt}_K{K}.csv", index=False)
    head = R[R.variant.isin([HEADLINE, "no-routing", "room-align", "LR-naive", "strict(no-unlabelled)"])]
    print(head.groupby(["rep", "variant", "target_room", "K"]).macro_f1.mean().unstack("K").round(3))


if __name__ == "__main__":
    main()
