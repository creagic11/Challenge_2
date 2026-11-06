"""Required figures (challenge §11) from the results CSVs.

    python scripts/50_make_figures.py --rep S --variant SAPT-OT+LR

  figures/task3_label_efficiency.png   macro-F1 vs K (target-exclusive vs transferred shared)
  figures/task3_routing.png            predicted vs oracle vs no routing
  figures/task3_confusion_K5.png       confusion matrices at K=5 (row-normalised)
  figures/task2_embeddings.png         UMAP of validation embeddings, by room / by activity
  figures/qc_observation_types.png     the three observation types found in the raw data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stride.utils import ensure_dir, load_config  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
C_EXCL, C_SHARED, C_ALL = "#2a6f97", "#e07a5f", "#3d405b"


def ci(x):
    x = np.asarray(x)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0


def label_efficiency(R, rep, variant, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, tgt in zip(axes, ["A", "B"]):
        g = R[(R.rep == rep) & (R.variant == variant) & (R.target_room == tgt)]
        for col, lab, c, mk in [("macro_f1", "all classes", C_ALL, "o-"),
                                ("exclusive_macro_f1", "target-labelled (exclusive)", C_EXCL, "s--"),
                                ("shared_macro_f1", "transferred (SIT/STAND/WALK)", C_SHARED, "^:")]:
            a = g.groupby("K")[col].agg(["mean", ci])
            ax.errorbar(a.index, a["mean"], yerr=a["ci"], fmt=mk, color=c, label=lab, ms=4, capsize=2)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        src = "B" if tgt == "A" else "A"
        ax.set_title(f"target room {tgt} ({src}$\\rightarrow${tgt})")
        ax.set_xlabel("K (labelled captures per exclusive class)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1 (validation)")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "task3_label_efficiency.png", dpi=220)


def routing(R, rep, variant, out):
    names = {variant: "predicted room", f"{variant}-oracle": "oracle room",
             ("no-routing+LR" if variant.endswith("+LR") else "no-routing"): "no routing"}
    have = [v for v in names if v in R.variant.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, tgt in zip(axes, ["A", "B"]):
        w = 0.8 / len(have)
        for i, v in enumerate(have):
            g = R[(R.rep == rep) & (R.variant == v) & (R.target_room == tgt)].groupby("K").macro_f1
            m = g.mean()
            ax.bar(np.arange(len(m)) + i * w, m.values, w, yerr=g.agg(ci).values, label=names[v], capsize=2)
        ax.set_xticks(np.arange(4) + w * (len(have) - 1) / 2)
        ax.set_xticklabels([f"K={k}" for k in [1, 5, 10, 20]])
        ax.set_title(f"target room {tgt}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("macro-F1 (validation)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "task3_routing.png", dpi=220)


def confusion(res, rep, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, tgt in zip(axes, ["A", "B"]):
        f = res / f"task3_confusion_K5_{rep}_{tgt}.csv"
        if not f.exists():
            continue
        cm = pd.read_csv(f, index_col=0)
        M = cm.values / cm.values.sum(1, keepdims=True)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        labs = [c.replace("pred_", "") for c in cm.columns]
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=45, ha="right")
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs)
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.5 else "black")
        ax.set_title(f"target room {tgt}, K=5")
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out / "task3_confusion_K5.png", dpi=220)


def embeddings(cfg, reps, out):
    import umap
    man = pd.read_csv(Path(cfg["paths"]["results"]) / "windows_manifest.csv")
    val = (man.split == "validation").values
    fig, axes = plt.subplots(len(reps), 2, figsize=(6.4, 2.5 * len(reps)), squeeze=False)
    acts = sorted(man.label.unique())
    cmap = plt.get_cmap("tab10")
    for row, (rep, title) in enumerate(reps):
        f = Path(cfg["paths"]["derived"]) / "cache" / f"emb_{rep}_s0.npz"
        if not f.exists():
            continue
        Z = np.load(f)["F"][val]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
        E = umap.UMAP(random_state=0, n_neighbors=20, min_dist=0.2).fit_transform(Z)
        for room, mk in [("A", "o"), ("B", "^")]:
            m = man.room.values[val] == room
            axes[row, 0].scatter(E[m, 0], E[m, 1], s=5, marker=mk, label=f"room {room}", alpha=0.7)
            for i, a in enumerate(acts):
                mm = m & (man.label.values[val] == a)
                axes[row, 1].scatter(E[mm, 0], E[mm, 1], s=5, marker=mk, color=cmap(i), alpha=0.7,
                                     label=a if room == "A" or a in ("empty", "lying") else None)
        axes[row, 0].set_title(f"{title}: by room", fontsize=8)
        axes[row, 1].set_title(f"{title}: by activity (o = A, ^ = B)", fontsize=8)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].legend(frameon=False, fontsize=6)
        axes[row, 1].legend(frameon=False, fontsize=6, ncol=2, markerscale=2)
    fig.tight_layout()
    fig.savefig(out / "task2_embeddings.png", dpi=220)


def qc_types(cfg, out):
    from stride.cleaning import scale_free_log
    from stride.ingest import read_pcap_capture
    caps = pd.read_csv(Path(cfg["paths"]["splits"]) / "captures.csv")
    dev = caps[(caps.room == "A") & (caps.split == "development")].iloc[::40]
    full, narrow, fixed = [], [], []
    for r in dev.itertuples():
        c = read_pcap_capture(Path(cfg["paths"]["room_A_raw"]) / r.source_relpath)
        L = scale_free_log(c.amp)
        full.append(L[c.reasons == ""])
        narrow.append(L[c.reasons == "partial_band"])
        fixed.append(L[c.reasons == "fixed_pattern"])
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    sc = np.arange(-128, 128)
    for arr, lab, col in [(full, "valid full-band (80 MHz)", C_ALL), (narrow, "narrow-band frame (rejected)", C_SHARED),
                          (fixed, "seq. 0xFFFF report (rejected)", C_EXCL)]:
        prof = np.fft.fftshift(np.median(np.concatenate(arr), 0))
        ax.plot(sc, prof, color=col, lw=1, label=lab)
    ax.set_xlabel("subcarrier index")
    ax.set_ylabel("median log amplitude\n(rel. to row median)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "qc_observation_types.png", dpi=220, bbox_inches="tight")


def denoise_example(cfg, out):
    """Dynamic part D vs denoised D for development windows of Room B (STAND vs WALK)."""
    from stride.data import Dataset
    from stride.transform import decompose, denoise_dynamic
    from stride.cleaning import DATA_BINS
    ds = Dataset(cfg)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 3.4), sharex=True, sharey=True)
    for col, lab in enumerate(["stand", "walk"]):
        i = ds.idx(room="B", split="development", labels=[lab])[3]
        _, D = decompose(ds.W[i:i + 1])
        D = D[0]
        Dd = denoise_dynamic(D)
        v = np.percentile(np.abs(D[:, DATA_BINS]), 98)
        for row, (M, ttl) in enumerate([(D, "dynamic part $D$"), (Dd, "denoised $\\hat D$ (Gavish-Donoho)")]):
            ax = axes[row, col]
            ax.imshow(M[:, DATA_BINS].T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title(f"Room B {lab.upper()}: {ttl}", fontsize=7)
            if col == 0:
                ax.set_ylabel("data subcarrier")
            if row == 1:
                ax.set_xlabel("time (window sample)")
    fig.tight_layout()
    fig.savefig(out / "denoise_example.png", dpi=220)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="Hd")
    ap.add_argument("--variant", default="room-align+LR")
    ap.add_argument("--skip-umap", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    res = Path(cfg["paths"]["results"])
    out = ensure_dir(cfg["paths"]["figures"])
    R = pd.read_csv(res / "task3_runs.csv")
    label_efficiency(R, args.rep, args.variant, out)
    routing(R, args.rep, args.variant, out)
    confusion(res, args.rep, out)
    qc_types(cfg, out)
    denoise_example(cfg, out)
    if not args.skip_umap:
        embeddings(cfg, [("Hd", "H$_d$"), ("S-raw", "S$_{raw}$"), ("S", "STRIDE-SSL (S)")], out)


if __name__ == "__main__":
    main()
