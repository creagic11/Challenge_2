"""Generate report/numbers.tex and report/tab_*.tex from results CSVs.

    python scripts/60_report_numbers.py --rep S --variant room-align+LR

Every number printed in the report comes from here, so it is traceable to results/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stride.utils import ensure_dir, load_config  # noqa: E402


def f3(x):
    return f"{x:.3f}"


def pm(m, s):
    return f"{m:.3f}\\,$\\pm$\\,{s:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="Hd")
    ap.add_argument("--variant", default="room-align+LR")
    args = ap.parse_args()
    cfg = load_config()
    res = Path(cfg["paths"]["results"])
    out = ensure_dir(Path(cfg["paths"]["results"]).parent / "report")
    N = {}
    # ---------------- QC
    qs = pd.read_csv(res / "qc_summary.csv", index_col=0)
    for room in "AB":
        for k in ["n_decoded", "n_valid", "rej_fixed_pattern", "rej_partial_band"]:
            N[f"qc{room}{k.replace('_', '')}"] = f"{int(float(qs.loc[k, room])):,}"
    qc = pd.read_csv(res / "qc_captures.csv", keep_default_na=False)
    qc = qc[qc.excluded.astype(str) != "True"]
    N["qcAvalidmed"] = f"{int(qc[qc.room == 'A'].n_valid.median())}"
    N["qcBvalidmed"] = f"{int(qc[qc.room == 'B'].n_valid.median())}"
    N["qcAvalidmax"] = f"{int(qc[qc.room == 'A'].n_valid.max())}"
    # ---------------- Task 1
    t1 = pd.read_csv(res / "task1_window_length.csv")
    st = t1[t1.features == "static"].set_index("L").val_accuracy
    N["tOneLonefifty"] = f3(st.loc[150])
    N["tOneLone"] = f3(st.loc[1])
    N["tOneLfive"] = f3(st.loc[5])
    au = pd.read_csv(res / "task1_audit.csv").set_index("feature_set").val_accuracy
    for k, v in {"static_profile_all256": "auStatic", "dynamic_motion_energy": "auDyn",
                 "texture_autocorr_probe": "auTex", "quantisation_probe": "auQuant"}.items():
        N[v] = f3(au.loc[k])
    # ---------------- Task 2
    t2 = pd.read_csv(res / "task2_probe_by_seed.csv")
    reps = [r for r in ["H", "Hd", "R", "S-raw", "S"] if r in set(t2.rep)]
    names = {"H": "Handcrafted (H)", "Hd": "Handcrafted, denoised (H$_d$)", "R": "Random-weight CNN (R)",
             "S-raw": "SSL, raw log-amplitude (S$_{raw}$)", "S": "\\textbf{STRIDE-SSL} (S)"}
    lines = []
    for r in reps:
        g = t2[t2.rep == r]
        cells = []
        for c in ["A_macro_f1", "B_macro_f1", "joint_macro_f1", "xfer_BtoA_shared_f1", "xfer_AtoB_shared_f1"]:
            cells.append(pm(g[c].mean(), g[c].std(ddof=0)) if len(g) > 1 else f3(g[c].mean()))
        lines.append(f"{names[r]} & " + " & ".join(cells) + " \\\\")
    (out / "tab_task2.tex").write_text("\n".join(lines) + "\n")
    # ---------------- Task 3
    R = pd.read_csv(res / "task3_runs.csv")
    head = R[(R.rep == args.rep) & (R.variant == args.variant)]
    lines = []
    for tgt in "AB":
        src = "B" if tgt == "A" else "A"
        for col, nm in [("macro_f1", "macro-F1 (all)"), ("exclusive_macro_f1", "macro-F1 exclusive"),
                        ("shared_macro_f1", "macro-F1 shared"), ("balanced_accuracy", "balanced acc.")]:
            g = head[head.target_room == tgt].groupby("K")[col]
            cells = [pm(m, s) for m, s in zip(g.mean(), g.std(ddof=0))]
            lines.append(f"{src}$\\to${tgt} & {nm} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    (out / "tab_task3_main.tex").write_text("\n".join(lines[:-1]) + "\n")
    # ablation table: mean macro-F1 over K for each variant x rep, per target
    abl = R.groupby(["variant", "rep", "target_room"]).macro_f1.mean().unstack("target_room")
    abl["mean"] = abl.mean(axis=1)
    order = ["LR-naive", "strict(no-unlabelled)", "no-routing", "room-align", "room-align+LR", "SAPT-UDA",
             "SAPT-UDA+LR", "SAPT-OT", "SAPT-OT+LR", "room-OT+LR", "shared-align-norefine", "SAPT-UDA+LR-ST",
             "SAPT-strict+LR", "no-routing+LR"]
    lines = []
    for v in order:
        if v not in abl.index.get_level_values(0):
            continue
        cells = []
        for r in ["H", "Hd", "S", "S+Hd"]:
            if (v, r) in abl.index:
                cells.append(f"{abl.loc[(v, r), 'A']:.3f} / {abl.loc[(v, r), 'B']:.3f}")
            else:
                cells.append("--")
        vv = v.replace("+", "\\,+\\,")
        lines.append(f"{vv} & " + " & ".join(cells) + " \\\\")
    (out / "tab_task3_ablation.tex").write_text("\n".join(lines) + "\n")
    # routing numbers (K=5 and mean over K)
    for tag, v in [("Pred", args.variant), ("Orac", args.variant + "-oracle"), ("None", "no-routing+LR")]:
        for tgt in "AB":
            g = R[(R.rep == args.rep) & (R.variant == v) & (R.target_room == tgt)]
            if len(g):
                N[f"rout{tag}{tgt}"] = f3(g.macro_f1.mean())
    for tgt in "AB":
        g = head[head.target_room == tgt]
        for K in [1, 5, 20]:
            N[f"head{tgt}K{['', 'one', '', '', '', 'five'][K] if K < 6 else 'twenty'}"] = f3(g[g.K == K].macro_f1.mean())
    N["headRep"] = {"Hd": "H$_d$", "S+Hd": "S+H$_d$"}.get(args.rep, args.rep)
    N["headVariant"] = args.variant
    N["nRuns"] = str(int(head.groupby(["target_room", "K"]).size().min()))
    # denoising ablation (headline variant): H vs Hd
    for rep, tag in [("H", "Raw"), ("Hd", "Den")]:
        g = R[(R.rep == rep) & (R.variant == args.variant)].groupby("target_room")
        for tgt in "AB":
            N[f"dn{tag}Shared{tgt}"] = f3(g.shared_macro_f1.mean()[tgt])
            N[f"dn{tag}Macro{tgt}"] = f3(g.macro_f1.mean()[tgt])
    # per-class P/R/F1 at K=5 for the headline representation
    lines = []
    for tgt in "AB":
        pc = pd.read_csv(res / f"task3_perclass_K5_{args.rep}_{tgt}.csv")
        for r in pc.itertuples():
            kind = "shared" if r.label in ("sit", "stand", "walk") else "exclusive"
            lines.append(f"{tgt} & {r.label} ({kind}) & {r.precision:.2f} & {r.recall:.2f} & {r.f1:.2f} \\\\")
        lines.append("\\midrule")
    (out / "tab_perclass.tex").write_text("\n".join(lines[:-1]) + "\n")
    # QC table
    qcs = pd.read_csv(res / "qc_summary.csv", index_col=0)
    rows = [("decoded observations", "n_decoded"), ("rejected: seq.\\ 0xFFFF fixed pattern", "rej_fixed_pattern"),
            ("rejected: narrow-band (20\\,MHz) frame", "rej_partial_band"),
            ("rejected: duplicate / non-finite / zero", None), ("valid full-band observations", "n_valid")]
    lines = []
    for name, key in rows:
        if key is None:
            vals = [int(sum(float(qcs.loc[k, r]) for k in ["rej_duplicate_obs", "rej_non_finite", "rej_all_zero"]))
                    for r in "AB"]
        else:
            vals = [int(float(qcs.loc[key, r])) for r in "AB"]
        lines.append(f"{name} & {vals[0]:,} & {vals[1]:,} \\\\")
    (out / "tab_qc.tex").write_text("\n".join(lines) + "\n")
    # Task-1 audit table
    names = {"static_profile_all256": "static profile $s$ (256)", "static_profile_data_bins": "static profile, data bins (242)",
             "dynamic_motion_energy": "motion-energy profile (242)", "texture_autocorr_probe": "temporal-texture probe (4)",
             "quantisation_probe": "quantisation/level probe (3)"}
    au = pd.read_csv(res / "task1_audit.csv")
    (out / "tab_audit.tex").write_text("\n".join(f"{names[r.feature_set]} & {r.val_accuracy:.3f} \\\\" for r in au.itertuples()) + "\n")
    # inspection statistics (scripts/01_data_inspection.py)
    ins = pd.read_csv(res / "inspection_stats.csv")
    g = lambda st, room="A", lab="all": float(ins[(ins.statistic == st) & (ins.room == room) & (ins.label == lab)].value.iloc[0])
    N["insCorrFixed"] = f"{g('cross_capture_profile_corr_0xFFFF'):.2f}"
    N["insCorrNormal"] = f"{g('cross_capture_profile_corr_normal'):.2f}"
    N["insAmpRatio"] = f"{g('median_amplitude_ratio_normal_over_0xFFFF'):.0f}"
    for lab, k in [("sit", "Sit"), ("stand", "Stand"), ("lying", "Lying"), ("walk", "Walk"), ("empty", "Empty")]:
        N[f"insNb{k}"] = f"{100 * g('narrowband_share_of_non_fixed_rows', 'B', lab):.1f}"
        N[f"insMeRaw{k}"] = f"{g('motion_energy_naive_median', 'B', lab):.2f}"
        N[f"insMeDn{k}"] = f"{g('motion_energy_denoised_median', 'B', lab):.2f}"
    N["insLagA"] = f"{g('walk_pc_lag1_native', 'A', 'walk'):.2f}"
    N["insLagBfour"] = f"{g('walk_pc_lag1_decimated_x4', 'B', 'walk'):.2f}"
    N["insLagBone"] = f"{g('walk_pc_lag1_decimated_x1', 'B', 'walk'):.2f}"
    # window-rule ablation (scripts/12_window_rule_ablation.py)
    ab = pd.read_csv(res / "ablation_window_rule.csv")
    t3 = ab[ab.task == "task3"].groupby(["mode", "target_room", "metric"]).value.mean()
    for mode, mk in [("full", "Full"), ("consecutive", "Consec")]:
        for tgt in "AB":
            N[f"win{mk}{tgt}"] = f3(t3[(mode, tgt, "macro_f1")])
            N[f"win{mk}Sh{tgt}"] = f3(t3[(mode, tgt, "shared_macro_f1")])
    tex = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in N.items())
    (out / "numbers.tex").write_text(tex + "\n")
    print(tex)


if __name__ == "__main__":
    main()
