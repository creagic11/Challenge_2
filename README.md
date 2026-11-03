# STRIDE — Cross-Room Few-Label WiFi-CSI Activity Recognition (CS60055 Hackathon 2)

**STRIDE** = *Static-profile Routing with Invariant Dynamics Encoding*. A fully Python pipeline from
raw Nexmon PCAP (Room A, Experiment 2) and amplitude CSV (Room B, Experiment 3) to room identification
(Task 1), self-supervised representation learning (Task 2) and few-label cross-room activity recognition
(Task 3). Technical report: [`report/report.pdf`](report/report.pdf).

> **Clean-room statement.** All preprocessing is our own Python code, written from the public Nexmon CSI
> format documentation and validated against physical properties of the signal. The dataset authors'
> MATLAB code, `.mat` files, notebook and result folders were never extracted or opened. The dataset is not
> part of this repository.

## Headline validation results (fixed validation sets)
| Task | Result |
|---|---|
| 1 — room identification (L = 150) | accuracy 1.000 (0.949 from a single observation) |
| 2 — linear probe, frozen SSL encoder (S_raw) | macro-F1 0.872 (Room A), 0.987 (Room B); handcrafted 0.752 / 0.840; random CNN 0.553 / 0.887 |
| 3 — few-label cross-room, K = 20 | macro-F1 0.341 (B→A), 0.418 (A→B); see `report/` for all K, ablations and routing study |

All numbers are produced by the scripts below and stored in `results/`; the report's numbers and tables
are generated from those CSVs by `scripts/60_report_numbers.py`.

## Layout
```
Hackathon2/
├── datasets/            # read-only source data (NOT in git)
│   ├── exp_2_modified/  # Room A: realdata/input_data/30ms/csidata/... (.pcap, TA package)
│   └── exp_3_modified/  # Room B: Data/*.csv (TA package)
├── derived/             # everything generated: intermediate CSVs, windows, caches (NOT in git)
└── code_files/          # this repository
    ├── configs/         # experiment.yaml + committed split manifests
    ├── src/stride/      # pipeline (pcap_reader, csi_decode, cleaning, amplitude, windowing, ...)
    │                    # and models (router, models/encoder, ssl, features, fewshot)
    ├── scripts/         # numbered stages (see below)
    ├── tests/           # decoder, pipeline-contract and leakage tests
    ├── results/         # CSV outputs (QC, manifests, metrics, predictions)
    ├── figures/         # required figures
    ├── artifacts/       # router, encoders, deployment model, SSL logs
    ├── docs/            # data_findings.md (what inspection revealed), walkthrough.md (viva prep)
    ├── report/          # LaTeX source + PDF
    └── run_episode.py   # end-to-end entry point
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
Paths to the data are relative (`configs/experiment.yaml`): `../datasets/...` next to this folder.

## Reproduce everything (CPU only)
```bash
python scripts/split_exp3.py               # capture-level splits (Room-B validation: sorted names, seed 42)
python scripts/10_build_intermediate.py    # decode + QC + harmonise + intermediate CSVs   (~1 min)
python scripts/11_build_windows.py         # official (150, 256) windows + provenance      (~1 min)
python scripts/01_data_inspection.py       # statistics quoted in docs/ and the report
python scripts/12_window_rule_ablation.py  # full-capture vs consecutive window rule
python scripts/20_task1_router.py          # Task 1 + audit + window-length curve
python scripts/30_task2_ssl_train.py --kind dyn_dn --seeds 0 1 2   # SSL (~30 min/seed on CPU)
python scripts/30_task2_ssl_train.py --kind raw    --seeds 0 1 2
python scripts/31_task2_probe.py           # Task 2 probes (+ caches embeddings)
python scripts/40_task3_fewshot.py --reps H Hd Hdyn R S-raw S S+Hd S+Hdyn   # Task 3 grid
python scripts/45_build_deployment.py --rep Hd --variant room-align+LR      # model for run_episode
python scripts/50_make_figures.py          # figures/
python scripts/60_report_numbers.py        # report/numbers.tex + tables
python -m pytest -q tests                  # 18 tests
```
(`S+Hd`, `S+Hdyn` embeddings are concatenations of the cached `S` and `Hd`/`Hdyn` features; see
`scripts/31`/`40` and the report.)

## Run an episode
The committed `artifacts/deployment.pkl` contains only NumPy objects, so it loads under any library
version. To rebuild it (and the router) from the raw data on a new machine:
```bash
python scripts/prepare_deployment.py      # splits -> QC -> windows -> router -> deployment (dev data only)
```
Then:
```bash
python run_episode.py --episode path/to/episode_dir [--support support.csv] --out predictions.csv
```
The episode folder may contain raw `.pcap`, raw `.csv` and/or `window_*.npy` (150×256) files. Raw files go
through exactly the same pipeline as development/validation data; the file format only selects the reader.
`support.csv` (columns `file,label`) optionally provides labelled target captures for the target-exclusive
classes. Output columns: `window_id, room, label, confidence`.

## Output contract
* `results/experiments.csv` — `experiment, target_room, K, metric, value` (+ representation, variant, seed, hashes)
* `results/predictions/task3_target{A,B}_K{1,5,10,20}.csv` — `window_id, room, label, confidence`
* `results/windows_manifest.csv` — provenance of every window (raw file, SHA-1, method, config hash)

## Acknowledgements and AI use
Dataset: CSI Human Activity, IEEE DataPort (Schäfer et al., Applied Sciences 2021). Libraries: NumPy, SciPy,
pandas, scikit-learn, PyTorch, matplotlib, umap-learn. AI assistance (Claude, Anthropic) was used for code,
analysis and drafting, as disclosed in the report.
