# STRIDE — Cross-Room Few-Shot WiFi-CSI Activity Recognition (CS60055 Hackathon 2)

**STRIDE** = *Static-profile Routing with Invariant Dynamics Encoding*. A fully Python pipeline from
raw Nexmon PCAP (Room A) and amplitude CSV (Room B) to room identification, self-supervised
representation learning, and few-label cross-room activity recognition.

> **Clean-room statement.** All preprocessing is our own Python code, written from the public Nexmon
> CSI format documentation. The dataset authors' MATLAB code, `.mat` files, notebook and result
> folders were never extracted or opened. The dataset is not part of this repository.

## Layout
```
Hackathon2/
├── datasets/            # read-only source data (NOT in git)
│   ├── exp_2_modified/  # Room A: realdata/input_data/30ms/csidata/... (.pcap)
│   └── exp_3_modified/  # Room B: Data/*.csv
├── derived/             # everything generated (NOT in git)
└── code_files/          # this repository
    ├── configs/         # experiment.yaml + committed split manifests
    ├── src/stride/      # pipeline + models
    ├── scripts/         # numbered pipeline stages
    ├── tests/           # decoder / pipeline tests
    ├── results/         # CSV outputs (QC, manifests, metrics)
    ├── docs/            # data findings, walkthrough
    └── run_episode.py   # end-to-end entry point (coming)
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Reproduce
```bash
python scripts/split_exp3.py              # capture-level splits (Room-B validation: seed 42)
python scripts/10_build_intermediate.py   # decode + QC + common 256-column CSVs
python scripts/11_build_windows.py        # official (150, 256) windows + provenance
python -m pytest -q tests                 # decoder and pipeline tests
```
See `docs/data_findings.md` for what the data inspection revealed and why each QC rule exists.
