"""One-command setup of everything run_episode.py needs, built with the local library versions.

    python scripts/prepare_deployment.py

Runs: splits -> decode/QC/intermediate (no CSV export) -> windows (no .npy export) -> Task-1 router
-> deployment model (denoised handcrafted features + room-wise alignment + LR head, K=20).
Uses development data only for fitting; takes a few minutes on a laptop CPU.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
steps = [
    ["scripts/split_exp3.py"],
    ["scripts/10_build_intermediate.py", "--no-csv"],
    ["scripts/11_build_windows.py", "--no-npy"],
    ["scripts/20_task1_router.py"],
    ["scripts/45_build_deployment.py", "--rep", "Hd", "--variant", "room-align+LR", "--K", "20"],
]
for s in steps:
    print(">>", " ".join(s), flush=True)
    subprocess.run([sys.executable, str(ROOT / s[0])] + s[1:], check=True, cwd=ROOT)
print("deployment ready: python run_episode.py --episode <dir> --out predictions.csv")
