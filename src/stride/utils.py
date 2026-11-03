"""Shared helpers: configuration loading, path resolution, seeding, hashing."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml

# abspath (not resolve) so that a symlinked checkout still finds ../datasets next to it
REPO_ROOT = Path(os.path.abspath(__file__)).parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the YAML config; resolve relative paths against the repo root."""
    path = Path(path) if path else REPO_ROOT / "configs" / "experiment.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: str(resolve(v)) for k, v in cfg["paths"].items()}
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def resolve(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path(os.path.normpath(REPO_ROOT / p))


def config_hash(cfg: dict) -> str:
    """Short, stable hash of the configuration (excluding private keys)."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha1(json.dumps(clean, sort_keys=True).encode()).hexdigest()[:10]


def git_hash() -> str:
    """Current git commit of the repo, or 'nogit' if unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_file(path: str | os.PathLike) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
