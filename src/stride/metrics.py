"""Metrics and result-row helpers (challenge §10/§11 output contract)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .utils import git_hash

REQUIRED = ["experiment", "target_room", "K", "metric", "value"]


def macro_f1(y, p, labels=None):
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


def activity_metrics(y, p, labels, exclusive=None, shared=None) -> dict:
    out = {"macro_f1": macro_f1(y, p, labels), "balanced_accuracy": balanced_accuracy_score(y, p),
           "accuracy": accuracy_score(y, p)}
    if exclusive:
        out["exclusive_macro_f1"] = macro_f1(y, p, exclusive)
    if shared:
        out["shared_macro_f1"] = macro_f1(y, p, shared)
    return out


class ResultLog:
    """Append-only experiment log with the required columns first."""

    def __init__(self, path, cfg_hash: str):
        self.path = Path(path)
        self.cfg_hash = cfg_hash
        self.rows = []

    def add(self, experiment, target_room, K, metric, value, **extra):
        self.rows.append(dict(experiment=experiment, target_room=target_room, K=K, metric=metric,
                              value=float(value), **extra, config_hash=self.cfg_hash, git_hash=git_hash()))

    def save(self, replace_experiments=True):
        new = pd.DataFrame(self.rows)
        if self.path.exists() and replace_experiments:
            old = pd.read_csv(self.path)
            old = old[~old.experiment.isin(new.experiment.unique())]
            new = pd.concat([old, new], ignore_index=True)
        cols = REQUIRED + [c for c in new.columns if c not in REQUIRED]
        new[cols].to_csv(self.path, index=False)
        return new


def mean_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
