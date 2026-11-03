"""Capture-consistency contrastive pretraining (no activity or room labels).

Positive pair = two random crops of the *same capture*, independently augmented.
Uses only capture identity, which the challenge defines as the atomic unit.
Physics-aware augmentations:
  * random time-scale/shift: a random 50-100 % sub-span of the capture timeline, resampled
    to 150 points with the same rule as the official windows
  * contiguous subcarrier-band masking, temporal masking, small Gaussian jitter
  * NO time reversal (sit-down and stand-up are time-reversed versions of each other)
  * NO subcarrier flips (frequency order is physical)
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .models.encoder import Encoder, Projector, nt_xent, prepare_input
from .windowing import resample_uniform


def augment_crop(X: np.ndarray, t: np.ndarray, rng: np.random.Generator, L: int = 150) -> np.ndarray:
    """One random training crop of a capture (never mixes captures)."""
    # same rule as the official 'full' window, on a random 50-100 % sub-span of the
    # capture's timeline (a random time-scale + time-shift augmentation)
    frac = rng.uniform(0.5, 1.0)
    a = t[0] + rng.uniform(0, 1 - frac) * (t[-1] - t[0])
    W = resample_uniform(X, t, L, a, a + frac * (t[-1] - t[0]))
    return np.rint(W).astype(np.float32)


def tensor_augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """x: (B, C, T) model input. Band mask, time mask, jitter."""
    B, C, T = x.shape
    x = x.clone()
    for i in range(B):
        w = int(torch.randint(10, 41, (1,), generator=g))
        s = int(torch.randint(0, C - w, (1,), generator=g))
        x[i, s:s + w] = 0
        tw = int(T * float(torch.empty(1).uniform_(0.1, 0.2, generator=g)))
        ts = int(torch.randint(0, T - tw, (1,), generator=g))
        x[i, :, ts:ts + tw] = 0
    return x + 0.05 * torch.randn(x.shape, generator=g)


def fit_input_scale(W: np.ndarray, kind: str) -> float:
    """Single global robust scale (median |input|) fitted on development windows."""
    x = prepare_input(torch.as_tensor(W, dtype=torch.float32), kind, 1.0)
    return float(x.abs().median().clamp_min(1e-3))


def pretrain(captures: list, kind: str, scale: float, epochs: int = 100, batch: int = 128,
             lr: float = 1e-3, seed: int = 0, log=print) -> Encoder:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    enc, proj = Encoder(), Projector()
    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(captures) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for ep in range(epochs):
        enc.train(); proj.train()
        order = rng.permutation(len(captures))
        tot = 0.0
        for b in range(0, len(order), batch):
            ids = order[b:b + batch]
            if len(ids) < 8:
                continue
            v1 = np.stack([augment_crop(*captures[i], rng) for i in ids])
            v2 = np.stack([augment_crop(*captures[i], rng) for i in ids])
            x1 = tensor_augment(prepare_input(torch.as_tensor(v1), kind, scale), g)
            x2 = tensor_augment(prepare_input(torch.as_tensor(v2), kind, scale), g)
            loss = nt_xent(proj(enc(x1)), proj(enc(x2)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(ids)
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"epoch {ep:3d} loss {tot / len(order):.4f} ({time.time() - t0:.0f}s)")
    return enc
