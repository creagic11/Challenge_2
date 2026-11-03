"""Compact 1-D CNN encoder for (150, 256) CSI windows (~0.15 M parameters).

Subcarriers are channels; a 1x1 convolution learns subcarrier combinations
("spectral PCA"), dilated depthwise-separable temporal blocks model dynamics,
and statistics pooling (mean ‖ std over time) summarises motion energy without
depending on exactly when within the window the activity happens.
The STRIDE transform is computed inside `prepare_input`, so the model always
receives the same representation for every room.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cleaning import DATA_BINS

DATA_IDX = torch.as_tensor(DATA_BINS, dtype=torch.long)


def prepare_input(W: torch.Tensor, kind: str = "dynamic", scale: float = 1.0) -> torch.Tensor:
    """W: (B, T, 256) amplitudes -> (B, C=242, T) model input.

    kind='dyn_dn'  : Gavish-Donoho low-rank denoised D (STRIDE default)
    kind='dynamic' : D = L' - median_t L'   (static profile removed, not denoised)
    kind='raw'     : L' only (gain-normalised log amplitude; static profile kept)
    """
    L = torch.log1p(W.clamp_min(0))
    Ld = L[..., DATA_IDX]
    Lp = Ld - Ld.median(dim=-1, keepdim=True).values
    if kind in ("dynamic", "dyn_dn"):
        Lp = Lp - Lp.median(dim=-2, keepdim=True).values
    elif kind != "raw":
        raise ValueError(kind)
    if kind == "dyn_dn":
        Lp = denoise_batch(Lp)
    return (Lp / scale).transpose(1, 2).contiguous()


def denoise_batch(D: torch.Tensor) -> torch.Tensor:
    """Batched Gavish-Donoho hard-threshold denoising of (B, T, C) dynamic parts."""
    mu = D.mean(dim=1, keepdim=True)
    U, S, Vh = torch.linalg.svd(D - mu, full_matrices=False)
    T, C = D.shape[1], D.shape[2]
    m, n = min(T, C), max(T, C)
    b = m / n
    w = 0.56 * b ** 3 - 0.95 * b ** 2 + 1.82 * b + 1.43
    thr = w * S[:, :m].median(dim=1, keepdim=True).values
    S = torch.where(S > thr, S, torch.zeros_like(S))
    return (U * S.unsqueeze(1)) @ Vh


class Block(nn.Module):
    def __init__(self, cin, cout, dilation):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, 5, padding=2 * dilation, dilation=dilation, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.skip = nn.Identity() if cin == cout else nn.Conv1d(cin, cout, 1, bias=False)

    def forward(self, x):
        return F.gelu(self.bn(self.pw(self.dw(x)))) + self.skip(x)


class Encoder(nn.Module):
    def __init__(self, in_ch: int = len(DATA_BINS), emb: int = 128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, 64, 1, bias=False), nn.BatchNorm1d(64), nn.GELU())
        self.blocks = nn.Sequential(Block(64, 96, 1), Block(96, 128, 2), Block(128, 128, 4), Block(128, 128, 8))
        self.head = nn.Linear(256, emb)

    def forward(self, x):                     # x: (B, C, T)
        h = self.blocks(self.stem(x))
        h = torch.cat([h.mean(-1), h.std(-1)], dim=1)
        return self.head(h)


class Projector(nn.Module):
    def __init__(self, emb=128, out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, out))

    def forward(self, z):
        return self.net(z)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """SimCLR NT-Xent loss; positives = the two views of the same capture."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


@torch.no_grad()
def embed(encoder: Encoder, W: np.ndarray, kind: str, scale: float, batch: int = 256) -> np.ndarray:
    encoder.eval()
    out = []
    for i in range(0, len(W), batch):
        x = prepare_input(torch.as_tensor(W[i:i + batch], dtype=torch.float32), kind, scale)
        out.append(encoder(x).numpy())
    return np.concatenate(out)
