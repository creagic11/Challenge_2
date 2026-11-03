"""Task 3: SAPT — Shared-Anchored Prototype Transfer for cross-room few-label adaptation.

Setting (target room T, source room S):
  * labelled source development windows (all source classes)
  * K labelled support captures per target-exclusive class (target development)
  * optional unlabelled target development pool (labels hidden)  -> "UDA" tier
  * validation windows: never used for fitting

Pipeline in a frozen feature space z:
  1. global standardisation (fitted on all development windows, unlabelled)
  2. room-wise alignment  z~ = (z - mu_r) / sd_r
       'none'   : no alignment
       'room'   : mu_r, sd_r over all development windows of room r (unlabelled)
       'shared' : SAPT. mu_S, sd_S over the *shared-class* source windows (labels known);
                  mu_T, sd_T over target-pool windows weighted by their predicted
                  probability of belonging to a shared class (iterated). This removes the
                  bias of plain per-room centring when the two rooms contain different
                  activity mixes (label shift).
  3. prototypes (cosine): shared classes from the source, exclusive classes from supports
  4. optional anchored soft k-means refinement on the target pool
  5. inference: route -> room statistics + room label space -> nearest prototype
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHARED = ["sit", "stand", "walk"]
EXCLUSIVE = {"A": ["sitdown", "standup", "fall"], "B": ["empty", "lying"]}
CLASSES = {"A": ["fall", "sit", "sitdown", "stand", "standup", "walk"],
           "B": ["empty", "lying", "sit", "stand", "walk"]}


def _norm(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def _wstats(Z, w=None):
    if w is None:
        return Z.mean(0), Z.std(0) + 1e-6
    w = w / max(w.sum(), 1e-9)
    mu = (w[:, None] * Z).sum(0)
    sd = np.sqrt((w[:, None] * (Z - mu) ** 2).sum(0)) + 1e-6
    return mu, sd


def sinkhorn(S: np.ndarray, tau: float, n_iter: int = 100, col_marg=None) -> np.ndarray:
    """Entropic optimal transport of N pool samples onto C classes.
    S: similarities (N, C). Row marginals uniform (each sample has mass 1/N); column
    marginals uniform by default (every class of the target room is present in the pool).
    Returns a row-normalised assignment matrix."""
    N, C = S.shape
    K = np.exp((S - S.max()) / tau)
    r = np.full(N, 1.0 / N)
    c = np.full(C, 1.0 / C) if col_marg is None else col_marg
    u = np.ones(N)
    for _ in range(n_iter):
        v = c / (K.T @ u + 1e-30)
        u = r / (K @ v + 1e-30)
    Pm = u[:, None] * K * v[None, :]
    return Pm / Pm.sum(1, keepdims=True)


def _softmax(S):
    S = S - S.max(1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(1, keepdims=True)


@dataclass
class SAPTConfig:
    align: str = "shared"        # none | room | shared
    use_pool: bool = True        # UDA tier (unlabelled target development pool)
    refine: bool = True          # anchored soft k-means on the pool
    tau: float = 0.1             # cosine temperature
    anchor: float = 20.0         # anchor weight (in pseudo-sample units)
    n_align_iter: int = 5
    n_refine_iter: int = 10
    routing: str = "predicted"   # predicted | oracle | none
    head: str = "proto"          # proto | lr | lr_st | ot_lr
    balanced: bool = False       # balanced (Sinkhorn OT) instead of softmax pool refinement
    st_conf: float = 0.8         # self-training confidence threshold


class SAPT:
    def __init__(self, cfg: SAPTConfig):
        self.cfg = cfg

    # ---------------------------------------------------------------- fit
    def fit(self, Z_dev_all, Zs, ys, src, Zsup, ysup, tgt, Zpool=None, room_all=None):
        """Z_dev_all: all development features (unlabelled) for global standardisation.
        Zs/ys: source dev features/labels; Zsup/ysup: support features/labels (target);
        Zpool: unlabelled target development features (UDA tier only)."""
        c = self.cfg
        self.src, self.tgt = src, tgt
        self.g_mu, self.g_sd = _wstats(Z_dev_all)
        g = lambda Z: (Z - self.g_mu) / self.g_sd
        Zs, Zsup = g(Zs), g(Zsup)
        Zp = g(Zpool) if (Zpool is not None and c.use_pool) else None
        self.stats = {}
        if c.routing == "none" or c.align == "none":
            # single pooled domain (no room-specific statistics)
            mu, sd = np.zeros(Zs.shape[1]), np.ones(Zs.shape[1])
            self.stats = {src: (mu, sd), tgt: (mu, sd), "pooled": (mu, sd)}
        elif c.align == "room":
            self.stats[src] = _wstats(Zs)
            ref = Zp if Zp is not None else Zsup
            self.stats[tgt] = _wstats(ref)
        elif c.align == "shared":
            msk = np.isin(ys, SHARED)
            self.stats[src] = _wstats(Zs[msk])
            if Zp is None:
                self.stats[tgt] = _wstats(Zsup)          # strict tier: supports only
            else:
                self.stats[tgt] = _wstats(Zp)            # init: plain room statistics
        else:
            raise ValueError(c.align)

        # label space and anchors
        if c.routing == "none":
            classes = sorted(set(ys) | set(ysup))      # union over both rooms
        else:
            classes = CLASSES[tgt]
        self.classes = classes
        al = lambda Z, r: (Z - self.stats[r][0]) / self.stats[r][1]
        As = _norm(al(Zs, src))
        Asup = _norm(al(Zsup, tgt))

        def anchors():
            P = []
            for k in classes:
                if k in EXCLUSIVE[tgt]:
                    P.append(Asup[ysup == k].mean(0))
                else:
                    P.append(As[ys == k].mean(0))
            return _norm(np.stack(P))

        self.protos = anchors()
        shared_idx = [i for i, k in enumerate(classes) if k in SHARED]
        if Zp is not None and c.align == "shared" and c.routing != "none":
            for _ in range(c.n_align_iter):
                Pp = _softmax(_norm(al(Zp, tgt)) @ self.protos.T / c.tau)
                self.stats[tgt] = _wstats(Zp, Pp[:, shared_idx].sum(1))
                Asup = _norm(al(Zsup, tgt))
                self.protos = anchors()
        if Zp is not None and c.refine:
            Ap = _norm(al(Zp, tgt))
            A0 = self.protos.copy()
            for _ in range(c.n_refine_iter):
                Sim = Ap @ self.protos.T
                Pp = sinkhorn(Sim, c.tau) if c.balanced else _softmax(Sim / c.tau)
                self.protos = _norm(c.anchor * A0 + Pp.T @ Ap)
            self.pool_assign = sinkhorn(Ap @ self.protos.T, c.tau) if c.balanced else _softmax(Ap @ self.protos.T / c.tau)
        # optional discriminative head on the aligned space
        self.lr = None
        if c.head == "ot_lr" and Zp is not None:
            # within-target-room classifier trained on balanced-OT pseudo-labels of the
            # unlabelled pool, plus the true support labels (validation is never used)
            from sklearn.linear_model import LogisticRegression
            key_t = tgt if c.routing != "none" else "pooled"
            Ap = _norm(al(Zp, key_t))
            yp = np.array(classes)[self.pool_assign.argmax(1)]
            X = np.r_[Ap, np.repeat(_norm(al(Zsup, key_t)), 5, axis=0)]
            y = np.r_[yp, np.repeat(ysup, 5)]
            self.lr = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(X, y)
        elif c.head in ("lr", "lr_st"):
            from sklearn.linear_model import LogisticRegression
            ms = np.isin(ys, [k for k in classes if k not in EXCLUSIVE[tgt]])
            key_t = tgt if c.routing != "none" else "pooled"
            key_s = src if c.routing != "none" else "pooled"
            X = np.r_[_norm(al(Zs[ms], key_s)), _norm(al(Zsup, key_t))]
            y = np.r_[ys[ms], ysup]
            self.lr = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(X, y)
            if c.head == "lr_st" and Zp is not None:
                Ap = _norm(al(Zp, key_t))
                for _ in range(2):
                    Pp = self.lr.predict_proba(Ap)
                    keep = Pp.max(1) >= c.st_conf
                    Xa = np.r_[X, Ap[keep]]
                    ya = np.r_[y, self.lr.classes_[Pp[keep].argmax(1)]]
                    self.lr = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(Xa, ya)
        # the source room's own model (used when a window is routed to the source room)
        src_classes = CLASSES[src]
        self.src_classes = src_classes
        self.src_protos = _norm(np.stack([As[ys == k].mean(0) for k in src_classes]))
        return self

    def _al(self, Z, key):
        return (Z - self.stats[key][0]) / self.stats[key][1]

    def _g(self, Z):
        return (Z - self.g_mu) / self.g_sd

    # ------------------------------------------------------------ predict
    def predict(self, Z, routed_rooms):
        """Z: features of query windows; routed_rooms: predicted (or oracle) room per window."""
        Z = self._g(Z)
        labels = np.empty(len(Z), dtype=object)
        conf = np.zeros(len(Z))
        for r in np.unique(routed_rooms):
            m = routed_rooms == r
            if self.cfg.routing == "none" or r == self.tgt:
                key = self.tgt if self.cfg.routing != "none" else "pooled"
                A = _norm(self._al(Z[m], key))
                if self.lr is not None:
                    P = self.lr.predict_proba(A)
                    labels[m] = self.lr.classes_[P.argmax(1)]
                else:
                    P = _softmax(A @ self.protos.T / self.cfg.tau)
                    labels[m] = np.array(self.classes)[P.argmax(1)]
            else:
                A = _norm(self._al(Z[m], self.src))
                P = _softmax(A @ self.src_protos.T / self.cfg.tau)
                labels[m] = np.array(self.src_classes)[P.argmax(1)]
            conf[m] = P.max(1)
        return labels, conf


def sample_supports(dev_capture_labels: dict, exclusive: list, K_list: list, seed: int) -> dict:
    """Nested support sets: K distinct development captures per exclusive class.
    dev_capture_labels: capture_id -> label (target development captures only)."""
    rng = np.random.default_rng(seed)
    out = {K: [] for K in K_list}
    for cls in exclusive:
        ids = sorted(c for c, l in dev_capture_labels.items() if l == cls)
        order = list(rng.permutation(ids))
        for K in K_list:
            out[K] += order[:K]
    return out
