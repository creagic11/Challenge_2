"""Ablation of the window rule: 'full' (official) vs 'consecutive' (150 consecutive valid rows
when available, else full-capture resampling), with the headline Task-3 pipeline (denoised handcrafted
features, room-wise alignment + LR head) and the Task-1 router. Validation is used for evaluation only.

    python scripts/12_window_rule_ablation.py
Writes results/ablation_window_rule.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from stride.data import Dataset  # noqa: E402
from stride.features import HandcraftedFeatures  # noqa: E402
from stride.fewshot import CLASSES, EXCLUSIVE, SAPT, SHARED, SAPTConfig, sample_supports  # noqa: E402
from stride.metrics import activity_metrics  # noqa: E402
from stride.router import RoomRouter  # noqa: E402
from stride.windowing import official_window  # noqa: E402


def main():
    ds = Dataset()
    man = ds.man
    y, room, cid = man.label.values, man.room.values, man.capture_id.values
    rows = []
    for mode in ["full", "consecutive"]:
        W = np.stack([official_window(*ds.captures[c], mode=mode)[0] for c in cid])
        dev, val = ds.idx(split="development"), ds.idx(split="validation")
        router = RoomRouter(0.1).fit(W[dev], room[dev])
        rows.append(dict(mode=mode, task="task1", target_room="both", K="NA", metric="accuracy",
                         value=accuracy_score(room[val], router.predict(W[val]))))
        Z = HandcraftedFeatures(denoise=True).fit(W[dev]).transform(W)
        routed = router.predict(W)
        for src, tgt in [("B", "A"), ("A", "B")]:
            s_idx = ds.idx(room=src, split="development")
            t_dev = ds.idx(room=tgt, split="development")
            q = ds.idx(room=tgt, split="validation")
            lab = {cid[i]: y[i] for i in t_dev}
            pos = {cid[i]: i for i in t_dev}
            for ep in range(10):
                sup = sample_supports(lab, EXCLUSIVE[tgt], [1, 5, 10, 20], seed=1000 * ep + (0 if tgt == "A" else 1))
                for K in [1, 5, 10, 20]:
                    si = np.array([pos[c] for c in sup[K]])
                    m = SAPT(SAPTConfig(align="room", use_pool=True, refine=True, routing="predicted", head="lr"))
                    m.fit(Z[dev], Z[s_idx], y[s_idx], src, Z[si], y[si], tgt, Zpool=Z[t_dev])
                    pl, _ = m.predict(Z[q], routed[q])
                    met = activity_metrics(y[q], pl, CLASSES[tgt], EXCLUSIVE[tgt], SHARED)
                    for k in ["macro_f1", "exclusive_macro_f1", "shared_macro_f1"]:
                        rows.append(dict(mode=mode, task="task3", target_room=tgt, K=K, metric=k,
                                         value=met[k], episode=ep))
        print("done", mode, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(ds.cfg["paths"]["results"]) / "ablation_window_rule.csv", index=False)
    t3 = df[df.task == "task3"].groupby(["mode", "target_room", "metric"]).value.mean().unstack("metric")
    print(df[df.task == "task1"][["mode", "value"]])
    print(t3.round(3))


if __name__ == "__main__":
    main()
