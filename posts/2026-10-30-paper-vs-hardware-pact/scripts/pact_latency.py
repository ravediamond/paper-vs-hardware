#!/usr/bin/env python3
"""WS-059 gate A (projects/pact-vla/PROTOCOL.md): the cost of PACT's extra pass on the Orin, nothing else on the GPU.

vla.cpp has no batched sampling path and no velocity query on a given x_τ, so:
- T_gen(1) and the server split (vision, prefill, per-step denoise) come from WS-057 A2
  (projects/horizon-depth/results/a2_latency.json, same server flags as the 238 ms reference, N = 10);
- T_gen(4) = 4 sequential vla.cpp calls (4 × T_gen(1)), labelled "sequential";
- T_prefix = vision + prefill;
- T_ver is projected: ratio r = T(16 velocity rows on one prefix) / T(one denoise step, batch 1), measured in PyTorch
  on the Orin (`ratio`), times the vla.cpp per-step denoise time. Labelled "projected".

    IMAGE=lerobot-jetson:rc LEROBOT_SRC=image shared/gate/jetson_run.sh --python scripts/pact_latency.py ratio \\
        --policy-path ravediamond/smolvla_2026-09-24_14-34-40 --revision c1d76f6
    python3 scripts/pact_latency.py combine      # -> projects/pact-vla/results/gate_a.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "projects/pact-vla/results"
A2 = ROOT / "projects/horizon-depth/results/a2_latency.json"
TICK = 1000 / 30


def ratio(a):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
    import pact_flowres as PF
    from huggingface_hub import snapshot_download
    from so101_harness.episodes import EpisodeDataset, download

    m = PF.Pact(snapshot_download(a.policy_path, revision=a.revision), "cuda")
    ds = EpisodeDataset(download("ravediamond/so101_bowls_all", with_videos=True))
    ep = json.loads(PF.G13B.read_text())["meta"]["episodes"][0]; t = 50
    obs = {"state": ds.states(ep)[t], "task": ds.task(ep), **{k: ds.frame(ep, t, k) for k in m.image_keys}}
    pre = m.prefix(obs)
    x1 = torch.randn(1, m.horizon, m.action_dim)
    x16 = torch.randn(16, m.horizon, m.action_dim)

    def timed(fn, n=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts)
    step = timed(lambda: m.velocity(pre, x1, 0.5))
    rows16 = timed(lambda: [m.velocity(pre, x16[i:i + 1], 0.5) for i in range(16)])
    res = {"pytorch_step_ms": step, "pytorch_16_rows_ms": rows16, "ratio": rows16 / step,
           "note": "16 rows run one by one on one cached prefix, as the gate-B scorer does"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ver_ratio.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def combine(a):
    a2 = json.loads(A2.read_text()); r = json.loads((OUT / "ver_ratio.json").read_text())
    n10 = a2["by_n"]["10"]["median"]; sp = a2["split"]
    t1 = n10["rtt"]; t4 = 4 * t1
    t_prefix = sp["vision_ms"] + sp["prefill_ms"]
    t_ver = r["ratio"] * sp["per_step_denoise_ms"]
    t_pact = t4 + t_prefix + t_ver
    opt = max(t4, TICK + t_prefix) + t_ver
    d_naive, d_pact = math.ceil(t1 / TICK), math.ceil(t_pact / TICK)
    verdict = "pass" if d_pact <= d_naive else "fail" if d_pact >= d_naive + 2 else "not proven"
    res = {"T_gen_1_ms": t1, "T_gen_4_ms": t4, "T_gen_4_label": "sequential (vla.cpp has no batched sampling)",
           "T_prefix_ms": t_prefix, "T_ver_ms": t_ver, "T_ver_label": "projected (PyTorch ratio x vla.cpp step)",
           "T_PACT_ms": t_pact, "optimistic_overlap_ms": opt, "d_naive": d_naive, "d_PACT": d_pact,
           "d_PACT_optimistic": math.ceil(opt / TICK), "verdict_A": verdict, "sources": [str(A2), "ver_ratio.json"]}
    (OUT / "gate_a.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ratio"); r.add_argument("--policy-path", required=True); r.add_argument("--revision")
    sub.add_parser("combine")
    a = ap.parse_args()
    {"ratio": ratio, "combine": combine}[a.cmd](a)


if __name__ == "__main__":
    main()
