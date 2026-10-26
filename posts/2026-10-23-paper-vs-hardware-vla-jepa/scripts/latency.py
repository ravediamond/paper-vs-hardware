"""Step D: per-decision latency on the Jetson for N = 1, 4, 8 sampled chunks, deployed precision and image size.

One decision = preprocess (rename, batch, to GPU, normalize state) + backbone once (image resize + Qwen3-VL) + head
batched over N (4 Euler steps) + postprocess (clip, unnormalize). Each part is timed with CUDA syncs, 5 warm-up then
50 timed runs per N, on a real dataset observation. The deployed single-chunk path (`predict_action_chunk`) is timed
the same way as a reference. Medians and p95 in ms.

    IMAGE=lerobot-jetson:rc LEROBOT_SRC=image shared/gate/jetson_run.sh --python projects/vla-jepa-world-model/prechecks/latency.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

WARMUP, RUNS = 5, 50
NS = [1, 4, 8]
OUT = C.OUT / "latency"


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1e3


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    C.seed_everything()
    OUT.mkdir(parents=True, exist_ok=True)
    policy, pre, post, cfg, path = C.load_policy()
    item = LeRobotDataset(C.DATASET_REPO)[5000]
    obs = {k: v for k, v in item.items() if k.startswith("observation.")} | {"task": item["task"]}

    results = {}
    for n in NS:
        parts = {"preprocess": [], "backbone": [], "head": [], "postprocess": [], "total": []}
        for r in range(WARMUP + RUNS):
            t_start = time.perf_counter()
            batch, t_pre = timed(lambda: pre(dict(obs)))
            (tokens, state), t_bb = timed(lambda: C.encode(policy, batch))
            noise = C.draw_noise(policy, n, seed=r)
            chunks, t_head = timed(lambda: C.sample_head(policy, tokens, state, noise))
            _, t_post = timed(lambda: post(chunks.reshape(-1, chunks.shape[-1])))
            total = (time.perf_counter() - t_start) * 1e3
            if r >= WARMUP:
                for k, v in zip(parts, (t_pre, t_bb, t_head, t_post, total)):
                    parts[k].append(v)
        results[f"N={n}"] = {k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
                             for k, v in parts.items()}
        print(f"N={n}: " + ", ".join(f"{k} {np.median(v):.1f}" for k, v in parts.items()), flush=True)

    ref = []
    for r in range(WARMUP + RUNS):
        batch = pre(dict(obs))
        _, t = timed(lambda: policy.predict_action_chunk(batch))
        if r >= WARMUP:
            ref.append(t)
    results["deployed predict_action_chunk (N=1, no preprocess)"] = {
        "median": float(np.median(ref)), "p95": float(np.percentile(ref, 95))}
    results["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

    t1 = results["N=1"]["total"]["median"]
    md = ["# Step D: latency on the Jetson Orin NX 16 GB (MAXN_SUPER, bf16 backbone, 224x224, 4 Euler steps)", "",
          f"{RUNS} timed runs after {WARMUP} warm-up, one real observation; ms, median (p95).", "",
          "| N | preprocess | backbone | head (batched N) | postprocess | total | total / N=1 |", "|---|---|---|---|---|---|---|"]
    for n in NS:
        r = results[f"N={n}"]
        cell = lambda k: f"{r[k]['median']:.1f} ({r[k]['p95']:.1f})"  # noqa: E731
        md.append(f"| {n} | {cell('preprocess')} | {cell('backbone')} | {cell('head')} | {cell('postprocess')} | "
                  f"{cell('total')} | {r['total']['median'] / t1:.2f}x |")
    ref = results["deployed predict_action_chunk (N=1, no preprocess)"]
    md += ["", f"Reference: deployed `predict_action_chunk` (backbone + head, N = 1) {ref['median']:.1f} "
               f"({ref['p95']:.1f}) ms. Peak torch allocation {results['peak_allocated_gb']:.2f} GB.", ""]
    (OUT / "latency.json").write_text(json.dumps(results, indent=1))
    (OUT / "summary.md").write_text("\n".join(md))
    C.fix_owner(OUT)
    print("\n".join(md))


if __name__ == "__main__":
    main()
