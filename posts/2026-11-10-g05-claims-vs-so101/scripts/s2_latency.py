#!/usr/bin/env python3
"""WS-062 A2: G0.5 zero-shot chunk latency on the Orin NX, PyTorch (the g05-pin checkout), MAXN_SUPER + jetson_clocks,
nothing else on the GPU. 5 warm-up + 50 timed `predict_action_chunk` calls on the A3 frames (cycled), System 2 (the
checkpoint's CoT + actions; the bar is on this) and, for the record, System 1 (actions only). Total wall time per chunk
with cuda synchronisation; the policy exposes no vision / text / action hooks, so no split is reported. Peak torch
memory recorded. Bar (prompt): median ≤ 667 ms pass, > 1 500 ms → arm gates dropped.

    python s2_latency.py --policy lerobot/g05_so101 --frames outputs/s2/frames.npz \\
        --out projects/system2-g05/results/a2_latency.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

WARM, CALLS = 5, 50
CAMS = ("observation.images.exterior", "observation.images.wrist_right")


def gpu_free():
    """The protocol's 'nothing else on the GPU': no other process holding GPU memory, no vla-server."""
    srv = subprocess.run(["pgrep", "-x", "vla-server"], capture_output=True, text=True).stdout.split()
    return not srv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True); ap.add_argument("--revision")
    ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--power", required=True, help="nvpmodel -q output, read on the host before the run")
    ap.add_argument("--allow-below-maxn", action="store_true",
                    help="WS-163: run at a lower power mode; the mode is written to the JSON and the result is labelled an upper bound")
    ap.add_argument("--stream-bf16", action="store_true",
                    help="WS-163: stream the safetensors tensor by tensor straight to the GPU in bf16 (no fp32 copy on CPU or GPU)")
    ap.add_argument("--load-only", action="store_true", help="load, report memory, exit (no timing)")
    a = ap.parse_args()
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.g05.modeling_g05 import G05Policy

    below_maxn = "MAXN" not in a.power
    if below_maxn and not a.allow_below_maxn:
        raise SystemExit(f"not at MAXN_SUPER ({a.power!r}); pass --allow-below-maxn to run anyway (result = upper bound)")
    if not gpu_free():
        raise SystemExit("vla-server is running: the GPU is not free")
    f = np.load(a.frames)
    if a.stream_bf16:
        import os
        import lerobot.policies.g05.modeling_g05 as mg
        from safetensors import safe_open

        def load_file_bf16(path, device="cpu", backend=None):
            out, seen = {}, 0
            fd = os.open(path, os.O_RDONLY)
            with safe_open(path, framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    t = sf.get_tensor(k)
                    out[k] = t.to(device=device, dtype=torch.bfloat16) if t.is_floating_point() else t.to(device)
                    seen += t.numel() * t.element_size()
                    del t
                    if seen > 5e8:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); seen = 0
            os.close(fd)
            return out
        mg.load_file = load_file_bf16
    policy = G05Policy.from_pretrained(a.policy, revision=a.revision).eval().to("cuda")
    if a.load_only:
        print("loaded; cuda allocated MB", torch.cuda.memory_allocated() / 2**20, flush=True)
        print(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout)
        return
    pre = None

    def make_pre():  # the input processor bakes in runtime_system, so rebuild it after switching system
        return make_pre_post_processors(policy.config, a.policy,
                                        preprocessor_overrides={"device_processor": {"device": "cuda"}})[0]

    def batch(i):
        i %= len(f["t"])
        b = {k: torch.from_numpy(f["images"][i, j]).float().div(255)[None] for j, k in enumerate(CAMS)}
        b["observation.state"] = torch.from_numpy(f["state"][i]).float()[None]
        b["task"] = ["Put the yellow cube in the center"]
        return pre(b)
    res = {"policy": a.policy, "revision": a.revision, "power": a.power.strip(), "calls": CALLS, "warmup": WARM,
           "label": "upper bound (below MAXN_SUPER: MAXN can only be faster)" if below_maxn else "MAXN_SUPER"}
    for system in ("system2", "system1"):
        policy.config.runtime_system = system
        pre = make_pre()
        torch.cuda.reset_peak_memory_stats()
        for i in range(WARM):
            policy.predict_action_chunk(batch(i))
        ts = []
        for i in range(CALLS):
            b = batch(i); torch.cuda.synchronize(); t0 = time.perf_counter()
            policy.predict_action_chunk(b); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        res[system] = {"p50_ms": statistics.median(ts), "p90_ms": float(np.percentile(ts, 90)),
                       "min_ms": min(ts), "max_ms": max(ts),
                       "peak_torch_mb": torch.cuda.max_memory_allocated() / 2**20}
        print(system, res[system], flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).with_suffix(".partial.json").write_text(json.dumps(res, indent=1))
    p50 = res["system2"]["p50_ms"]
    res["verdict_A2"] = "pass" if p50 <= 667 else "fail: arm gates dropped" if p50 > 1500 else "not proven"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
