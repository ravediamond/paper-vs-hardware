#!/usr/bin/env python3
"""WS-057 gate A2 (projects/horizon-depth/PROTOCOL.md): SmolVLA chunk latency vs Euler steps N in vla.cpp, on the Orin.

For N in 1,2,3,5,10: start vla-server with `--num-steps N` (same flags as the 238 ms reference: --flash-attn 1
--mm-prec default), then the real client (VlaCppClient, in the robot-day container) makes 5 warm-up and 50 timed chunk
calls on the 20 A1 frames, cycled. Records client wall-clock, the server's total / vision / inference fields and peak
RAM (tegrastats). Run only at MAXN_SUPER + jetson_clocks with nothing else on the GPU.

    python3 scripts/hd_latency.py frames      # (in the container) save the 20 A1 frames to outputs/hd/frames/
    python3 scripts/hd_latency.py run         # (host) the timing, -> projects/horizon-depth/results/a2_latency.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
FRAMES = ROOT / "outputs/hd/frames"
OUT = ROOT / "projects/horizon-depth/results/a2_latency.json"
G13B = ROOT / "shared/gate/data/2026-10-19_g13b_offline/g13b_summary.json"
STEPS, WARM, CALLS, PORT, REF_MS = (1, 2, 3, 5, 10), 5, 50, 5561, 238.0
KEYS = ("observation.images.front_top_camera", "observation.images.wrist_camera")


def save_frames(a):
    from so101_harness.episodes import EpisodeDataset, download

    ds = EpisodeDataset(download("ravediamond/so101_bowls_all", with_videos=True))
    FRAMES.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, ep in enumerate(json.loads(G13B.read_text())["meta"]["episodes"]):
        t = int(0.4 * len(ds.states(ep)))
        for k in KEYS:
            np.save(FRAMES / f"{i:02d}_{k.split('.')[-1]}.npy", ds.frame(ep, t, k))
        np.save(FRAMES / f"{i:02d}_state.npy", ds.states(ep)[t].astype(np.float32))
        meta.append({"i": i, "episode": ep, "frame": t, "task": ds.task(ep)})
    (FRAMES / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {len(meta)} frames to {FRAMES}")


def worker(a):
    """In the container: warm-ups, then CALLS timed calls; one JSON line per call."""
    from vla_cpp import VlaCppClient

    meta = json.loads((FRAMES / "meta.json").read_text())
    client = VlaCppClient(a.server, tokenizer=a.tokenizer, image_keys=KEYS, action_dim=6, n_action_steps=50)
    chw = lambda im: np.ascontiguousarray(im.transpose(2, 0, 1), dtype=np.float32) / 255.0  # noqa: E731

    def obs(i):
        m = meta[i % len(meta)]
        o = {k: chw(np.load(FRAMES / f"{m['i']:02d}_{k.split('.')[-1]}.npy")) for k in KEYS}
        return o | {"observation.state": np.load(FRAMES / f"{m['i']:02d}_state.npy"), "task": m["task"]}
    for i in range(WARM):
        client.predict_chunk(obs(i))
    for i in range(CALLS):
        o = obs(i); t0 = time.perf_counter(); client.predict_chunk(o); rtt = (time.perf_counter() - t0) * 1e3
        r = client.last_response
        print(json.dumps({"rtt": rtt, "total": r.latency_ms_total, "vision": r.latency_ms_vision,
                          "inference": r.latency_ms_inference}), flush=True)


def gpu_free():
    """Protocol: nothing else on the GPU. The only GPU users here are robot-day containers and vla-server."""
    others = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True).stdout.split()
    srv = subprocess.run(["pgrep", "-f", "vla-server"], capture_output=True, text=True).stdout.split()
    return not others and not srv


def run(a):
    import bench_glat_seg as B  # VLA_SERVER, GGUF, TOKENIZER (the 238 ms reference setup)

    mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout
    if "MAXN" not in mode:
        sys.exit(f"not at MAXN_SUPER (nvpmodel -q: {mode.strip()!r}); timing would not match the 238 ms reference")
    vla_commit = subprocess.run(["git", "-C", str(B.VLA_SERVER.parents[1]), "log", "-1", "--format=%h"],
                                capture_output=True, text=True).stdout.strip()
    res = {"vla_cpp_commit": vla_commit, "power": mode.strip(), "by_n": {}}
    for n in STEPS:
        if not gpu_free():
            sys.exit("another container or vla-server is running: the GPU is not free (protocol)")
        log = open(ROOT / f"outputs/hd/vla-server_n{n}.log", "w")
        srv = subprocess.Popen([str(B.VLA_SERVER), "--bind", f"tcp://127.0.0.1:{PORT}", "--flash-attn", "1",
                                "--mm-prec", "default", "--num-steps", str(n), str(B.GGUF)], stdout=log, stderr=subprocess.STDOUT)
        time.sleep(20)
        tg = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE, text=True)
        try:
            out = subprocess.run([str(ROOT / "shared/gate/jetson_run.sh"), "--python", "/work/scripts/hd_latency.py",
                                  "worker", "--server", f"tcp://127.0.0.1:{PORT}"], capture_output=True, text=True,
                                 timeout=1800).stdout
        finally:
            tg.terminate(); tg_out = tg.communicate()[0]; srv.terminate(); srv.wait()
        rows = [json.loads(x) for x in out.splitlines() if x.startswith("{")]
        ram = [int(m.group(1)) for m in re.finditer(r"RAM (\d+)/", tg_out)]
        med = {k: statistics.median(r[k] for r in rows) for k in ("rtt", "total", "vision", "inference")}
        res["by_n"][str(n)] = {"calls": len(rows), "median": med, "peak_ram_mb": max(ram) if ram else None}
        print(n, med, flush=True)
    b = res["by_n"]
    step = (b["10"]["median"]["inference"] - b["1"]["median"]["inference"]) / 9
    res["split"] = {"per_step_denoise_ms": step, "prefill_ms": b["1"]["median"]["total"] - b["1"]["median"]["vision"] - step,
                    "vision_ms": b["10"]["median"]["vision"]}
    r10, r2 = b["10"]["median"]["rtt"], b["2"]["median"]["rtt"]
    res["n10_vs_238_pct"] = 100 * (r10 / REF_MS - 1)
    res["n2_over_n10"] = r2 / r10
    if abs(res["n10_vs_238_pct"]) > 5:
        res["verdict_A2"] = "void (N = 10 not within ±5 % of 238 ms)"
    elif res["n2_over_n10"] <= 0.85:
        res["verdict_A2"] = "pass"
    elif res["n2_over_n10"] > 0.92:
        res["verdict_A2"] = "fail"
    else:
        res["verdict_A2"] = "not proven"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n10_vs_238_pct", "n2_over_n10", "split", "verdict_A2")}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("frames")
    w = sub.add_parser("worker"); w.add_argument("--server", required=True)
    w.add_argument("--tokenizer", default="/root/.cache/huggingface/hub/models--ravediamond--smolvla_2026-09-24_14-34-40/"
                   "snapshots/c1d76f627a559f488d9b043d20036171b3cca604/tokenizer")
    sub.add_parser("run")
    a = ap.parse_args()
    {"frames": save_frames, "worker": worker, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
