#!/usr/bin/env python3
"""WS-049 gate 1 (projects/aac-chunking/PROTOCOL.md): wall time per AAC decision on the Orin, vla.cpp SmolVLA GGUF.

A decision samples N chunks for one observation. vla.cpp has no batched sampling path, so the N chunks are N
sequential client calls on the same observation (the protocol's upper bound; A4 is judged on it). For N in
1, 5, 10, 20, 40: 2 warm-up decisions, then 10 timed decisions on the 20 WS-057 frames, cycled. The server runs once
with the flags of the 238 ms reference (--flash-attn 1 --mm-prec default, the checkpoint's 10 Euler steps).
A4 bar: N = 20 median decision ≤ 298 ms → pass, otherwise fail. Peak RAM from tegrastats.
Run only at MAXN_SUPER + jetson_clocks with nothing else on the GPU.

    python3 scripts/hd_latency.py frames   # (in the container, once) the 20 frames, shared with WS-057 A2
    python3 scripts/aac_latency.py run     # (host) -> projects/aac-chunking/results/gate1_latency.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "projects/aac-chunking/results/gate1_latency.json"
NS, WARM, DECISIONS, PORT, REF_MS, BAR_MS = (1, 5, 10, 20, 40), 2, 10, 5562, 238.0, 298.0


def worker(a):
    """In the container: one JSON line per timed decision."""
    import numpy as np
    from vla_cpp import VlaCppClient

    import hd_latency as H

    meta = json.loads((H.FRAMES / "meta.json").read_text())
    client = VlaCppClient(a.server, tokenizer=a.tokenizer, image_keys=H.KEYS, action_dim=6, n_action_steps=50)
    chw = lambda im: np.ascontiguousarray(im.transpose(2, 0, 1), dtype=np.float32) / 255.0  # noqa: E731

    def obs(i):
        m = meta[i % len(meta)]
        o = {k: chw(np.load(H.FRAMES / f"{m['i']:02d}_{k.split('.')[-1]}.npy")) for k in H.KEYS}
        return o | {"observation.state": np.load(H.FRAMES / f"{m['i']:02d}_state.npy"), "task": m["task"]}

    def decision(o, n):
        t0 = time.perf_counter()
        for _ in range(n):
            client.predict_chunk(o)
        return (time.perf_counter() - t0) * 1e3
    for n in (NS if a.ns is None else [int(x) for x in a.ns.split(",")]):
        for i in range(WARM):
            decision(obs(i), n)
        for i in range(DECISIONS):
            print(json.dumps({"n": n, "ms": decision(obs(i), n)}), flush=True)


def run(a):
    import bench_glat_seg as B  # VLA_SERVER, GGUF (the 238 ms reference setup)
    import hd_latency as H

    mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout
    if "MAXN" not in mode:
        sys.exit(f"not at MAXN_SUPER (nvpmodel -q: {mode.strip()!r})")
    if not H.gpu_free():
        sys.exit("another container or vla-server is running: the GPU is not free (protocol)")
    vla_commit = subprocess.run(["git", "-C", str(B.VLA_SERVER.parents[1]), "log", "-1", "--format=%h"],
                                capture_output=True, text=True).stdout.strip()
    log = open(ROOT / "outputs/hd/vla-server_aac.log", "w")
    srv = subprocess.Popen([str(B.VLA_SERVER), "--bind", f"tcp://127.0.0.1:{PORT}", "--flash-attn", "1",
                            "--mm-prec", "default", str(B.GGUF)], stdout=log, stderr=subprocess.STDOUT)
    time.sleep(20)
    tg = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE, text=True)
    try:
        out = subprocess.run([str(ROOT / "shared/gate/jetson_run.sh"), "--python", "/work/scripts/aac_latency.py",
                              "worker", "--server", f"tcp://127.0.0.1:{PORT}"], capture_output=True, text=True,
                             timeout=3600).stdout
    finally:
        tg.terminate(); tg_out = tg.communicate()[0]; srv.terminate(); srv.wait()
    rows = [json.loads(x) for x in out.splitlines() if x.startswith("{")]
    ram = [int(m.group(1)) for m in re.finditer(r"RAM (\d+)/", tg_out)]
    by_n = {str(n): {"decisions": len(v), "median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)}
            for n in NS if (v := [r["ms"] for r in rows if r["n"] == n])}
    n20 = by_n["20"]["median_ms"]
    res = {"vla_cpp_commit": vla_commit, "power": mode.strip(), "path": "sequential (vla.cpp has no batched sampling)",
           "by_n": by_n, "peak_ram_mb": max(ram) if ram else None, "n1_vs_238_pct": 100 * (by_n["1"]["median_ms"] / REF_MS - 1),
           "A4_bar_ms": BAR_MS, "A4_n20_ms": n20, "verdict_A4": "pass" if n20 <= BAR_MS else "fail"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker"); w.add_argument("--server", required=True)
    w.add_argument("--ns", help="smoke test only: a subset of N, e.g. 1,5 (the run always uses all of NS)")
    w.add_argument("--tokenizer", default="/root/.cache/huggingface/hub/models--ravediamond--smolvla_2026-09-24_14-34-40/"
                   "snapshots/c1d76f627a559f488d9b043d20036171b3cca604/tokenizer")
    sub.add_parser("run")
    a = ap.parse_args()
    {"worker": worker, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
