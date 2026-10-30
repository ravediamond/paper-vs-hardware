#!/usr/bin/env python3
"""WS-057 gate A4 (projects/horizon-depth/PROTOCOL.md): thermal reading for N = 10 vs N = 2 on the Orin.

10 min continuous chunk loop (one call every 1.67 s, the s = 50 cadence) at N = 10, 10 min idle, then 10 min at N = 2;
tegrastats at 1 Hz (GPU clock, GPU temperature, total power). Run once per power mode (the mode is read, not set:
switching modes is the owner's sudo). Throttling = GPU clock below 95 % of its first-minute median for >= 10
consecutive seconds. Reading: no throttling at N = 10 -> "not applicable here"; throttling at 10 and none at 2 ->
"replicates"; throttling in both -> "does not replicate".

    python3 scripts/hd_thermal.py run      # -> projects/horizon-depth/results/a4_thermal_<mode>.csv + a4_thermal.json
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "projects/horizon-depth/results"
PORT, PHASE_S, IDLE_S, CADENCE_S = 5562, 600, 600, 1.67


GPU_FREQ = Path("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq")


def parse(line):
    """GPU clock (MHz), GPU temperature (C), total power (mW) from one tegrastats line. This JetPack's tegrastats
    prints `GR3D_FREQ n%` without the clock, so the clock is read from devfreq at the same second when it is absent."""
    clk = re.search(r"GR3D_FREQ \d+%@\[?(\d+)", line) or re.search(r"GR3D_FREQ \d+%@(\d+)", line)
    if not clk and GPU_FREQ.exists():
        return (int(GPU_FREQ.read_text()) // 1_000_000, *parse_rest(line))
    return (int(clk.group(1)) if clk else None, *parse_rest(line))


def parse_rest(line):
    tmp = re.search(r"gpu@([\d.]+)C", line, re.I)
    pw = re.search(r"VDD_IN (\d+)mW", line)
    return (float(tmp.group(1)) if tmp else None, int(pw.group(1)) if pw else None)


def throttled(clock):
    c = np.array([x for x in clock if x is not None], float)
    if len(c) < 70:
        return None
    ref = np.median(c[:60]); low = c < 0.95 * ref; run = best = 0
    for v in low:
        run = run + 1 if v else 0; best = max(best, run)
    return bool(best >= 10)


def phase(n, mode, rows):
    import bench_glat_seg as B

    srv = subprocess.Popen([str(B.VLA_SERVER), "--bind", f"tcp://127.0.0.1:{PORT}", "--flash-attn", "1", "--mm-prec",
                            "default", "--num-steps", str(n), str(B.GGUF)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(20)
    calls = subprocess.Popen([str(ROOT / "shared/gate/jetson_run.sh"), "--python", "/work/scripts/hd_thermal.py", "loop",
                              "--server", f"tcp://127.0.0.1:{PORT}", "--seconds", str(PHASE_S)], stdout=subprocess.DEVNULL)
    tg = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=subprocess.PIPE, text=True)
    t0 = time.time(); clocks = []
    for line in tg.stdout:
        c, t, p = parse(line)
        rows.append({"mode": mode, "n": n, "t_s": round(time.time() - t0, 1), "gpu_mhz": c, "gpu_c": t, "power_mw": p})
        clocks.append(c)
        if calls.poll() is not None:
            break
    tg.terminate(); srv.terminate(); srv.wait()
    return throttled(clocks)


def loop(server, seconds):
    """In the container: one chunk call every CADENCE_S for `seconds` (frames from hd_latency)."""
    sys.argv = [sys.argv[0]]
    import hd_latency as L
    from vla_cpp import VlaCppClient

    meta = json.loads((L.FRAMES / "meta.json").read_text())
    tok = ("/root/.cache/huggingface/hub/models--ravediamond--smolvla_2026-09-24_14-34-40/snapshots/"
           "c1d76f627a559f488d9b043d20036171b3cca604/tokenizer")
    client = VlaCppClient(server, tokenizer=tok, image_keys=L.KEYS, action_dim=6, n_action_steps=50)
    chw = lambda im: np.ascontiguousarray(im.transpose(2, 0, 1), dtype=np.float32) / 255.0  # noqa: E731
    end, i = time.time() + seconds, 0
    while time.time() < end:
        m = meta[i % len(meta)]; i += 1
        o = {k: chw(np.load(L.FRAMES / f"{m['i']:02d}_{k.split('.')[-1]}.npy")) for k in L.KEYS}
        t0 = time.time()
        client.predict_chunk(o | {"observation.state": np.load(L.FRAMES / f"{m['i']:02d}_state.npy"), "task": m["task"]})
        time.sleep(max(0.0, CADENCE_S - (time.time() - t0)))


def run():
    mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout.strip().splitlines()[0]
    tag = "maxn" if "MAXN" in mode else "25w" if "25" in mode else re.sub(r"\W+", "_", mode).lower()
    rows = []
    t10 = phase(10, tag, rows)
    time.sleep(IDLE_S)
    t2 = phase(2, tag, rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"a4_thermal_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    reading = ("not applicable here" if t10 is False else "replicates" if (t10 and t2 is False)
               else "does not replicate" if (t10 and t2) else "undetermined")
    j = OUT / "a4_thermal.json"
    res = json.loads(j.read_text()) if j.exists() else {}
    res[tag] = {"power_mode": mode, "throttled_n10": t10, "throttled_n2": t2, "reading": reading}
    j.write_text(json.dumps(res, indent=1)); print(json.dumps(res[tag], indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["loop"]:
        loop(sys.argv[sys.argv.index("--server") + 1], float(sys.argv[sys.argv.index("--seconds") + 1]))
    else:
        run()
