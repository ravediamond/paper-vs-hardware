#!/usr/bin/env python3
"""Summarise a `sudo tegrastats --interval 100 --logfile ...` log over time windows.

    tegra_summary.py <tegrastats.log> <phases.jsonl> [--kind profile] [--markdown]
    tegra_summary.py <tegrastats.log> --window <t0> <t1> [--label name]

tegrastats stamps whole seconds; lines that share a second are spread evenly across it.
Per window: GPU load (GR3D %) and clock, memory-controller load (EMC %), CPU load,
peak RAM, mean board power (VDD_IN) and GPU+CPU+CV rail, peak temperatures, and
energy per call = mean VDD_IN x mean latency, when the phase has a mean latency.
"""
import argparse, json, re, statistics as st, sys
from collections import defaultdict
from datetime import datetime

TS = re.compile(r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)")
RAM = re.compile(r"RAM (\d+)/(\d+)MB")
CPU = re.compile(r"CPU \[([^\]]*)\]")
EMC = re.compile(r"EMC_FREQ (\d+)%@(\d+)")
GR3D = re.compile(r"GR3D_FREQ (\d+)%@\[?(\d+)")
TEMP = re.compile(r"(\w+)@(-?[\d.]+)C")
RAIL = re.compile(r"(VDD_\w+) (\d+)mW/(\d+)mW")


def parse(path):
    by_sec = defaultdict(list)
    for line in open(path, errors="replace"):
        m = TS.match(line)
        if not m:
            continue
        sec = datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S").timestamp()
        s = {}
        if (r := RAM.search(line)):
            s["ram_mb"] = int(r.group(1))
        if (c := CPU.search(line)):
            loads = [int(x.split("%")[0]) for x in c.group(1).split(",") if "%" in x]
            s["cpu_pct"] = sum(loads) / len(loads) if loads else 0.0
        if (e := EMC.search(line)):
            s["emc_pct"], s["emc_mhz"] = int(e.group(1)), int(e.group(2))
        if (g := GR3D.search(line)):
            s["gpu_pct"], s["gpu_mhz"] = int(g.group(1)), int(g.group(2))
        for name, val in TEMP.findall(line):
            s["t_" + name] = float(val)
        for rail, cur, _avg in RAIL.findall(line):
            s[rail] = int(cur)
        by_sec[sec].append(s)
    samples = []
    for sec, rows in sorted(by_sec.items()):
        for i, s in enumerate(rows):
            s["t"] = sec + i / len(rows)
            samples.append(s)
    return samples


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))] if vals else None


def summarise(samples, t0, t1, mean_ms=None):
    w = [s for s in samples if t0 <= s["t"] <= t1]
    if not w:
        return {"n": 0}

    def col(k):
        return [s[k] for s in w if k in s]

    out = {"n": len(w)}
    for k in ("gpu_pct", "emc_pct", "cpu_pct"):
        v = col(k)
        if v:
            out[k + "_mean"] = round(st.mean(v), 1)
            out[k + "_p95"] = pct(v, 0.95)
    for k in ("gpu_mhz", "emc_mhz"):
        v = col(k)
        if v:
            out[k + "_min"], out[k + "_max"] = min(v), max(v)
    if (v := col("ram_mb")):
        out["ram_mb_max"] = max(v)
    for rail in ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"):
        if (v := col(rail)):
            out[rail + "_mw"] = round(st.mean(v))
    for k in ("t_tj", "t_gpu", "t_cpu"):
        if (v := col(k)):
            out[k + "_max"] = max(v)
    if mean_ms and "VDD_IN_mw" in out:
        out["mj_per_call"] = round(out["VDD_IN_mw"] * mean_ms / 1000.0, 1)
    return out


COLS = [("gpu_pct_mean", "GPU %"), ("gpu_pct_p95", "GPU p95 %"), ("gpu_mhz_min", "GPU MHz min"),
        ("emc_pct_mean", "EMC %"), ("emc_pct_p95", "EMC p95 %"), ("cpu_pct_mean", "CPU %"),
        ("ram_mb_max", "RAM MB max"), ("VDD_IN_mw", "Board mW"), ("VDD_CPU_GPU_CV_mw", "GPU+CPU mW"),
        ("t_tj_max", "Tj max C"), ("mj_per_call", "mJ / call"), ("n", "samples")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("phases", nargs="?")
    ap.add_argument("--kind", default="profile")
    ap.add_argument("--window", nargs=2, type=float)
    ap.add_argument("--label", default="window")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    samples = parse(a.log)
    rows = []
    if a.window:
        rows.append((a.label, "", summarise(samples, *a.window)))
    else:
        for line in open(a.phases):
            p = json.loads(line)
            if p.get("kind", "bench") != a.kind:
                continue
            start = p.get("timed_from", p["t0"])
            rows.append((p["model"], " ".join(p["flags"]) or "(defaults)",
                         summarise(samples, start, p["t1"], p.get("mean_ms"))))
    if a.markdown:
        print("| Model | Flags | " + " | ".join(h for _, h in COLS) + " |")
        print("|---|---|" + "--:|" * len(COLS))
        for model, flags, s in rows:
            print(f"| {model} | `{flags}` | " + " | ".join(str(s.get(k, "-")) for k, _ in COLS) + " |")
    else:
        for model, flags, s in rows:
            print(json.dumps({"model": model, "flags": flags, **s}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
