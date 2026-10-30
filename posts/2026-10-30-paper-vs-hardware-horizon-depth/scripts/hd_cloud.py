#!/usr/bin/env python3
"""WS-057 gate A1 / A3 (projects/horizon-depth/PROTOCOL.md): SmolVLA's action cloud vs Euler steps N, PyTorch.

A1:  --steps 1,2,3,5,10 --draws 32 --frames-at 0.4   -> projects/horizon-depth/results/a1_cloud.json
A3:  --openloop --s 10,25,50 --steps 2,10            -> projects/horizon-depth/results/a3_openloop.json

Frames: G13b's 20 episodes of ravediamond/so101_bowls_all (seed-0 list in g13b_summary.json). Noise: G13b's generator,
seed k for draw k, shared across N. N is set per call by overriding the policy config's `num_steps`. Every call is a
single-sample call (no batching), so draw 0 at N = 10 can be checked against the normal path to 0.01°. Degrees for
the 5 arm joints; the gripper is reported separately. Run inside lerobot-jetson:rc like G13b:

    IMAGE=lerobot-jetson:rc LEROBOT_SRC=image shared/gate/jetson_run.sh --python scripts/hd_cloud.py \\
        --policy-path ravediamond/smolvla_2026-09-24_14-34-40 --revision c1d76f6 --device cuda [--openloop ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_g13b_rtc_offline as G  # noqa: E402  LeRobotSmolVLA (loading, pre/post-processors, noise generator)

OUT = ROOT / "projects/horizon-depth/results"
G13B = ROOT / "shared/gate/data/2026-10-19_g13b_offline/g13b_summary.json"
ARM = slice(0, 5)


def chunk(m, obs, seed, n_steps):
    m.policy.config.num_steps = n_steps
    return m.to_robot(m.predict(obs, m.noise(seed), "naive", 0, None))  # (H, 6) robot units (degrees)


def obs_at(ds, ep, t, keys):
    states, task = ds.states(ep), ds.task(ep)
    imgs = {k: ds.frame(ep, t, k) for k in keys}
    return {"state": states[t], "task": task, **imgs}


def cloud_metrics(c, c10):
    """c, c10: (K, H, 6) draws at N and at 10 (paired by seed)."""
    def on(x, cols):
        return x[:, :, cols].reshape(len(x), -1)
    out = {}
    for name, cols in (("arm", ARM), ("gripper", slice(5, 6))):
        a, a10 = on(c, cols), on(c10, cols)
        mu, mu10 = a.mean(0), a10.mean(0)
        R = float(np.mean(np.linalg.norm(a - mu, axis=1)))
        R10 = float(np.mean(np.linalg.norm(a10 - mu10, axis=1)))
        dmu = float(np.linalg.norm(mu - mu10) / np.mean(np.linalg.norm(a10, axis=1)))
        x, y = (a10 - mu10).ravel(), (a - mu).ravel()  # best scalar map a(N) - mean ~ c * (a(10) - mean)
        k = float(x @ y / (x @ x)) if x @ x > 0 else 0.0
        resid = float(np.sum((y - k * x) ** 2) / np.sum(y ** 2)) if np.sum(y ** 2) > 0 else 0.0
        out[name] = {"R": R, "R10": R10, "R_ratio": R / R10 if R10 > 0 else None, "dmu": dmu,
                     "max_abs_centroid_deg": float(np.max(np.abs(mu - mu10))), "linfit_unexplained": resid}
    return out


def boot_ci(vals, n=2000, seed=0):
    v = np.asarray([x for x in vals if x is not None], float)
    rng = np.random.default_rng(seed)
    meds = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def a1(m, ds, eps, steps, draws, at, keys):
    rows = []
    for e, ep in enumerate(eps):
        T = len(ds.states(ep)); t = int(at * T)
        obs = obs_at(ds, ep, t, keys)
        t0 = time.perf_counter()
        cl = {n: np.stack([chunk(m, obs, k, n) for k in range(draws)]) for n in steps}
        rows.append({"episode": ep, "frame": t, "by_n": {str(n): cloud_metrics(cl[n], cl[10]) for n in steps}})
        print(f"A1 ep {ep}: frame {t}, {time.perf_counter() - t0:.0f} s", flush=True)
    summ = {}
    for n in steps:
        r = [x["by_n"][str(n)]["arm"] for x in rows]
        summ[str(n)] = {k: {"median": float(np.median([y[k] for y in r])), "ci95": boot_ci([y[k] for y in r])}
                        for k in ("R_ratio", "dmu", "max_abs_centroid_deg", "linfit_unexplained")}
        g = [x["by_n"][str(n)]["gripper"] for x in rows]
        summ[str(n)]["gripper_R_ratio_median"] = float(np.median([y["R_ratio"] for y in g if y["R_ratio"] is not None])) \
            if any(y["R_ratio"] is not None for y in g) else None
    s2 = summ["2"]
    bar = {"R_ratio": s2["R_ratio"]["median"], "dmu": s2["dmu"]["median"], "max_abs_deg": s2["max_abs_centroid_deg"]["median"]}
    if bar["R_ratio"] <= 0.70 and bar["dmu"] <= 0.05 and bar["max_abs_deg"] <= 3.0:
        verdict = "pass"
    elif bar["R_ratio"] > 0.85 or bar["dmu"] > 0.10:
        verdict = "fail"
    else:
        verdict = "not proven"
    return {"rows": rows, "summary": summ, "bar_A1": bar, "verdict_A1": verdict}


def a3(m, ds, eps, s_list, steps, keys):
    res = {}
    for s in s_list:
        for n in steps:
            res[f"s{s}_N{n}"] = []
    for e, ep in enumerate(eps):
        states, acts, task = ds.states(ep), ds.actions(ep), ds.task(ep)
        T = len(states)
        need = sorted({t for s in s_list for t in range(0, T - max(s_list), s)})
        frames = {k: {i: im for i, im in ds.iter_frames(ep, k) if i in set(need)} for k in keys}
        for s in s_list:
            for b, t in enumerate(range(0, T - s, s)):
                if t not in frames[keys[0]]:
                    continue
                obs = {"state": states[t], "task": task, **{k: frames[k][t] for k in keys}}
                seed = 1000 * e + b
                for n in steps:
                    c = chunk(m, obs, seed, n)
                    res[f"s{s}_N{n}"].append(float(np.mean(np.abs(c[:s, ARM] - acts[t:t + s, ARM]))))
        print(f"A3 ep {ep} done", flush=True)
    return {"median_mean_abs_deg": {k: float(np.median(v)) for k, v in res.items()},
            "boundaries": {k: len(v) for k, v in res.items()}, "note": "descriptive; training episodes"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--repo-id", default="ravediamond/so101_bowls_all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", default="1,2,3,5,10")
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--frames-at", type=float, default=0.4)
    ap.add_argument("--openloop", action="store_true")
    ap.add_argument("--s", default="10,25,50")
    ap.add_argument("--check-only", action="store_true", help="the A1 measurement checks, then stop")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoModel  # noqa: F401  (loaded by LeRobot)
    from so101_harness.episodes import EpisodeDataset, download

    G.check_hook()
    path = snapshot_download(a.policy_path, revision=a.revision) if a.revision else a.policy_path
    m = G.LeRobotSmolVLA(path, a.device)
    default_steps = int(m.policy.config.num_steps)
    ds = EpisodeDataset(download(a.repo_id, with_videos=True))
    eps = json.loads(G13B.read_text())["meta"]["episodes"]
    keys = m.image_keys
    OUT.mkdir(parents=True, exist_ok=True)
    steps = [int(x) for x in a.steps.split(",")]
    # measurement check: the override at the checkpoint's default reproduces the normal path for draw 0
    T0 = len(ds.states(eps[0])); obs = obs_at(ds, eps[0], int(a.frames_at * T0), keys)
    normal = m.to_robot(m.predict(obs, m.noise(0), "naive", 0, None))
    over = chunk(m, obs, 0, 10)
    m.policy.config.num_steps = default_steps
    check = {"checkpoint_num_steps": default_steps, "draw0_N10_vs_normal_max_abs": float(np.max(np.abs(over - normal)))}
    print("check:", check, flush=True)
    if default_steps != 10 or check["draw0_N10_vs_normal_max_abs"] > 0.01:
        sys.exit(f"measurement check failed: {check}")
    if a.check_only:
        return 0
    if a.openloop:
        res = {"check": check, **a3(m, ds, eps, [int(x) for x in a.s.split(",")], steps, keys)}
        (OUT / "a3_openloop.json").write_text(json.dumps(res, indent=1))
    else:
        res = {"check": check, "episodes": eps, "draws": a.draws, "frames_at": a.frames_at,
               **a1(m, ds, eps, steps, a.draws, a.frames_at, keys)}
        (OUT / "a1_cloud.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ("bar_A1", "verdict_A1")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
