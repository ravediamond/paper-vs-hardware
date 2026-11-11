#!/usr/bin/env python3
"""WS-062 A3 / B1: the task-swap probe for a G0.5 checkpoint (TurboVLA's probe, projects/turbovla/faithful/
horizon_probe.py, adapted): same frame, the yellow and the green instruction, the same noise seed for both; metric =
max |Δ| over the first 10 actions × the 5 arm joints, degrees, after the policy's own postprocessor; median over frames,
episode-bootstrap 95 % CI. Also logs, per frame and instruction, the System 2 subtask text and boxes.

Frames (fixed before any number): the two two-cube rollouts of 2026-10-16 (NOTES.md), yellow named
(`…_110532`) and green named (`…_111039`), 3 episodes each, frames t = 0, 10, 20, 30 of each episode (before and at the
start of the motion) → 24 frames. The cameras are renamed to the checkpoint's SO-101 slots: front_top → exterior,
wrist → wrist_right; the left wrist slot is left empty (the checkpoint's documented single-arm contract).

    # in lerobot-jetson:rc with PYTHONPATH=/lang/src (the g05-pin checkout)
    python s2_swap_probe.py frames --out outputs/s2/frames.npz
    python s2_swap_probe.py probe --policy lerobot/g05_so101 --frames outputs/s2/frames.npz --name zeroshot \\
        --out projects/system2-g05/results/a3_swap_zeroshot.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROLLOUTS = {"yellow": "rollout_smolvla_multitask_orin_nx_normal_20261016_110532",
            "green": "rollout_smolvla_multitask_orin_nx_normal_20261016_111039"}
CACHE = Path.home() / ".cache/huggingface/lerobot/ravediamond"
TIMES = (0, 10, 20, 30)
TASK = {"yellow": "Put the yellow cube in the center", "green": "Put the green cube in the center"}
CAMS = {"observation.images.front_top_camera": "observation.images.exterior",
        "observation.images.wrist_camera": "observation.images.wrist_right"}
WIN, ARM, N_BOOT = 10, slice(0, 5), 2000


def cmd_frames(a):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    rows = {"images": [], "state": [], "episode": [], "t": [], "source": []}
    for colour, name in ROLLOUTS.items():
        ds = LeRobotDataset(f"ravediamond/{name}", root=CACHE / name, video_backend="pyav")
        for ep in range(ds.meta.total_episodes):
            start = int(ds.meta.episodes[ep]["dataset_from_index"])
            for t in TIMES:
                it = ds[start + t]
                assert int(it["episode_index"]) == ep and int(it["frame_index"]) == t
                rows["images"].append(np.stack([(it[k].numpy() * 255).round().astype(np.uint8) for k in CAMS]))
                rows["state"].append(it["observation.state"].numpy())
                rows["episode"].append(f"{colour}-{ep}"); rows["t"].append(t); rows["source"].append(name)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **{k: np.asarray(v) for k, v in rows.items()})
    print(f"{len(rows['t'])} frames -> {a.out}")


def boot_ci(values, groups, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted(set(groups)); by = {k: [v for v, g in zip(values, groups) if g == k] for k in keys}
    stats = [np.median(sum((by[k] for k in rng.choice(keys, len(keys))), [])) for _ in range(N_BOOT)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def cmd_probe(a):
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.g05.modeling_g05 import G05Policy

    f = np.load(a.frames)
    policy = G05Policy.from_pretrained(a.policy, revision=a.revision)
    policy.config.runtime_system = "system2"
    policy.eval().to("cuda")
    pre, post = make_pre_post_processors(policy.config, a.policy, preprocessor_overrides={"device_processor": {"device": "cuda"}})
    rows = []
    for i in range(len(f["t"])):
        out = {}
        for colour in ("yellow", "green"):
            batch = {CAMS[k]: torch.from_numpy(f["images"][i, j]).float().div(255)[None] for j, k in enumerate(CAMS)}
            batch["observation.state"] = torch.from_numpy(f["state"][i]).float()[None]
            batch["task"] = [TASK[colour]]
            torch.manual_seed(a.seed + i)                      # the same noise for both instructions
            t0 = time.perf_counter()
            res = policy.predict_action_chunk(pre(batch))
            ms = (time.perf_counter() - t0) * 1e3
            chunk = res["action"] if not isinstance(res, torch.Tensor) else res
            pred = {} if isinstance(res, torch.Tensor) else (res.get("prediction") or {})
            deg = post(chunk).float().cpu().numpy()[0]
            out[colour] = {"chunk": deg, "ms": ms, "subtask": (pred.get("language") or {}).get("subtask", [None])[0],
                           "boxes": str((pred.get("boxes") or {}))[:400]}
        swap = float(np.abs(out["yellow"]["chunk"][:WIN, ARM] - out["green"]["chunk"][:WIN, ARM]).max())
        rows.append({"frame": i, "episode": str(f["episode"][i]), "t": int(f["t"][i]), "swap_deg": swap,
                     **{f"{c}_{k}": out[c][k] for c in out for k in ("ms", "subtask", "boxes")}})
        print(i, f["episode"][i], f["t"][i], round(swap, 3), out["yellow"]["subtask"], "|", out["green"]["subtask"], flush=True)
    sw = [r["swap_deg"] for r in rows]; eps = [r["episode"] for r in rows]
    named = sum(("yellow" in (r["yellow_subtask"] or "").lower()) + ("green" in (r["green_subtask"] or "").lower()) for r in rows)
    res = {"name": a.name, "policy": a.policy, "revision": a.revision, "frames": len(rows), "episodes": len(set(eps)),
           "metric": "max |Δ| over the first 10 actions × 5 arm joints, degrees, same noise for both instructions",
           "swap_median_deg": float(np.median(sw)), "swap_ci95": boot_ci(sw, eps),
           "subtask_names_the_instructed_colour": f"{named}/{2 * len(rows)}", "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("frames"); fr.add_argument("--out", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--policy", required=True); pr.add_argument("--revision")
    pr.add_argument("--frames", required=True); pr.add_argument("--name", required=True)
    pr.add_argument("--seed", type=int, default=1000); pr.add_argument("--out", required=True)
    a = ap.parse_args()
    {"frames": cmd_frames, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
