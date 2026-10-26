"""FAITHFUL_PROTOCOL.md step 2 gate: sensitivity_gate_v3.py's task swap, per-layout and episode-start probes, ported to
an upstream TurboVLA checkpoint trained by train_so101.py, on the val cache (so101_data.build_cache --which val).

Same rules as v3: task swap = max |chunk(task) - chunk(other)| in degrees over the chunk and joints, at --frames frames
spread over the val frames, median >= 5 deg passes; per-layout = 2 x --start-episodes episodes at 0 / 10 / 20 % progress,
share where the left-bowl instruction ends its plan with more shoulder_pan than the right-bowl one, >= 75 % passes;
start reach = max |chunk - state| at the first frame of --start-episodes episodes. Upstream env (transformers 4.56).
Plus HORIZON_PROTOCOL.md's band probe (horizon_probe.metrics) on every band / out-of-band frame of the val cache, as
`band_probe` in the output (the chunk length is the checkpoint's: 12 for path A, 50 for H=50).

    python gate_upstream.py --upstream third_party/TurboVLA --ckpt <turbovla_step_35000.pth> --cache <val cache> \\
        --stats stats.json --dinov3 <dir> --bert <dir> --out gate.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

TASKS = ("Put the blue bowl in the pink bowl.", "Put the pink bowl in the blue bowl.")   # meta/tasks.parquet
PAN = 0                                                                                    # shoulder_pan.pos

ap = argparse.ArgumentParser()
ap.add_argument("--upstream", type=Path, required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--stats", required=True)
ap.add_argument("--dinov3", required=True)
ap.add_argument("--bert", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--start-episodes", type=int, default=6)
ap.add_argument("--min-task-swap", type=float, default=5.0)
ap.add_argument("--min-layout-correct", type=float, default=0.75)
args = ap.parse_args()
sys.path.insert(0, str(args.upstream))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_data import Cache, Normaliser, sample  # noqa: E402
from turbovla.evaluation.policy import _checkpoint_state_dict, _strip_module_prefix  # noqa: E402
from turbovla.models.configuration import TurboVLAConfig  # noqa: E402
from turbovla.models.turbovla import build_turbovla  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = TurboVLAConfig.from_mapping(ck["model_config"])
cfg.text.model_name_or_path, cfg.vision.model_name_or_path = args.bert, args.dinov3
model = build_turbovla(cfg)
model.load_state_dict(_strip_module_prefix(_checkpoint_state_dict(ck)), strict=True)
model.to(dev).float().eval()                                    # trained in fp32 with bf16 autocast for DINOv3
del ck
cache, norm = Cache(args.cache), Normaliser(json.loads(Path(args.stats).read_text()))


@torch.inference_mode()
def chunk(i, task):
    (v1, v2), instr, state, _, _ = sample(cache, i, norm, task=task)
    samples = {"dinov3": torch.stack([v1["dinov3"], v2["dinov3"]])[None].to(dev)}
    return norm.unaction(model([instr], samples, state[None].to(dev)).float().cpu()[0]).numpy()


task_of = lambda i: cache.meta["tasks"][cache.task[i]]  # noqa: E731
other = lambda t: next(x for x in TASKS if x != t)  # noqa: E731
rows = []
for i in np.linspace(0, len(cache) - 1, args.frames + 2, dtype=int)[1:-1]:
    a, b = chunk(int(i), task_of(i)), chunk(int(i), other(task_of(i)))
    rows.append({"frame": int(i), "episode": int(cache.episode[i]), "task_swap": float(np.abs(a - b).max())})
    print(f"frame {i:6d} (ep {cache.episode[i]})  task_swap {rows[-1]['task_swap']:6.2f} deg", flush=True)

eps = sorted(set(cache.episode.tolist()))
first = {e: int(np.flatnonzero(cache.episode == e)[0]) for e in eps}
last = {e: int(np.flatnonzero(cache.episode == e)[-1]) for e in eps}
reach = []
for k in np.linspace(0, len(eps) - 1, args.start_episodes, dtype=int):
    i = first[eps[k]]
    reach.append(float(np.abs(chunk(i, task_of(i)) - cache.state[i]).max()))

colour = {t: re.search(r"(blue|pink)", t).group(1) for t in TASKS}
diffs = []
for k in np.linspace(0, len(eps) - 1, 2 * args.start_episodes, dtype=int):
    e = eps[k]
    for progress in (0.0, 0.1, 0.2):
        i = first[e] + int(progress * (last[e] - first[e]))
        left = "blue" if cache.layout[i] == "B" else "pink"
        t_left = next(t for t in TASKS if colour[t] == left)
        diffs.append(float(chunk(i, t_left)[-1, PAN] - chunk(i, other(t_left))[-1, PAN]))

# HORIZON_PROTOCOL.md band probe: fact 7's progress per val episode from the cache's gripper command
from horizon_probe import kinds, metrics  # noqa: E402

kind, prog = np.full(len(cache), "-", dtype="<U5"), np.zeros(len(cache))
for e in eps:
    ix = np.flatnonzero(cache.episode == e)
    ix = ix[np.argsort(cache.frame[ix])]
    kp, _ = kinds(cache.action[ix, 0, 5])
    if kp is not None:
        kind[ix], prog[ix] = kp
sel = np.flatnonzero(np.isin(kind, ("band", "out")))
pairs = {t: np.stack([chunk(int(i), t) for i in sel]) for t in TASKS}
band = metrics(pairs[TASKS[0]], pairs[TASKS[1]], cache.episode[sel], kind[sel], cache.layout[sel], prog[sel])
band["excluded_episodes_no_band"] = sorted(int(e) for e in eps if not (kind[cache.episode == e] == "band").any())
print("band probe:", {k: v for k, v in band["band"].items() if k != "per_episode"}, flush=True)

ts, lay = float(np.median([r["task_swap"] for r in rows])), float(np.mean(np.array(diffs) > 0))
res = {"protocol": "projects/turbovla/FAITHFUL_PROTOCOL.md step 2", "checkpoint": args.ckpt, "cache": args.cache,
       "val_episodes": len(eps), "task_swap_median_deg": ts, "task_swap_pass": ts >= args.min_task_swap,
       "per_layout_correct": lay, "per_layout_pass": lay >= args.min_layout_correct,
       "per_layout_median_pan_diff_deg": float(np.median(diffs)), "start_reach_median_deg": float(np.median(reach)),
       "rows": rows, "per_layout_diffs": diffs, "start_reach": reach, "band_probe": band}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(res, indent=1))
print({k: v for k, v in res.items() if k not in ("rows", "per_layout_diffs", "start_reach")})
