"""HORIZON_PROTOCOL.md probe (H50_BRIEF step 1): task swap and per-layout in the decision band of pre-grasp progress,
the same frames and metrics for every checkpoint. Frame choices: data/2026-10-24_horizon/README.md (fixed before any
number).

    # LeRobot env: pick the frames (band, out-of-band, episode starts) and cache them at 256 px for upstream's loader
    python horizon_probe.py cache --root <so101_bowls_all_trimmed> --split split.json --out <cache dir>
    # upstream env (transformers 4.56 overlay): a TurboVLA .pth checkpoint on the cached frames
    python horizon_probe.py upstream --upstream third_party/TurboVLA --ckpt <.pth> --cache <cache dir> --stats <stats.json> \\
        --dinov3 <dir> --bert <dir> --out <chunks.npz>
    # LeRobot env: a LeRobot policy (the SmolVLA control) on the same frames, from the dataset itself
    python horizon_probe.py lerobot --policy <dir> --root <so101_bowls_all_trimmed> --cache <cache dir> --out <chunks.npz>
    # any env: metrics
    python horizon_probe.py report --cache <cache dir> --chunks <chunks.npz> [--name x] --out <metrics.json>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASKS = ("Put the blue bowl in the pink bowl.", "Put the pink bowl in the blue bowl.")
FACT7_CHUNK, BAND, OUT_STRIDE, WIN, PAN = 12, (0.10, 0.30), 3, 12, 0
N_START, N_BOOT = 6, 2000


def grasp_frame_fn():
    sys.path.insert(0, str(HERE.parent / "scripts"))          # fact 7's grasp detector (on vast: /root/wm/scripts)
    from train_turbovla_paired import grasp_frame

    return grasp_frame


def kinds(gripper_cmd):
    """One episode's frames (in order): kind (band | out | -) and fact 7's progress p = t / (g - 12); None if no grasp."""
    g = grasp_frame_fn()(np.asarray(gripper_cmd))
    if g is None or g <= FACT7_CHUNK:
        return None, g
    p = np.arange(len(gripper_cmd)) / (g - FACT7_CHUNK)
    k = np.full(len(p), "-", dtype="<U5")
    k[(p >= BAND[0]) & (p < BAND[1])] = "band"
    out = np.flatnonzero(((p < BAND[0]) | (p >= BAND[1])) & (p < 1.0))[::OUT_STRIDE]
    k[out] = "out"
    return (k, p), g


def frames_of(root, episodes):
    """(global index, episode, t, kind, progress) per probe frame; kind = band | out | start."""
    import pandas as pd

    df = pd.concat([pd.read_parquet(p, columns=["index", "episode_index", "frame_index", "action"])
                    for p in sorted(Path(root).glob("data/*/*.parquet"))]).sort_values("index")
    rows, info = [], {}
    for e in episodes:
        d = df[df.episode_index == e]
        kp, g = kinds(np.stack(d.action.to_numpy())[:, 5])
        idx = d["index"].to_numpy()
        info[e] = {"grasp_frame": g, "length": len(d)}
        if kp is None:
            continue
        k, p = kp
        for kind in ("band", "out"):
            rows += [(int(idx[t]), int(e), int(t), kind, float(p[t])) for t in np.flatnonzero(k == kind)]
        info[e]["band_frames"] = int((k == "band").sum())
    for j in np.linspace(0, len(episodes) - 1, N_START, dtype=int):
        d = df[df.episode_index == episodes[j]]
        rows.append((int(d["index"].iloc[0]), int(episodes[j]), 0, "start", 0.0))
    return rows, info


def metrics(blue, pink, episode, kind, layout, progress):
    """HORIZON_PROTOCOL.md metrics 1-2 (+ reported-beside values) from paired chunks (N, H, 6) in degrees."""
    sw12 = np.abs(blue[:, :WIN] - pink[:, :WIN]).max((1, 2))
    swf = np.abs(blue - pink).max((1, 2))
    sign = np.where(np.asarray(layout) == "B", 1.0, -1.0)    # left-bowl instruction minus the other (gate v3's sign)
    d12 = sign * (blue[:, WIN - 1, PAN] - pink[:, WIN - 1, PAN])
    dend = sign * (blue[:, -1, PAN] - pink[:, -1, PAN])
    res = {"horizon": int(blue.shape[1])}
    for k in ("band", "out"):
        m = kind == k
        eps = sorted(set(episode[m].tolist()))
        pe = {e: m & (episode == e) for e in eps}
        s12 = [float(sw12[pe[e]].mean()) for e in eps]
        sf = [float(swf[pe[e]].mean()) for e in eps]
        l12 = [float((d12[pe[e]] > 0).mean()) for e in eps]
        lend = [float((dend[pe[e]] > 0).mean()) for e in eps]
        res[k] = {"episodes": len(eps), "frames": int(m.sum()),
                  "task_swap_12_median_deg": float(np.median(s12)), "task_swap_12_ci95": boot_ci(s12),
                  "task_swap_full_median_deg": float(np.median(sf)), "task_swap_full_ci95": boot_ci(sf),
                  "per_layout_step12_mean": float(np.mean(l12)), "per_layout_chunk_end_mean": float(np.mean(lend)),
                  "per_episode": {str(e): {"task_swap_12": s12[j], "task_swap_full": sf[j], "per_layout_12": l12[j]}
                                  for j, e in enumerate(eps)}}
    bins = np.linspace(0, 1, 11)                                  # exploratory: swap against progress
    pre = np.isin(kind, ("band", "out"))
    res["exploratory_swap_vs_progress"] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (progress >= lo) & (progress < hi) & pre
        res["exploratory_swap_vs_progress"].append({"p": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                                                    "task_swap_12_median_deg": float(np.median(sw12[m])) if m.any() else None})
    return res


def cmd_cache(a):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.nn import functional as F

    sys.path.insert(0, str(HERE))
    from so101_data import CAMS, CHUNK, FPS, SIZE, layout

    split = json.loads(Path(a.split).read_text())
    rows, info = frames_of(a.root, split["val_episodes"])
    ds = LeRobotDataset(f"local/{Path(a.root).name}", root=a.root, video_backend="pyav",
                        delta_timestamps={"action": [i / FPS for i in range(CHUNK)]})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    imgs = np.lib.format.open_memmap(out / "images.npy", "w+", np.uint8, (n, 2, SIZE, SIZE, 3))
    state, action, pad = np.zeros((n, 6), np.float32), np.zeros((n, CHUNK, 6), np.float32), np.zeros((n, CHUNK), bool)
    task, lay, tasks = np.zeros(n, np.int64), np.full(n, "-", dtype="<U1"), {t: i for i, t in enumerate(TASKS)}
    for i, (gi, e, t, _, _) in enumerate(rows):
        it = ds[gi]
        assert int(it["episode_index"]) == e and int(it["frame_index"]) == t, (gi, e, t)
        for v, c in enumerate(CAMS):
            x = F.interpolate(it[c][None].float(), size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
            imgs[i, v] = (x[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
        state[i], action[i], pad[i] = it["observation.state"].numpy(), it["action"].numpy(), it["action_is_pad"].numpy()
        task[i], lay[i] = tasks[it["task"]], layout(it[CAMS[0]].numpy())
        if i % 200 == 0:
            print(f"cache {i}/{n}", flush=True)
    imgs.flush()
    cols = dict(state=state, action=action, pad=pad, task=task, layout=lay,
                index=np.array([r[0] for r in rows]), episode=np.array([r[1] for r in rows]),
                frame=np.array([r[2] for r in rows]), kind=np.array([r[3] for r in rows]),
                progress=np.array([r[4] for r in rows]))
    for name, arr in cols.items():
        np.save(out / f"{name}.npy", arr)
    meta = {"root": str(a.root), "frames": n, "episodes": split["val_episodes"], "tasks": list(TASKS), "size": SIZE,
            "chunk": CHUNK, "episode_info": {str(k): v for k, v in info.items()},
            "kinds": {k: int((cols["kind"] == k).sum()) for k in ("band", "out", "start")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(meta["kinds"], sum(1 for v in info.values() if v.get("band_frames")), "episodes with band frames")


def load_cache(path):
    sys.path.insert(0, str(HERE))
    from so101_data import Cache

    c = Cache(path)
    for name in ("index", "kind", "progress"):
        setattr(c, name, np.load(Path(path) / f"{name}.npy"))
    return c


def cmd_upstream(a):
    import torch

    sys.path.insert(0, str(a.upstream))
    sys.path.insert(0, str(HERE))
    from so101_data import Normaliser, sample
    from turbovla.evaluation.policy import _checkpoint_state_dict, _strip_module_prefix
    from turbovla.models.configuration import TurboVLAConfig
    from turbovla.models.turbovla import build_turbovla

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = TurboVLAConfig.from_mapping(ck["model_config"])
    cfg.text.model_name_or_path, cfg.vision.model_name_or_path = a.bert, a.dinov3
    model = build_turbovla(cfg)
    model.load_state_dict(_strip_module_prefix(_checkpoint_state_dict(ck)), strict=True)
    model.to(dev).float().eval()                                    # as gate_upstream.py
    del ck
    cache, norm = load_cache(a.cache), Normaliser(json.loads(Path(a.stats).read_text()))
    out = {t: [] for t in TASKS}
    with torch.inference_mode():
        for i in range(len(cache)):
            for t in TASKS:
                (v1, v2), instr, st, _, _ = sample(cache, i, norm, task=t)
                s = {"dinov3": torch.stack([v1["dinov3"], v2["dinov3"]])[None].to(dev)}
                out[t].append(norm.unaction(model([instr], s, st[None].to(dev)).float().cpu()[0]).numpy())
            if i % 100 == 0:
                print(f"upstream {i}/{len(cache)}", flush=True)
    np.savez_compressed(a.out, blue=np.stack(out[TASKS[0]]), pink=np.stack(out[TASKS[1]]), ckpt=str(a.ckpt))


def cmd_lerobot(a):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PreTrainedConfig.from_pretrained(a.policy); cfg.device = dev
    pol = get_policy_class(cfg.type).from_pretrained(a.policy, config=cfg).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=a.policy,
                                         preprocessor_overrides={"device_processor": {"device": dev}})
    ds = LeRobotDataset(f"local/{Path(a.root).name}", root=a.root, video_backend="pyav")
    cache = load_cache(a.cache)
    H, A = cfg.chunk_size, cfg.max_action_dim
    out = {t: [] for t in TASKS}
    with torch.inference_mode():
        for i, gi in enumerate(cache.index):
            it = ds[int(gi)]
            noise = torch.randn((1, H, A), generator=torch.Generator().manual_seed(int(gi))).to(dev)
            for t in TASKS:
                b = {k: (v[None] if torch.is_tensor(v) else v) for k, v in it.items()}
                b["task"] = [t]
                b = pre(b)
                ch = pol.predict_action_chunk(b, noise=noise)
                out[t].append(post(ch)[0].float().cpu().numpy())
            if i % 50 == 0:
                print(f"lerobot {i}/{len(cache)}", flush=True)
    np.savez_compressed(a.out, blue=np.stack(out[TASKS[0]]), pink=np.stack(out[TASKS[1]]), ckpt=str(a.policy))


def boot_ci(per_ep, stat=np.median, seed=0):
    x = np.asarray(per_ep, float)
    rng = np.random.default_rng(seed)
    b = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def cmd_report(a):
    c = load_cache(a.cache)
    z = np.load(a.chunks)
    blue, pink = z["blue"], z["pink"]
    res = {"chunks": str(a.chunks), "name": a.name,
           "frames": {k: int((c.kind == k).sum()) for k in ("band", "out", "start")},
           **metrics(blue, pink, c.episode, c.kind, c.layout, c.progress)}
    st = c.kind == "start"
    own = np.where((c.task[st] == 0)[:, None, None], blue[st], pink[st])
    res["start_reach_median_deg"] = float(np.median(np.abs(own - c.state[st][:, None, :]).max((1, 2))))
    res["excluded_episodes_no_band"] = sorted(int(e) for e in c.meta["episodes"]
                                              if not c.meta["episode_info"][str(e)].get("band_frames"))
    b, o = res["band"], res["out"]
    res["instrument_check"] = {"band_median_ge_5": b["task_swap_12_median_deg"] >= 5.0,
                               "band_ci_low_ge_2_5": b["task_swap_12_ci95"][0] >= 2.5,
                               "band_ge_2x_out": b["task_swap_12_median_deg"] >= 2 * o["task_swap_12_median_deg"]}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("band", "out", "exploratory_swap_vs_progress")}))
    for k in ("band", "out"):
        print(k, {kk: v for kk, v in res[k].items() if kk != "per_episode"})


def cmd_gate3_layout(a):
    """Amendment 1.3: the probe's per-layout metric (chunk-end shoulder_pan, gate v3's sign, this file's inference path:
    dataset item -> checkpoint processors, noise seeded by the global index) on gate v3's own frames: 2 x 6 episodes
    spread over the untrimmed dataset, at 0 / 10 / 20 % of each episode."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    sys.path.insert(0, str(HERE))
    from so101_data import CAMS, layout

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PreTrainedConfig.from_pretrained(a.policy); cfg.device = dev
    pol = get_policy_class(cfg.type).from_pretrained(a.policy, config=cfg).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=a.policy,
                                         preprocessor_overrides={"device_processor": {"device": dev}})
    ds = LeRobotDataset(f"local/{Path(a.root).name}", root=a.root, video_backend="pyav")
    starts, ends = ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"]
    tasks = sorted(ds.meta.tasks.index.tolist())
    colour = {t: ("blue" if t.index("blue") < t.index("pink") else "pink") for t in tasks}
    H, A = cfg.chunk_size, cfg.max_action_dim
    rows = []
    with torch.inference_mode():
        for ep in np.linspace(0, len(starts) - 1, 12, dtype=int):
            for prog in (0.0, 0.1, 0.2):
                gi = int(starts[ep]) + int(prog * (int(ends[ep]) - int(starts[ep])))
                it = ds[gi]
                left = "blue" if layout(it[CAMS[0]].numpy()) == "B" else "pink"
                noise = torch.randn((1, H, A), generator=torch.Generator().manual_seed(gi)).to(dev)
                out = {}
                for t in tasks:
                    b = {k: (v[None] if torch.is_tensor(v) else v) for k, v in it.items()}
                    b["task"] = [t]
                    out[colour[t]] = post(pol.predict_action_chunk(pre(b), noise=noise))[0].float().cpu().numpy()
                other = "pink" if left == "blue" else "blue"
                rows.append({"episode": int(ep), "progress": prog, "index": gi, "left": left,
                             "diff_end": float(out[left][-1, PAN] - out[other][-1, PAN])})
    correct = float(np.mean([r["diff_end"] > 0 for r in rows]))
    res = {"policy": a.policy, "dataset": a.root, "frames": len(rows), "per_layout_chunk_end_correct": correct,
           "median_diff_deg": float(np.median([r["diff_end"] for r in rows])), "reproduces_gate_v3_ge_75": correct >= 0.75,
           "gate_v3_reference": "86 % (2026-10-01, M1 MPS, random noise, robot-observation path)", "rows": rows}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != "rows"})


def cmd_report_a1(a):
    """HORIZON_PROTOCOL amendment 1: swap over each checkpoint's full chunk; out-of-band = p in [0, 0.10) only;
    per-layout at the chunk end (gated only if gate3-layout reproduced gate v3). First-12-step swap reported beside."""
    c = load_cache(a.cache)
    z = np.load(a.chunks)
    kind = c.kind.copy()
    kind[(kind == "out") & (c.progress >= BAND[0])] = "-"            # frames after the band leave the null set
    res = {"chunks": str(a.chunks), "name": a.name, "amendment": "HORIZON_PROTOCOL.md amendment 1",
           "frames": {k: int((kind == k).sum()) for k in ("band", "out", "start")},
           **metrics(z["blue"], z["pink"], c.episode, kind, c.layout, c.progress)}
    b, o = res["band"], res["out"]
    res["instrument_check_a1"] = {"band_full_median_ge_5": b["task_swap_full_median_deg"] >= 5.0,
                                  "band_full_ci_low_ge_2_5": b["task_swap_full_ci95"][0] >= 2.5,
                                  "band_full_ge_2x_out_full": b["task_swap_full_median_deg"] >= 2 * o["task_swap_full_median_deg"]}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["instrument_check_a1"]), json.dumps(res["frames"]))
    for k in ("band", "out"):
        print(k, {kk: v for kk, v in res[k].items() if kk != "per_episode"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cache"); p.add_argument("--root", required=True); p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("upstream")
    for k in ("upstream", "ckpt", "cache", "stats", "dinov3", "bert", "out"):
        p.add_argument("--" + k, required=True)
    p = sub.add_parser("lerobot")
    for k in ("policy", "root", "cache", "out"):
        p.add_argument("--" + k, required=True)
    p = sub.add_parser("report")
    for k in ("cache", "chunks", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--name", default="")
    p = sub.add_parser("report-a1")
    for k in ("cache", "chunks", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--name", default="")
    p = sub.add_parser("gate3-layout")
    for k in ("policy", "root", "out"):
        p.add_argument("--" + k, required=True)
    a = ap.parse_args()
    {"cache": cmd_cache, "upstream": cmd_upstream, "lerobot": cmd_lerobot, "report": cmd_report, "report-a1": cmd_report_a1,
     "gate3-layout": cmd_gate3_layout}[a.cmd](a)
