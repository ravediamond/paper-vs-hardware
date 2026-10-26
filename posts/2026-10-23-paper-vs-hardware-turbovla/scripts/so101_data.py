"""SO-101 bowls data in upstream TurboVLA's training format (FAITHFUL_PROTOCOL.md step 2, path A; amendment 1).

Two halves, two environments:
- `build_cache` (LeRobot env): decodes the dataset once into a frame cache: per view 256 x 256 uint8 (antialiased resize
  of the native 640 x 480), raw state, `chunk`-step raw action chunk (default 12, `--chunk`) + pad mask (LeRobot
  repeats the last action past the
  episode end), instruction id, episode / frame index, and the bowl layout of the native top frame (gate v3's
  `layout()`, val frames only).
- `So101BowlsDataset` (upstream env, numpy only): yields exactly what upstream's `LiberoRLDSDataset` yields and
  `vla_collate_fn` takes: ((front_top, wrist) as {"dinov3": (3, 256, 256)}, instruction, state (6,), action chunk
  (chunk, 6), mask (chunk,), the cache's chunk length). Images get DINOv3's ImageNet normalisation (upstream's
  processor on 256-px frames); state is
  z-scored, actions min-max to [-1, 1] and clamped, with train-split statistics (stats file), like upstream's loader.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

CAMS = ("observation.images.front_top_camera", "observation.images.wrist_camera")
SIZE, CHUNK, FPS = 256, 12, 30
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def compute_stats(root, train_episodes) -> dict:
    import pandas as pd

    df = pd.concat([pd.read_parquet(p, columns=["episode_index", "action", "observation.state"])
                    for p in sorted(Path(root).glob("data/*/*.parquet"))])
    df = df[df.episode_index.isin(train_episodes)]
    a, s = np.stack(df["action"].to_numpy()), np.stack(df["observation.state"].to_numpy())
    return {"episodes": "train_episodes of split.json", "n_frames": int(len(df)),
            "action": {"min": a.min(0).tolist(), "max": a.max(0).tolist()},
            "state": {"mean": s.mean(0).tolist(), "std": s.std(0).tolist()}}


def layout(img):
    """Gate v3's bowl layout from the native (3, 480, 640) top frame: "B" = blue bowl on the left."""
    r, g, b = img[:, 150:330, :]
    sc = np.where((r + g + b) > 2.0, b - r, np.nan)
    return "B" if np.nanmean(sc[:, :320]) > np.nanmean(sc[:, 320:]) else "P"


def build_cache(root, episodes, out, with_layout=False, workers=8, chunk=CHUNK):
    """LeRobot env: decode `episodes` of the dataset at `root` into `out` (memmapped .npy + meta.json)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.nn import functional as F

    ds = LeRobotDataset(f"local/{Path(root).name}", root=root, episodes=list(episodes), video_backend="pyav",
                        delta_timestamps={"action": [i / FPS for i in range(chunk)]})
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    n = len(ds)
    imgs = np.lib.format.open_memmap(out / "images.npy", "w+", np.uint8, (n, 2, SIZE, SIZE, 3))
    state, action = np.zeros((n, 6), np.float32), np.zeros((n, chunk, 6), np.float32)
    pad, ep, frame, task = np.zeros((n, chunk), bool), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
    lay, tasks = np.full(n, "-", dtype="<U1"), {}
    loader = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers)
    for i, it in enumerate(loader):
        for v, c in enumerate(CAMS):
            x = F.interpolate(it[c][None].float(), size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
            imgs[i, v] = (x[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
        state[i], action[i], pad[i] = it["observation.state"].numpy(), it["action"].numpy(), it["action_is_pad"].numpy()
        ep[i], frame[i] = int(it["episode_index"]), int(it["frame_index"])
        task[i] = tasks.setdefault(it["task"], len(tasks))
        if with_layout:
            lay[i] = layout(it[CAMS[0]].numpy())
        if i % 2000 == 0:
            print(f"cache {out.name}: {i}/{n}", flush=True)
    imgs.flush()
    for name, arr in dict(state=state, action=action, pad=pad, episode=ep, frame=frame, task=task, layout=lay).items():
        np.save(out / f"{name}.npy", arr)
    meta = {"root": str(root), "frames": n, "episodes": sorted(set(ep.tolist())),
            "tasks": [t for t, _ in sorted(tasks.items(), key=lambda kv: kv[1])], "size": SIZE, "chunk": chunk}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


class Cache:
    def __init__(self, path):
        p = Path(path)
        self.meta = json.loads((p / "meta.json").read_text())
        self.images = np.load(p / "images.npy", mmap_mode="r")
        for name in ("state", "action", "pad", "episode", "frame", "task", "layout"):
            setattr(self, name, np.load(p / f"{name}.npy"))

    def __len__(self):
        return len(self.state)


class Normaliser:
    def __init__(self, stats: dict):
        self.a_min, self.a_max = (torch.tensor(stats["action"][k], dtype=torch.float32) for k in ("min", "max"))
        self.s_mean, self.s_std = (torch.tensor(stats["state"][k], dtype=torch.float32) for k in ("mean", "std"))

    def state(self, s):
        return (s - self.s_mean) / (self.s_std + 1e-6)

    def action(self, a):
        return (2.0 * (a - self.a_min) / (self.a_max - self.a_min + 1e-6) - 1.0).clamp(-1.0, 1.0)

    def unaction(self, a):
        return 0.5 * (a + 1.0) * (self.a_max - self.a_min + 1e-6) + self.a_min


def view(u8: np.ndarray) -> dict:
    """(256, 256, 3) uint8 -> DINOv3 input, as upstream's manual processor (x / 255, ImageNet mean / std)."""
    return {"dinov3": (torch.from_numpy(np.array(u8)).permute(2, 0, 1).float() / 255.0 - MEAN) / STD}


def sample(cache: Cache, i: int, norm: Normaliser, task: str | None = None, state=None):
    views = (view(cache.images[i, 0]), view(cache.images[i, 1]))
    s = torch.from_numpy(cache.state[i] if state is None else state)
    return (views, task or cache.meta["tasks"][cache.task[i]], norm.state(s),
            norm.action(torch.from_numpy(cache.action[i])), torch.from_numpy(~cache.pad[i]).float())


class So101BowlsDataset(IterableDataset):
    """Drop-in for upstream's LiberoRLDSDataset (same constructor keywords; `dataset_dir` = a train cache): uniform
    random frames, forever, seeded per rank and worker with upstream's formula."""

    stats_path: str = ""

    def __init__(self, dataset_dir, rank=0, world_size=1, chunk_size=CHUNK, seed=42, expected_image_size=SIZE, **_):
        cache_chunk = json.loads((Path(dataset_dir) / "meta.json").read_text())["chunk"]
        assert chunk_size == cache_chunk and expected_image_size == SIZE, (chunk_size, cache_chunk, expected_image_size)
        self.path, self.rank, self.seed = dataset_dir, int(rank), int(seed)
        self.norm = Normaliser(json.loads(Path(self.stats_path).read_text()))

    def __iter__(self):
        w = get_worker_info()
        cache = Cache(self.path)
        rng = np.random.default_rng(self.seed + 1009 * self.rank + 9176 * (0 if w is None else w.id))
        while True:
            yield sample(cache, int(rng.integers(len(cache))), self.norm)


def load_libero_init(model, ckpt_path, reinit_prefixes) -> dict:
    """Every checkpoint tensor whose name and shape match; the listed prefixes keep their fresh init. Raises if any
    other tensor is missing or mismatched (no silent partial load)."""
    from turbovla.evaluation.policy import _checkpoint_state_dict, _strip_module_prefix  # upstream's (EMA first)

    src = _strip_module_prefix(_checkpoint_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)))
    tgt = model.state_dict()
    fresh = [k for k in tgt if k.startswith(tuple(reinit_prefixes))]
    bad = [k for k in tgt if k not in fresh and (k not in src or src[k].shape != tgt[k].shape)]
    unused = [k for k in src if k not in tgt]
    if bad or unused:
        raise RuntimeError(f"LIBERO init does not fit outside {reinit_prefixes}: mismatched/missing {bad[:10]}, "
                           f"unused {unused[:10]}")
    merged = {k: (tgt[k] if k in fresh else src[k]) for k in tgt}
    model.load_state_dict(merged, strict=True)
    return {"loaded": len(tgt) - len(fresh), "reinitialised": fresh, "checkpoint_tensors": len(src)}


if __name__ == "__main__":  # LeRobot env: python so101_data.py --root <bowls> --split split.json --which train --out <dir>
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--which", choices=["train", "val"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", help="also write the train-split normalisation statistics here")
    ap.add_argument("--max-episodes", type=int, help="smoke only: first N episodes of the split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=CHUNK, help="action chunk length (H50_BRIEF step 4: 50)")
    a = ap.parse_args()
    split = json.loads(Path(a.split).read_text())
    eps = split[f"{a.which}_episodes"][:a.max_episodes]
    if a.stats_out:
        st = compute_stats(a.root, split["train_episodes"])
        st["dataset"] = f"ravediamond/so101_bowls_all_trimmed@{split['revision']}"
        Path(a.stats_out).write_text(json.dumps(st, indent=1))
    print(build_cache(a.root, eps, a.out, with_layout=a.which == "val", workers=a.workers, chunk=a.chunk))
