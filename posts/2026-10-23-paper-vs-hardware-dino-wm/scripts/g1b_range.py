"""G1b plannable range (G1_PROTOCOL.md): does the world model rank the true action chunk first as the target moves away?

For anchors t on the held-out val episodes: context = real frames t-2..t; 8 chunks (the true one + 7 distractors: 4
cross-episode, 2 time-shifted +-15, 1 held first action); each rolled out 30 steps; score = L2 distance between the
imagined latent after H steps and the encoded real frame at t+d. One 30-step rollout gives every H as a prefix (the
predictor is causal), so H in {5, 10, 30} and H = d come from the same pass.

    WM_SPLIT=projects/vla-jepa-world-model/wm_train/split.json python projects/vla-jepa-world-model/wm_train/g1b_range.py \\
        <weights dir> --data <transcoded snapshot> [--out results/<date>_g1b_range_w1.json]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_action_sensitivity as E  # noqa: E402
import probe as P  # noqa: E402

C = 3
DS = (5, 10, 15, 20, 30)
HS = (5, 10, 30, "d")
SHIFT = 15
HMAX = 30
STRIDE = 10
N_CROSS = 4
THRESH = 0.8
N_BOOT = 2000


class FrameSet(torch.utils.data.Dataset):
    """All frames of the given global indices, both cameras, ImageNet-normalized (same path as probe.Items)."""

    def __init__(self, ds, idx):
        self.ds, self.idx = ds, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        it = self.ds[int(self.idx[i])]
        return {v: ((it[cam] - P.MEAN) / P.STD).reshape(3, *it[cam].shape[-2:]) for v, cam in P.CAMS.items()}


class W1:
    """LeWM 1-vector: embed frames -> z (N, 192); roll out HMAX steps from a context with normalized actions."""

    name = "W1"

    def __init__(self, weights: Path):
        self.model, self.cfg = E.load(weights, P.DEV)

    @torch.no_grad()
    def embed(self, frames: dict) -> torch.Tensor:
        return P.encode_stages(self.model, frames)[3].float().cpu()

    @torch.no_grad()
    def rollout(self, z_ctx: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """z_ctx (N, C, D), a_norm (N, C + HMAX - 1, 6) -> imagined z_{t+1..t+HMAX} (N, HMAX, D)."""
        a = self.model.action_encoder(a_norm.to(P.DEV))
        return E.rollout(self.model, z_ctx.to(P.DEV), a, C, HMAX).float().cpu()


def embed_episodes(cand, data: Path, tab: dict, eps: list[int], workers: int) -> dict[int, torch.Tensor]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"local/{data.name}", root=data, delta_timestamps={cam: [0.0] for cam in P.CAMS.values()})
    out = {}
    for ep in eps:
        start, act, _ = tab[ep]
        loader = torch.utils.data.DataLoader(FrameSet(ds, np.arange(start, start + len(act))), batch_size=64,
                                             num_workers=workers, shuffle=False)
        out[ep] = torch.cat([cand.embed({v: b[v] for v in P.CAMS}) for b in loader])
        print(f"embedded episode {ep}: {len(act)} frames", flush=True)
    return out


def build_chunks(a_norm: dict, eps: list[int], rng: np.random.Generator):
    """Anchors (ep, t) and their 8 action chunks of length C + HMAX - 1, chunk 0 = true."""
    n = C + HMAX - 1
    anchors, chunks = [], []
    for ep in eps:
        a = a_norm[ep]
        T = len(a)
        others = [e for e in eps if e != ep and len(a_norm[e]) >= n]
        for t in range(SHIFT + C - 1, T - 1 - (HMAX + SHIFT - 1) + 1, STRIDE):  # t >= 17, t + 44 <= T - 1
            s0 = t - (C - 1)
            true = a[s0:s0 + n]
            cross = []
            for _ in range(N_CROSS):
                e2 = others[rng.integers(len(others))]
                st = rng.integers(0, len(a_norm[e2]) - n + 1)
                cross.append(a_norm[e2][st:st + n])
            shifted = [a[s0 - SHIFT:s0 - SHIFT + n], a[s0 + SHIFT:s0 + SHIFT + n]]
            held = np.repeat(true[:1], n, axis=0)
            anchors.append((ep, t))
            chunks.append(np.stack([true, *cross, *shifted, held]))
    return anchors, np.stack(chunks).astype(np.float32)


def rollout_stream(cand, Z: dict, anchors, chunks, batch: int = 16, extra: dict | None = None):
    """Roll out every chunk of every anchor (streaming: a batch of anchors at a time, nothing kept but small scores).
    Returns dist[(d, h)] (N, K) latent L2 distances of the last imagined latent to z_{t+d} for H = h (h = 'd': H = d), and
    err (N, K, HMAX): per-step MSE of the imagined latents vs the real z_{t+1..t+HMAX}; copy (N, HMAX): copy-last MSE."""
    N, K = chunks.shape[:2]
    dist = {(d, h): torch.zeros(N, K) for d in DS for h in HS}
    err, copy = torch.zeros(N, K, HMAX), torch.zeros(N, HMAX)
    if extra is not None:   # separation of every chunk from the true one: imagined end state (latent L2) and action chunk (RMS, normalized units)
        extra['sep'] = {(d, h): torch.zeros(N, K) for d in DS for h in HS}
        extra['act_sep'] = {(d, h): torch.zeros(N, K) for d in DS for h in HS}
    for i in range(0, N, batch):
        sl = slice(i, min(i + batch, N))
        an = anchors[sl]
        ctx = torch.stack([Z[ep][t - C + 1:t + 1] for ep, t in an]).float()                   # (b, C, *shape)
        b = len(an)
        a = torch.as_tensor(chunks[sl]).flatten(0, 1)
        rolled = cand.rollout(ctx.repeat_interleave(K, 0), a).view(b, K, HMAX, *ctx.shape[2:]).float()
        fut = torch.stack([Z[ep][t + 1:t + 1 + HMAX] for ep, t in an]).float()               # (b, HMAX, *shape)
        err[sl] = (rolled - fut[:, None]).flatten(3).pow(2).mean(-1).cpu()
        copy[sl] = (ctx[:, -1:] - fut).flatten(2).pow(2).mean(-1).cpu()
        for d in DS:
            target = fut[:, d - 1]
            for h in HS:
                hh = d if h == "d" else h
                dist[(d, h)][sl] = (rolled[:, :, hh - 1] - target[:, None]).flatten(2).norm(dim=-1).cpu()
                if extra is not None:
                    end = rolled[:, :, hh - 1].flatten(2)
                    extra['sep'][(d, h)][sl] = (end - end[:, :1]).norm(dim=-1).cpu()
                    ch = torch.as_tensor(chunks[sl])[:, :, :C + hh - 1]
                    extra['act_sep'][(d, h)][sl] = (ch - ch[:, :1]).pow(2).mean(dim=(2, 3)).sqrt()
    return dist, err, copy


def cells_from(dist: dict, anchors, k_rank: int = 8) -> dict:
    """Ranking accuracy / mean rank / episode-bootstrap CI per (d, H) from the first k_rank chunks (chunk 0 = true)."""
    eps = np.array([ep for ep, _ in anchors])
    uniq = np.unique(eps)
    cells = {}
    for (d, h), s in dist.items():
        s = s[:, :k_rank]
        rank = 1 + (s[:, 1:] <= s[:, :1]).sum(1)                                              # ties count against the true chunk
        hit = (rank == 1).numpy().astype(float)
        rk = rank.numpy().astype(float)
        per_ep = np.stack([[hit[eps == u].sum(), rk[eps == u].sum(), (eps == u).sum()] for u in uniq])
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(N_BOOT):
            x = per_ep[rng.integers(len(uniq), size=len(uniq))].sum(0)
            boots.append(x[0] / x[2])
        cells[f"d{d}_H{h}"] = {"d": d, "H": d if h == "d" else h, "H_rule": "d" if h == "d" else "fixed",
                               "accuracy": round(float(hit.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                               round(float(np.percentile(boots, 97.5)), 4)], "mean_rank": round(float(rk.mean()), 3)}
    return cells


def evaluate(cand, Z: dict, anchors, chunks) -> dict:
    dist, _, _ = rollout_stream(cand, Z, anchors, chunks)
    return cells_from(dist, anchors)


DECOY_TYPES = {"cross_episode": [1, 2, 3, 4], "shift_-15": [5], "shift_+15": [6], "held_first": [7]}


def separation_report(dist: dict, extra: dict, n_bins: int = 5, h="d") -> dict:
    """Owner's correction 3: accuracy against how far each decoy is from the true chunk. For every d (H = h): pairwise win rate of the true chunk
    against a decoy, binned by (a) the distance of the decoy's imagined end state from the true one, (b) the RMS distance of the decoy's action
    chunk from the true one, and per decoy type. Low accuracy at small separations = an unfair test (near-identical states), not a weak model."""
    out = {}
    for d in DS:
        s, sep, asep = dist[(d, h)][:, :8], extra["sep"][(d, h)][:, :8], extra["act_sep"][(d, h)][:, :8]
        win = (s[:, :1] < s[:, 1:]).numpy()                                         # (N, 7): true closer to the target than the decoy
        row = {"pairwise_accuracy": round(float(win.mean()), 4)}
        for name, x in (("by_state_separation", sep[:, 1:].numpy()), ("by_action_separation", asep[:, 1:].numpy())):
            edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
            idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
            row[name] = [{"bin": b, "pairs": int((idx == b).sum()), "mean_separation": round(float(x[idx == b].mean()), 4),
                          "win_rate": round(float(win[idx == b].mean()), 4)} for b in range(n_bins)]
        row["by_decoy_type"] = {n: {"win_rate": round(float(win[:, [i - 1 for i in ix]].mean()), 4),
                                    "mean_state_separation": round(float(sep[:, ix].mean()), 4),
                                    "mean_action_separation": round(float(asep[:, ix].mean()), 4)} for n, ix in DECOY_TYPES.items()}
        out[f"d{d}"] = row
    return out


def plannable_range(cells: dict, h) -> dict:
    ok = [d for d in DS if cells[f"d{d}_H{h}"]["accuracy"] >= THRESH]
    contiguous = 0
    for d in DS:
        if cells[f"d{d}_H{h}"]["accuracy"] >= THRESH:
            contiguous = d
        else:
            break
    return {"largest_d_meeting": max(ok) if ok else None, "largest_d_contiguous": contiguous or None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-episodes", type=int, help="smoke test: first N val episodes only (NOT a protocol run)")
    args = ap.parse_args()
    torch.manual_seed(0)
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    eps = split["val_episodes"][:args.max_episodes]
    t0 = time.time()
    cand = W1(args.weights)
    a_mean, a_std = P.action_stats(args.data, split["train_episodes"])
    tab = P.frame_table(args.data)
    a_norm = {ep: ((tab[ep][1] - a_mean) / a_std).astype(np.float32) for ep in eps}
    Z = embed_episodes(cand, args.data, tab, eps, args.workers)
    t1 = time.time()
    anchors, chunks = build_chunks(a_norm, eps, np.random.default_rng(0))
    print(f"{len(anchors)} anchors x {chunks.shape[1]} chunks, {len(eps)} val episodes", flush=True)
    cells = evaluate(cand, Z, anchors, chunks)
    res = {"candidate": cand.name, "protocol": "wm_train/G1_PROTOCOL.md", "revision": split["revision"],
           "val_episodes": len(eps), "anchors": len(anchors), "chance_accuracy": round(1 / chunks.shape[1], 4),
           "smoke": args.max_episodes is not None, "threshold": THRESH, "cells": cells,
           "range": {f"H={h}": plannable_range(cells, h) for h in HS},
           "seconds": {"embed": round(t1 - t0, 1), "rank": round(time.time() - t1, 1)}}
    print(f"\nranking accuracy of the true chunk (chance {res['chance_accuracy']}):")
    print("  d    " + "  ".join(f"H={h!s:>3}" for h in HS))
    for d in DS:
        print(f"  {d:>2}   " + "  ".join(f"{cells[f'd{d}_H{h}']['accuracy']:.3f}" for h in HS))
    print("range:", json.dumps(res["range"]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
