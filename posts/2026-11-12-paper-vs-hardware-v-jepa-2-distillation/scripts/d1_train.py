"""WS-110 D1 student (projects/jepa-distill-edge/D1_PROTOCOL.md, frozen 2026-11-07, G1-G4 "D1 defaults").

D1: the W1 encoder (multi-view ViT-Tiny, config lewm_mv_so101.yaml, predictor and SIGReg unused) regresses the frozen V-JEPA 2
ViT-L latent per frame. Target (G1, O3): the 4 x 4 tokens of the 8-frame clip ending at t (extract_vjepa_features.py) averaged to
a pooled 1024-d vector per view, then layer-normed (implementation choice, recorded in notes/WS-110.md before any number). Heads:
one Linear(192, 1024) per view on the fused latent z (training only: never saved with the model). Loss (O4): per view
(1 - cosine) + MSE, 1 : 1, averaged over the two views. Recipe (G2): G0's: AdamW lr 5e-5 wd 1e-3, batch 64, 1 % linear warmup then
cosine, grad clip 1.0, bf16, seed 3072, 81,100 steps (= 100 epochs x 811 steps of G0), 30 Hz frames.

    cache  decode frames of episodes into uint8 memmaps (two views) -> <dir>/{pixels,wrist}.u8 + index.json
    train  regress; val loss every --val-every steps; weights_best.pt (lowest val loss) and weights_final.pt
    rows   per-view projector outputs of every teleop frame (the §30 row of W1, id_latents.w1_views) for best and final
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CAMS = ("pixels", "wrist")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
STEPS, BATCH, LR, WD, SEED, CLIP = 81100, 64, 5e-5, 1e-3, 3072, 1.0


# ------------------------------------------------------------------ pure pieces (tested in test_d1.py)
def pooled_target(tokens: np.ndarray) -> np.ndarray:
    """(T, 16, 1024) teacher tokens (4 x 4 pool, last temporal slot) -> (T, 1024) fp16: mean over the 16 tokens, layer norm."""
    t = torch.from_numpy(tokens).float().mean(1)
    return F.layer_norm(t, t.shape[-1:]).half().numpy()


def d1_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """pred, tgt (B, V, 1024): per view (1 - cos) + MSE (1 : 1, O4), mean over batch and views."""
    pred, tgt = pred.float(), tgt.float()
    cos = 1 - F.cosine_similarity(pred, tgt, dim=-1)
    mse = (pred - tgt).pow(2).mean(-1)
    return (cos + mse).mean()


def lr_at(step: int, total: int) -> float:
    """LinearWarmupCosineAnnealingLR as train_lewm_mv.py: warmup max(1, 1 % of total) steps from 0, cosine to 0."""
    w = max(1, int(0.01 * total))
    if step < w:
        return LR * (step + 1) / w
    return 0.5 * LR * (1 + np.cos(np.pi * (step - w) / max(1, total - w)))


def epoch_batches(n: int, steps: int, seed: int, batch: int = BATCH):
    """Shuffled permutation per epoch over n frames, drop_last: yields `steps` index arrays."""
    rng = np.random.default_rng(seed)
    per, out = n // batch, []
    while len(out) < steps:
        p = rng.permutation(n)
        out += [p[i * batch:(i + 1) * batch] for i in range(per)]
    return out[:steps]


class Cache:
    """uint8 frame memmaps (N, 3, 224, 224) per view + the episode index, in the dataset's global frame order."""

    def __init__(self, root: Path, mode="r"):
        self.root = Path(root)
        self.idx = json.loads((self.root / "index.json").read_text())      # {"n": N, "episodes": {ep: [start, length]}}
        self.n = self.idx["n"]
        self.mm = {v: np.memmap(self.root / f"{v}.u8", dtype=np.uint8, mode=mode, shape=(self.n, 3, 224, 224)) for v in CAMS}
        self.eps = {int(e): tuple(se) for e, se in self.idx["episodes"].items()}

    def frames(self, rows: np.ndarray) -> torch.Tensor:
        """(B, V, 3, 224, 224) uint8 for sorted-or-not row indices."""
        return torch.stack([torch.from_numpy(np.stack([self.mm[v][int(i)] for i in rows])) for v in CAMS], 1)


def to_input(u8: torch.Tensor, dev) -> torch.Tensor:
    """(B, V, 3, H, W) uint8 -> ImageNet-normalised float (the probe.py / FrameSet path)."""
    x = u8.to(dev).float() / 255.0
    return (x - MEAN.to(dev).unsqueeze(0)) / STD.to(dev).unsqueeze(0)


class BatchData(torch.utils.data.Dataset):
    def __init__(self, cache: Cache, batches, targets):
        self.cache, self.batches, self.targets = cache, batches, targets

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        rows = self.batches[i]
        return self.cache.frames(rows), torch.from_numpy(self.targets[rows])


# ------------------------------------------------------------------ model
def build_model(stub: bool = False):
    """The W1 architecture (lewm_mv_so101.yaml model + fuse), exactly as eval_action_sensitivity.load rebuilds it."""
    import hydra  # noqa: F401
    from omegaconf import OmegaConf

    for d in (os.environ.get("WM_CONFIG_DIR"), HERE / "config", HERE.parent / "vla-jepa-world-model" / "wm_train" / "config"):
        if d and (Path(d) / "lewm_mv_so101.yaml").exists():
            cfg = OmegaConf.load(Path(d) / "lewm_mv_so101.yaml")
            break
    else:
        raise FileNotFoundError("lewm_mv_so101.yaml (set WM_CONFIG_DIR)")
    model = hydra.utils.instantiate(cfg.model)
    model.fuse = nn.Linear(cfg.embed_dim, cfg.embed_dim)
    return model, cfg


def encode_stages(model, x: torch.Tensor):
    """x (B, V, 3, H, W) normalised -> (cls, proj, mean, z): probe.encode_stages / train_lewm_mv.encode."""
    b, v = x.shape[:2]
    cls = model.encoder(x.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state[:, 0]
    proj = model.projector(cls).view(b, v, -1)
    mean = proj.mean(1)
    return cls.view(b, v, -1), proj, mean, model.fuse(mean)


def save(model, cfg, path: Path):
    from omegaconf import OmegaConf

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True)}, path)


# ------------------------------------------------------------------ cache
def cmd_cache(a):
    import probe as P
    import g1b_range as G
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    eps = a.episodes or {"all": split["train_episodes"] + split["val_episodes"]}["all"]
    tab = P.frame_table(a.data)
    ds = LeRobotDataset(f"local/{a.data.name}", root=a.data, delta_timestamps={c: [0.0] for c in P.CAMS.values()})
    a.out.mkdir(parents=True, exist_ok=True)
    eps = sorted(eps)
    n = sum(len(tab[e][1]) for e in eps)
    mm = {v: np.memmap(a.out / f"{v}.u8", dtype=np.uint8, mode="w+", shape=(n, 3, 224, 224)) for v in CAMS}
    pos, index, t0 = 0, {}, time.time()
    for e in eps:
        start, act, _ = tab[e]
        loader = torch.utils.data.DataLoader(RawFrames(ds, np.arange(start, start + len(act))), batch_size=64,
                                             num_workers=a.workers, shuffle=False)
        k = pos
        for b in loader:
            for v in CAMS:
                mm[v][k:k + len(b[v])] = b[v].numpy()
            k += len(b["pixels"])
        assert k - pos == len(act), (e, k - pos, len(act))
        index[e] = [pos, len(act)]
        pos = k
        print(f"cache episode {e}: {len(act)} frames, {pos / (time.time() - t0):.0f} frames/s", flush=True)
    for v in CAMS:
        mm[v].flush()
    (a.out / "index.json").write_text(json.dumps({"n": n, "episodes": index}))
    print(f"cache done: {n} frames, {time.time() - t0:.0f} s", flush=True)


class RawFrames(torch.utils.data.Dataset):
    """Frames of the given global indices, both cameras, uint8 (decoded video frames are exact k/255)."""

    def __init__(self, ds, idx):
        import probe as P
        self.ds, self.idx, self.cams = ds, idx, P.CAMS

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        it = self.ds[int(self.idx[i])]
        return {v: (it[c].reshape(3, 224, 224) * 255).round().to(torch.uint8) for v, c in self.cams.items()}


# ------------------------------------------------------------------ train
def load_targets(feats: Path, cache: Cache) -> np.ndarray:
    """(N, V, 1024) fp16 pooled + layer-normed teacher targets in the cache's frame order."""
    tgt = np.zeros((cache.n, len(CAMS), 1024), dtype=np.float16)
    for e, (s, n) in cache.eps.items():
        for j, v in enumerate(CAMS):
            z = np.load(feats / f"ep{e:06d}_{v}.npy")
            assert z.shape == (n, 16, 1024), (e, v, z.shape, n)
            tgt[s:s + n, j] = pooled_target(z)
    return tgt


@torch.no_grad()
def val_loss(model, heads, cache, tgt, rows, dev, bs=128):
    model.eval()
    tot, cs, n = 0.0, 0.0, 0
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            z = encode_stages(model, to_input(cache.frames(r), dev))[3]
            pred = torch.stack([h(z) for h in heads], 1)
        t = torch.from_numpy(tgt[r]).to(dev)
        tot += d1_loss(pred, t).item() * len(r)
        cs += (1 - F.cosine_similarity(pred.float(), t.float(), dim=-1)).mean().item() * len(r)
        n += len(r)
    model.train()
    return tot / n, cs / n


def cmd_train(a):
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cache = Cache(a.cache)
    tgt = load_targets(a.feats, cache)
    tr_eps = [e for e in split["train_episodes"] if e in cache.eps]
    va_eps = [e for e in split["val_episodes"] if e in cache.eps]
    rows = lambda es: np.concatenate([np.arange(cache.eps[e][0], cache.eps[e][0] + cache.eps[e][1]) for e in es])  # noqa: E731
    tr, va = rows(tr_eps), rows(va_eps)[::3]                     # val: every 3rd frame of the 25 val episodes
    print(f"train {len(tr_eps)} episodes {len(tr)} frames; val {len(va_eps)} episodes {len(va)} frames; steps {a.steps}", flush=True)
    model, cfg = build_model()
    model.to(dev).train()
    heads = nn.ModuleList([nn.Linear(cfg.embed_dim, 1024) for _ in CAMS]).to(dev)          # training only
    params = [p for p in list(model.encoder.parameters()) + list(model.projector.parameters()) + list(model.fuse.parameters())
              + list(heads.parameters())]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    batches = [tr[b] for b in epoch_batches(len(tr), a.steps, SEED)]
    loader = torch.utils.data.DataLoader(BatchData(cache, batches, tgt), batch_size=None, shuffle=False,
                                         num_workers=a.workers, prefetch_factor=4 if a.workers else None, pin_memory=True)
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "metrics.jsonl", "a")
    best, best_step, t0, stop = float("inf"), 0, time.time(), None

    def write(rec):
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)

    def check(step, tag):
        nonlocal best, best_step
        vl, vc = val_loss(model, heads, cache, tgt, va, dev)
        rec = {"kind": "val", "tag": tag, "step": step, "val_loss": vl, "val_cos_dist": vc}
        if vl < best:
            best, best_step = vl, step
            save(model, cfg, a.out / "best" / "weights_final.pt")
            rec["is_best"] = True
        write(rec)

    for step, (u8, t) in enumerate(loader):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps)
        x, t = to_input(u8, dev), t.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            z = encode_stages(model, x)[3]
            pred = torch.stack([h(z) for h in heads], 1)
        loss = d1_loss(pred, t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        opt.step()
        if (step + 1) % a.log_every == 0:
            write({"kind": "train", "step": step + 1, "loss": loss.item(), "lr": lr_at(step, a.steps),
                   "elapsed_s": round(time.time() - t0, 1)})
        if (step + 1) % a.val_every == 0 and step + 1 < a.steps:
            check(step + 1, f"step_{step + 1}")
        if a.max_time and time.time() - t0 > a.max_time:
            stop = f"max-time {a.max_time}s at step {step + 1} of {a.steps}"
            break
    done = step + 1
    check(done, "final")
    save(model, cfg, a.out / "final" / "weights_final.pt")
    write({"kind": "end", "steps_done": done, "steps_planned": a.steps, "stop_reason": stop, "best_step": best_step,
           "best_val_loss": best, "elapsed_s": round(time.time() - t0, 1)})


# ------------------------------------------------------------------ rows
def load_student(d: Path, dev):
    from omegaconf import OmegaConf
    import hydra

    ck = torch.load(d / "weights_final.pt", map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["config"])
    m = hydra.utils.instantiate(cfg.model)
    m.fuse = nn.Linear(cfg.embed_dim, cfg.embed_dim)
    m.load_state_dict(ck["state_dict"])                                # strict: the W1 architecture
    return m.to(dev).eval(), ck["state_dict"]


@torch.no_grad()
def cmd_rows(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    cache = Cache(a.cache)
    ref = torch.load(a.w1 / "weights_final.pt", map_location="cpu", weights_only=False)["state_dict"] if a.w1 else None
    check = {}
    for tag in a.tags:
        m, sd = load_student(a.students / tag, dev)
        if ref is not None:
            assert set(sd) == set(ref) and all(sd[k].shape == ref[k].shape for k in ref), "student keys differ from W1's"
            check[tag] = {"tensors": len(sd), "same_keys_and_shapes_as_W1": True}
        for name, eps in (("teleop_train", split["train_episodes"]), ("teleop_val", split["val_episodes"])):
            eps = [e for e in eps if e in cache.eps]
            meta = np.load(a.id30 / name / "meta.npz") if a.id30 else None
            arr = {}
            for e in eps:
                s, n = cache.eps[e]
                if meta is not None:
                    assert len(meta[f"ep{e:03d}_frame"]) == n, (name, e, n)
                zs = []
                for i in range(s, s + n, 128):
                    r = np.arange(i, min(i + 128, s + n))
                    proj = encode_stages(m, to_input(cache.frames(r), dev))[1]            # (b, 2, 192), fp32 eval as id_latents
                    zs.append(proj.float().cpu())
                z = torch.cat(zs).half().numpy()
                arr[f"ep{e:03d}_top"], arr[f"ep{e:03d}_wrist"] = z[:, 0], z[:, 1]
            (a.out / name).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.out / name / f"{a.prefix}_{tag}.npz", **arr)
            print(f"rows {tag} {name}: {len(eps)} episodes", flush=True)
    (a.out / "arch_check.json").write_text(json.dumps(check, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("--data", type=Path, required=True); c.add_argument("--out", type=Path, required=True)
    c.add_argument("--episodes", type=int, nargs="*"); c.add_argument("--workers", type=int, default=4)
    t = sub.add_parser("train"); t.add_argument("--cache", type=Path, required=True); t.add_argument("--feats", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True); t.add_argument("--steps", type=int, default=STEPS)
    t.add_argument("--val-every", type=int, default=4055); t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--workers", type=int, default=6); t.add_argument("--max-time", type=int, default=0)
    r = sub.add_parser("rows"); r.add_argument("--cache", type=Path, required=True); r.add_argument("--students", type=Path, required=True)
    r.add_argument("--tags", nargs="+", default=["best", "final"]); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--w1", type=Path); r.add_argument("--id30", type=Path); r.add_argument("--prefix", default="D1")
    a = ap.parse_args()
    {"cache": cmd_cache, "train": cmd_train, "rows": cmd_rows}[a.cmd](a)


if __name__ == "__main__":
    main()
