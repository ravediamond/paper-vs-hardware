"""WS-167: train La-JEPA S on the stored V-JEPA 2.1 features and evaluate held-out first-action MAE (A1 / A2 inputs).

  python train.py --data DIR --store DIR --out RUN [--fut-weight 1.0|0.0] [--seed 0] [--steps 20000] [--bs 32]
  python train.py --data DIR --store DIR --out RUN --eval-only          (reload RUN/model.pt, re-run the held-out evaluation)

Metric (PROTOCOL section 3): for every frame of the 10 held-out episodes the policy predicts a chunk, first action vs the recorded action,
mean absolute error in raw dataset units (joint degrees, gripper in dataset units), mean over the 6 dims and over frames; per-episode
mean first (heldout_mae.json). The identity null (first action = recorded state at that frame) is written next to it.
--fut-weight 0 is the A2 ablation: same code, same seed, same data order; only the future loss weight differs.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lajepa import Cfg, LaJEPA, n_params

SPLITS = Path(__file__).resolve().parent.parent / "gawm" / "splits"
CAMS = ("top", "wrist")


def split(name):
    return json.load(open(SPLITS / f"{name}.json"))["episodes"]


class Store:
    def __init__(self, data, store):
        self.meta = json.load(open(f"{data}/episodes.json"))
        z = np.load(f"{data}/arrays.npz")
        self.action, self.state, self.ep_of = z["action"], z["state"], z["ep_of_frame"]
        self.eps = self.meta["episodes"]
        self.by_orig = {m["orig"]: k for k, m in enumerate(self.eps)}
        self.obs = [np.load(f"{store}/obs_{c}.npy", mmap_mode="r") for c in CAMS]
        self.fut = [np.load(f"{store}/fut_{c}.npy", mmap_mode="r") for c in CAMS]
        self.anchor = np.load(f"{store}/anchor.npy")
        self.fut_ok = np.load(f"{store}/fut_ok.npy")
        self.arow = np.full(len(self.action), -1, np.int64)
        self.arow[self.anchor] = np.arange(len(self.anchor))
        nz = np.load(f"{store}/norm.npz")
        self.o_mu, self.o_sd = nz["obs_mean"], nz["obs_std"]
        self.f_mu, self.f_sd = nz["fut_mean"], nz["fut_std"]
        tz = np.load(f"{store}/text.npz")
        texts = [str(t) for t in tz["texts"]]
        self.t_emb, self.t_mask = tz["emb"], tz["mask"]
        self.ep_text = np.array([texts.index(m["task"]) for m in self.eps])

    def frames_of(self, orig):
        ks = [self.by_orig[e] for e in orig if e in self.by_orig]
        return np.concatenate([self.eps[k]["start"] + np.arange(self.eps[k]["length"]) for k in ks]) if ks else np.zeros(0, np.int64)

    def pct_norm(self, train_orig):
        idx = self.frames_of(train_orig)
        a, s = self.action[idx], self.state[idx]
        self.a_lo, self.a_hi = np.percentile(a, 1, axis=0), np.percentile(a, 99, axis=0)
        self.s_lo, self.s_hi = np.percentile(s, 1, axis=0), np.percentile(s, 99, axis=0)
        self.a_hi = np.maximum(self.a_hi, self.a_lo + 1e-6); self.s_hi = np.maximum(self.s_hi, self.s_lo + 1e-6)

    def norm_a(self, a): return 2 * (a - self.a_lo) / (self.a_hi - self.a_lo) - 1
    def denorm_a(self, a): return (a + 1) / 2 * (self.a_hi - self.a_lo) + self.a_lo
    def norm_s(self, s): return 2 * (s - self.s_lo) / (self.s_hi - self.s_lo) - 1

    def obs_of(self, g):
        return np.stack([(np.asarray(o[g], np.float32) - self.o_mu) / self.o_sd for o in self.obs])      # (n_cam, 64, 1024)

    def stats_dict(self):
        return {k: getattr(self, k).tolist() for k in ("a_lo", "a_hi", "s_lo", "s_hi")}


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, st, frames, H=32):
        self.st, self.frames, self.H = st, frames, H

    def __len__(self): return len(self.frames)

    def __getitem__(self, i):
        st, H = self.st, self.H
        g = int(self.frames[i])
        k = int(st.ep_of[g]); m = st.eps[k]
        s0, n = m["start"], m["length"]; t = g - s0
        r = int(st.arow[g])
        fut = np.stack([(np.asarray(f[r], np.float32) - st.f_mu) / st.f_sd for f in st.fut])             # (n_cam, 128, 1024)
        idx = np.arange(t, t + H)
        return {
            "obs": torch.from_numpy(st.obs_of(g)), "fut": torch.from_numpy(fut),
            "fut_ok": torch.from_numpy(np.repeat(st.fut_ok[r], 16))[None].expand(2, -1).clone(),
            "action": torch.from_numpy(st.norm_a(st.action[s0 + np.minimum(idx, n - 1)]).astype(np.float32)),
            "act_ok": torch.from_numpy(idx < n),
            "state": torch.from_numpy(st.norm_s(st.state[g]).astype(np.float32)),
            "text": torch.from_numpy(st.t_emb[st.ep_text[k]].astype(np.float32)), "text_mask": torch.from_numpy(st.t_mask[st.ep_text[k]]),
        }


def lr_at(step, total, peak, warm=0.05, tail=0.10, floor=0.01):
    """Table 9: 5% warmup, cosine decay, final 10% constant tail, floor 1% of peak (scaled from 150k to `total` updates)."""
    w, c_end = int(warm * total), int((1 - tail) * total)
    if step < w: return peak * (step + 1) / w
    if step >= c_end: return peak * floor
    p = (step - w) / max(1, c_end - w)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model, st, dev, orig_eps, steps=3, bs=128, seed=0, limit=None):
    model.eval()
    frames = st.frames_of(orig_eps)
    if limit: frames = frames[:limit]
    gen = torch.Generator(device=dev).manual_seed(seed)
    rows = []
    for lo in range(0, len(frames), bs):
        g = frames[lo:lo + bs]
        k = st.ep_of[g]
        obs = torch.from_numpy(np.stack([st.obs_of(int(x)) for x in g])).to(dev)
        state = torch.from_numpy(st.norm_s(st.state[g]).astype(np.float32)).to(dev)
        te = torch.from_numpy(st.t_emb[st.ep_text[k]].astype(np.float32)).to(dev)
        tm = torch.from_numpy(st.t_mask[st.ep_text[k]]).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            a = model.sample(obs, state, te, tm, steps=steps, generator=gen)
        a0 = st.denorm_a(a[:, 0].float().cpu().numpy())
        tgt = st.action[g]
        for j in range(len(g)):
            rows.append((int(st.eps[k[j]]["orig"]), np.abs(a0[j] - tgt[j]), np.abs(st.state[g[j]] - tgt[j])))
    model.train()
    return rows


def per_episode(rows):
    by = {}
    for e, l1, nl in rows: by.setdefault(e, []).append((l1, nl))
    out = {}
    for e, v in sorted(by.items()):
        l1 = np.stack([x[0] for x in v]); nl = np.stack([x[1] for x in v])
        out[str(e)] = {"mae": float(l1.mean()), "identity_null": float(nl.mean()), "frames": len(v), "mae_per_dim": l1.mean(0).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--store", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--train-split", default="train172"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--fut-weight", type=float, default=1.0); ap.add_argument("--eval-steps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-limit", type=int, default=None); ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--tiny", action="store_true", help="CPU smoke only: width 96, depth 2 (never a result)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    st = Store(a.data, a.store)
    st.pct_norm(split("train172"))      # normalisation from the 172 train episodes for every run
    mk = lambda: Cfg(fut_weight=a.fut_weight, **({"width": 96, "depth": 2, "heads": 2} if a.tiny else {}))
    model = LaJEPA(mk()).to(dev)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ema = LaJEPA(mk()).to(dev).eval()
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)
    print("params", n_params(model), "fut_weight", a.fut_weight, flush=True)
    if not a.eval_only:
        decay_p = [p for p in model.parameters() if p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.ndim < 2]
        opt = torch.optim.AdamW([{"params": decay_p, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
        tr_orig = [e for e in split(a.train_split) if e in st.by_orig]
        frames = np.array([g for g in st.frames_of(tr_orig) if st.arow[g] >= 0])
        ds = TrainDS(st, frames)
        sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=a.steps * a.bs, generator=torch.Generator().manual_seed(a.seed))
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, sampler=sampler, num_workers=a.workers, drop_last=True,
                                         persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
        log = open(out / "train_log.jsonl", "a")
        gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
        t0 = time.time()
        model.train()
        for step, b in enumerate(dl, 1):
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            for g_ in opt.param_groups: g_["lr"] = lr_at(step - 1, a.steps, a.lr)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = model(b, generator=gen)
            opt.zero_grad(set_to_none=True)
            o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            d = min(a.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            if step % a.log_every == 0 or step == 1:
                rec = {"step": step, "loss": float(o["loss"]), "action": float(o["action_loss"]), "future": float(o["future_loss"]),
                       "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "cfg": vars(model.c), "norm": st.stats_dict()}, out / "model.pt")
        json.dump({"args": vars(a), "params": n_params(model), "norm": st.stats_dict()}, open(out / "run.json", "w"))
    else:
        ck = torch.load(out / "model.pt", map_location=dev)
        ema.load_state_dict(ck["ema"])
    rows = evaluate(ema, st, dev, split("heldout10"), steps=a.eval_steps, seed=a.seed, limit=a.eval_limit)
    pe = per_episode(rows)
    res = {"per_episode": pe, "mean_mae": float(np.mean([v["mae"] for v in pe.values()])),
           "mean_identity_null": float(np.mean([v["identity_null"] for v in pe.values()])), "eval_steps": a.eval_steps, "weights": "ema",
           "units": "raw dataset units (joint degrees; gripper in dataset units)"}
    json.dump(res, open(out / "heldout_mae.json", "w"), indent=1)
    print("heldout mean MAE", res["mean_mae"], "identity null", res["mean_identity_null"], flush=True)


if __name__ == "__main__":
    main()
