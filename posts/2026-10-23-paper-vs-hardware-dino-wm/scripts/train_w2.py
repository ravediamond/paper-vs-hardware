"""W2: world-model predictor on frozen DINOv2 ViT-S/14 patch tokens (G1_PROTOCOL.md). Encoder frozen and precomputed
(extract_dino_features.py), so only the predictor trains; no SIGReg (nothing can collapse).

Same setting as the W1 LeWM recipe: 30 Hz, context C = 3, L = 5 teacher-forced targets per window, one joint action per
frame (action_i is the command leaving frame i), AdamW 5e-5 / wd 1e-3, cosine with 1 % warmup, bf16, grad clip 1.0,
same split.json. Differences (all by construction of token latents, recorded):
- Predictor: 4 AdaLN-zero blocks (width 384, 6 heads x 64, MLP 1536) over the tokens of both views (V x P tokens per frame),
  applied to a window of exactly C = 3 frames: causal across the 3 frames, full attention inside a frame, a learned time
  embedding per position in the window (as LeWM's predictor). Training takes the L = 5 windows of an 8-frame clip as separate
  sequences (LeWM's teacher forcing) and the loss is on the last position of each; the rollout uses the same windows.
- Action conditioning: per-frame action -> 2-layer MLP -> AdaLN (shift, scale, gate) of every token of that frame.
- Residual output: next tokens = current tokens + a zero-initialized predicted change (starts as copy-last; without it a
  toy run learned almost nothing in 200 steps and ignored the actions). LeWM predicts the next latent directly.
- Loss: MSE between predicted and true next-frame tokens after a parameter-free LayerNorm per token (feature scale).
- Own implementation; `facebookresearch/jepa-wms` (CC-BY-NC 4.0) was only a design reference, no code copied.

    WM_SPLIT=wm_train/split.json python train_w2.py --features <dir> --data <snapshot> --out <dir> [--steps N] [--max-time S]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

C, L, ACT = 3, 5, 6
VIEWS = ("pixels", "wrist")


class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp: int):
        super().__init__()
        self.h, self.n1, self.n2 = heads, nn.LayerNorm(d, elementwise_affine=False), nn.LayerNorm(d, elementwise_affine=False)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight), nn.init.zeros_(self.mod.bias)       # AdaLN-zero: block = identity at init

    def forward(self, x, cond, bias):
        b, f, t, d = x.shape
        s1, c1, g1, s2, c2, g2 = (m[:, :, None] for m in self.mod(F.silu(cond)).chunk(6, -1))
        h = (self.n1(x) * (1 + c1) + s1).view(b, f * t, d)
        q, k, v = self.qkv(h).view(b, f * t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + g1 * self.proj(a.transpose(1, 2).reshape(b, f * t, d)).view(b, f, t, d)
        return x + g2 * self.mlp(self.n2(x) * (1 + c2) + s2)


class W2Predictor(nn.Module):
    def __init__(self, tokens: int, feat: int = 384, d: int = 384, depth: int = 4, heads: int = 6, mlp: int = 1536):
        super().__init__()
        self.tokens, self.heads, self.ckpt = tokens, heads, False   # ckpt: activation checkpointing per block (--checkpoint)
        self.inp, self.pos = nn.Linear(feat, d), nn.Parameter(torch.randn(tokens, d) * 0.02)
        self.act = nn.Sequential(nn.Linear(ACT, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(Block(d, heads, mlp) for _ in range(depth))
        self.time = nn.Parameter(torch.randn(C, d) * 0.02)                   # position inside the C-frame window
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, feat))
        nn.init.zeros_(self.out[1].weight), nn.init.zeros_(self.out[1].bias)  # residual head: starts as copy-last

    def mask_bias(self, f: int, dev, dtype=None):
        """Block-causal across frames, full inside a frame: boolean mask (N, N), True = may attend (a float -inf mask gave
        NaN gradients on MPS)."""
        fi = torch.arange(f, device=dev).repeat_interleave(self.tokens)
        return fi[None, :] <= fi[:, None]

    def forward(self, tok, act):
        """tok (B, F<=C, T, 384) layer-normed tokens (views concatenated), act (B, F, 6) normalized -> (B, F, T, 384): the
        output at position i predicts frame i+1 (use the last position)."""
        b, f, t, _ = tok.shape
        x = self.inp(tok) + self.pos + self.time[:f, None]
        cond = self.act(act)
        bias = self.mask_bias(f, tok.device, x.dtype)
        for blk in self.blocks:
            x = torch.utils.checkpoint.checkpoint(blk, x, cond, bias, use_reentrant=False) if self.ckpt and self.training else blk(x, cond, bias)
        return tok.float() + self.out(x)                                    # residual: predicted change of the tokens

    @torch.no_grad()
    def rollout(self, ctx, act, k: int):
        """ctx (B, C, T, 384), act (B, C + k - 1, 6): autoregressive k steps, each from the last C frames. -> (B, k, T, 384)."""
        frames, outs = ctx, []
        for i in range(k):
            nxt = self(frames[:, -C:], act[:, i:i + C])[:, -1:]
            outs.append(nxt)
            frames = torch.cat([frames, nxt], 1)
        return torch.cat(outs, 1)


def ln(x):
    return F.layer_norm(x.float(), x.shape[-1:])


class Windows(torch.utils.data.Dataset):
    def __init__(self, feat_dir: Path, tab: dict, eps: list, a_mean, a_std):
        self.feat = {ep: [np.load(feat_dir / f"ep{ep:06d}_{v}.npy", mmap_mode="r") for v in VIEWS] for ep in eps}
        self.act = {ep: ((tab[ep][1] - a_mean) / a_std).astype(np.float32) for ep in eps}
        self.idx = [(ep, s) for ep in eps for s in range(len(self.act[ep]) - (C + L) + 1)]
        self.tokens = sum(f.shape[1] for f in self.feat[eps[0]])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        ep, s = self.idx[i]
        tok = np.concatenate([np.asarray(f[s:s + C + L]) for f in self.feat[ep]], axis=1)   # (8, T, 384) fp16
        return torch.from_numpy(tok), torch.from_numpy(self.act[ep][s:s + C + L])


def predict_windows(model, x, act):
    """x (B, C + L, T, 384) layer-normed, act (B, C + L, 6): the L windows of C frames as separate sequences, prediction of
    frame j + C from frames j..j+C-1 (last position). -> (B, L, T, 384)."""
    b = x.shape[0]
    win = torch.stack([x[:, j:j + C] for j in range(L)], 1).flatten(0, 1)
    aw = torch.stack([act[:, j:j + C] for j in range(L)], 1).flatten(0, 1)
    return model(win, aw)[:, -1].view(b, L, *x.shape[2:])


def losses(model, tok, act, ac):
    """Teacher-forced loss on the L targets, copy-last baseline, and the same with actions shuffled across the batch."""
    x = ln(tok)
    with ac:
        pred = predict_windows(model, x, act).float()
        pred_s = predict_windows(model, x, act[torch.randperm(act.shape[0], device=act.device)]).float()
    tgt = x[:, C:C + L]
    prev = x[:, C - 1:C - 1 + L]
    return F.mse_loss(pred, tgt), F.mse_loss(prev, tgt), F.mse_loss(pred_s, tgt)


@torch.no_grad()
def checks(model, val_loader, dev, ac, k=30, batches=20):
    model.eval()
    tf, cp, sh = [], [], []
    for i, (tok, act) in enumerate(val_loader):
        if i >= batches:
            break
        a, b, c = losses(model, tok.to(dev), act.to(dev), ac)
        tf.append(a.item()), cp.append(b.item()), sh.append(c.item())
    res = {"tf_mse": float(np.mean(tf)), "copy_last_mse": float(np.mean(cp)), "shuffled_action_mse": float(np.mean(sh))}
    res["pred_over_copy"] = res["tf_mse"] / res["copy_last_mse"]
    res["shuffle_ratio"] = res["shuffled_action_mse"] / res["tf_mse"]
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="transcoded snapshot (actions, split)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=81000)
    ap.add_argument("--max-time", type=float, help="seconds; training stops and saves when reached")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check-every", type=int, default=2000)
    ap.add_argument("--stop-at-step", type=int, help="stop here but keep the schedule of --steps (re-run of the first N steps)")
    ap.add_argument("--save-all", action="store_true", help="keep a copy of the checkpoint at every check in <out>/ckpts/")
    ap.add_argument("--checkpoint", action="store_true", help="activation checkpointing per block (less memory, ~1.3x time)")
    ap.add_argument("--device", help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--limit-episodes", type=int, help="smoke on a few episodes (NOT a protocol run)")
    ap.add_argument("--seed", type=int, default=3072, help="3072 = W2 (G1); W2' (2026-10-23) uses 3073")
    args = ap.parse_args()
    import probe as P  # frame_table, action_stats (pandas only)

    torch.manual_seed(args.seed)
    dev = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ac = torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    tr_eps, va_eps = split["train_episodes"], split["val_episodes"]
    if args.limit_episodes:
        tr_eps, va_eps = tr_eps[:args.limit_episodes], va_eps[:max(1, args.limit_episodes // 4)]
    tab = P.frame_table(args.data)
    a_mean, a_std = P.action_stats(args.data, split["train_episodes"])
    train, val = (Windows(args.features, tab, e, a_mean, a_std) for e in (tr_eps, va_eps))
    tl = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                                     pin_memory=dev == "cuda", persistent_workers=args.workers > 0,
                                     prefetch_factor=2 if args.workers else None)
    vl = torch.utils.data.DataLoader(val, batch_size=args.batch, shuffle=True, num_workers=min(2, args.workers))
    model = W2Predictor(train.tokens).to(dev)
    model.ckpt = args.checkpoint
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    warm = max(1, int(0.01 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = {"args": {k: str(v) for k, v in vars(args).items()}, "tokens": train.tokens, "params": sum(p.numel() for p in model.parameters()),
           "train_windows": len(train), "val_windows": len(val), "revision": split["revision"]}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"W2: {cfg['params'] / 1e6:.1f} M params, {train.tokens} tokens/frame, {len(train)} train / {len(val)} val windows, {dev}", flush=True)

    def write(rec):
        with open(args.out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def save(name):
        (args.out / name).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg}, args.out / name)

    step, t0, t_last, waits, steps_t, done = 0, time.time(), time.time(), [], [], False
    best = {"ratio": float("inf"), "step": 0}
    first = None
    while not done:
        for tok, act in tl:
            t_get = time.time()
            waits.append(t_get - t_last)
            tok, act = tok.to(dev, non_blocking=True), act.to(dev, non_blocking=True)
            loss, cp, _ = losses_train(model, tok, act, ac)
            if not torch.isfinite(loss):
                import copy
                cpu_loss = losses_train(copy.deepcopy(model).cpu(), tok.cpu(), act.cpu(), torch.autocast("cpu", enabled=False))[0].item()
                print(f"same batch on CPU: loss {cpu_loss}", flush=True)
                sys.exit(f"non-finite loss at step {step + 1}: tok finite={torch.isfinite(tok.float()).all().item()} "
                         f"act finite={torch.isfinite(act).all().item()} lr={sched.get_last_lr()[0]:.2e} "
                         f"param finite={all(torch.isfinite(p).all().item() for p in model.parameters())}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            step += 1
            t_last = time.time()
            steps_t.append(t_last - t_get)
            if step % 20 == 0:
                s, w = float(np.mean(steps_t)), float(np.mean(waits))
                rec = {"kind": "train", "step": step, "elapsed_s": round(t_last - t0, 1), "gpu_s_per_step": round(s, 4),
                       "data_wait_s_per_step": round(w, 4), "samples_per_s": round(args.batch / max(s + w, 1e-9), 1),
                       "lr": sched.get_last_lr()[0], "fit/pred_loss": loss.item(), "fit/loss": loss.item(),
                       "copy_last": cp.item(), "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None}
                first = first or rec
                write(rec)
                steps_t, waits = [], []
            if step % args.check_every == 0:
                chk = checks(model, vl, dev, ac)
                is_best = chk["pred_over_copy"] < best["ratio"]
                if is_best:
                    best.update(ratio=chk["pred_over_copy"], step=step)
                write({"kind": "checks", "step": step, **chk, "best_pred_over_copy": best["ratio"], "best_step": best["step"]})
                save("weights_last.pt")
                if is_best:
                    save("weights_best.pt")          # best validation error vs copy-last so far (owner's correction, 2026-10-12)
                if args.save_all:
                    save(f"ckpts/step{step}.pt")
            if step >= args.steps or (args.stop_at_step and step >= args.stop_at_step) or (args.max_time and time.time() - t0 > args.max_time):
                done = True
                break
    res = checks(model, vl, dev, ac)
    res |= {"pass": {"beats_copy_last": res["pred_over_copy"] <= 1.0, "action_shuffle_hurts": res["shuffle_ratio"] >= 1.2,
                     "finite": all(math.isfinite(v) for v in res.values() if isinstance(v, float))}}
    res["all_pass"] = all(res["pass"].values())
    rec = {"kind": "checks", "tag": "final", "step": step, **res}
    write(rec)
    (args.out / "collapse_checks.json").write_text(json.dumps(rec, indent=1))
    save("weights_final.pt")
    write({"kind": "end", "step": step, "elapsed_s": round(time.time() - t0, 1)})


def losses_train(model, tok, act, ac):
    x = ln(tok)
    with ac:
        pred = predict_windows(model, x, act).float()
    tgt, prev = x[:, C:C + L], x[:, C - 1:C - 1 + L]
    return F.mse_loss(pred, tgt), F.mse_loss(prev, tgt), None


if __name__ == "__main__":
    main()
