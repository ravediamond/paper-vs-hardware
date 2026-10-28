"""WS-051 gate 0, steps 2–3: train the HWM high level on frozen W1 latents and score H1–H3 (PROTOCOL.md).

    python hwm_train.py <extract dir> <results dir> [--steps 20000]

Three high-level models, identical except the macro-action input: learned (transformer action encoder, CLS → 32),
concat (actions zero-padded to 120 × 6, linear → 32), disp (net joint displacement, 6 → 32). Each: causal transformer
over (l_t1, z_t1, l_t2, z_t2) predicting z_t2, z_t3 with teacher-forcing ℓ1. Evaluation on the val anchors of
lowlevel.npz: one F2 step from (l(a_{t:t+h}), z_t) → ẑ_{t+h}, against W1's rollout at the same h.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, L, MAXA, SEED = 192, 32, 120, 1234


class ActEnc(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "learned":
            self.inp = nn.Linear(6, 128)
            self.cls = nn.Parameter(torch.zeros(1, 1, 128))
            self.pos = nn.Parameter(torch.zeros(1, MAXA + 1, 128))
            layer = nn.TransformerEncoderLayer(128, 4, 512, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 2)
            self.out = nn.Linear(128, L)
        elif kind == "concat":
            self.out = nn.Linear(MAXA * 6, L)
        else:
            self.out = nn.Linear(6, L)

    def forward(self, a, mask):  # a (B, MAXA, 6) normalised, zero-padded; mask (B, MAXA) True = real step
        if self.kind == "learned":
            x = torch.cat([self.cls.expand(len(a), -1, -1), self.inp(a)], 1) + self.pos
            pad = torch.cat([torch.zeros(len(a), 1, dtype=torch.bool, device=a.device), ~mask], 1)
            return self.out(self.tf(x, src_key_padding_mask=pad)[:, 0])
        if self.kind == "concat":
            return self.out(a.flatten(1))
        last = a[torch.arange(len(a), device=a.device), mask.sum(1) - 1]  # commands are absolute joint targets:
        return self.out(last - a[:, 0])  # net joint displacement over the chunk = last − first


class F2(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.enc = ActEnc(kind)
        self.lp, self.zp = nn.Linear(L, 256), nn.Linear(D, 256)
        self.pos = nn.Parameter(torch.zeros(1, 4, 256))
        layer = nn.TransformerEncoderLayer(256, 4, 1024, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 4)
        self.head = nn.Linear(256, D)
        self.register_buffer("causal", torch.triu(torch.ones(4, 4, dtype=torch.bool), 1))

    def forward(self, zs, acts, masks):  # zs: list of z_tk (B, D); acts / masks: list of chunks
        toks = []
        for zk, ak, mk in zip(zs, acts, masks):
            toks += [self.lp(self.enc(ak, mk)), self.zp(zk)]
        x = torch.stack(toks, 1) + self.pos[:, :len(toks)]
        h = self.tf(x, mask=self.causal[:len(toks), :len(toks)])
        return self.head(h[:, 0::2])  # predictions after each macro-action token: z_t2, z_t3, …


def chunk(actions, a0, a1, a_mean, a_std):
    seg = (actions[a0:a1] - a_mean) / a_std
    out = np.zeros((MAXA, 6), np.float32)
    out[:len(seg)] = seg
    m = np.zeros(MAXA, bool)
    m[:len(seg)] = True
    return out, m


def train(kind, d, steps, log):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    z, act, ep, fr, val = d["z"], d["actions"], d["episode"], d["frame"], d["is_val"]
    am, asd = d["a_mean"], d["a_std"]
    tr_eps = [e for e in np.unique(ep[(~val) & (ep >= 0)])]
    bounds = {e: (np.flatnonzero(ep == e)[0], np.flatnonzero(ep == e)[-1]) for e in tr_eps}
    model = F2(kind).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    t0 = time.time()
    for step in range(steps):
        Z, A, M = [[], [], []], [[], []], [[], []]
        for _ in range(256):
            e = tr_eps[rng.integers(len(tr_eps))]
            s, f = bounds[e]
            span = int(rng.integers(20, 121))
            if f - s < span + 1:
                span = f - s - 1
            t1 = int(rng.integers(s, f - span + 1))
            t3 = t1 + span
            t2 = int(rng.integers(t1 + 1, t3))
            for k, t in enumerate((t1, t2, t3)):
                Z[k].append(z[t])
            for k, (a0, a1) in enumerate(((t1, t2), (t2, t3))):
                c, m = chunk(act, a0, a1, am, asd)
                A[k].append(c)
                M[k].append(m)
        zt = [torch.tensor(np.stack(x), device=DEV) for x in Z]
        at = [torch.tensor(np.stack(x), device=DEV) for x in A]
        mt = [torch.tensor(np.stack(x), device=DEV) for x in M]
        pred = model(zt[:2], at, mt)
        loss = (pred - torch.stack(zt[1:], 1)).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            log(f"{kind} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    return model


@torch.no_grad()
def evaluate(model, d, low):
    z, act, am, asd = d["z"], d["actions"], d["a_mean"], d["a_std"]
    anchors, H = low["anchors"], low["H"]
    err, cos = np.zeros((len(anchors), len(H))), np.zeros((len(anchors), len(H)))
    model.eval()
    for j, h in enumerate(H):
        for i in range(0, len(anchors), 256):
            idx = anchors[i:i + 256]
            cm = [chunk(act, t, t + h, am, asd) for t in idx]
            a = torch.tensor(np.stack([c for c, _ in cm]), device=DEV)
            m = torch.tensor(np.stack([mm for _, mm in cm]), device=DEV)
            zt = torch.tensor(z[idx], device=DEV)
            p = model([zt], [a], [m])[:, 0].cpu().numpy()
            true = z[idx + h]
            err[i:i + len(idx), j] = np.abs(p - true).mean(1)
            dp, dt = p - z[idx], true - z[idx]
            cos[i:i + len(idx), j] = (dp * dt).sum(1) / (np.linalg.norm(dp, axis=1) * np.linalg.norm(dt, axis=1) + 1e-9)
    return err, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extract", type=Path)
    ap.add_argument("results", type=Path)
    ap.add_argument("--steps", type=int, default=20000)
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    logf = open(args.results / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    d = dict(np.load(args.extract / "latents.npz"))
    low = dict(np.load(args.extract / "lowlevel.npz"))
    H = low["H"]
    lo_err = np.abs(low["pred"] - low["true"]).mean(2)  # (anchors, H)
    res = {"device": DEV, "steps": args.steps, "anchors": int(len(low["anchors"])), "H_frames": H.tolist(),
           "low": {"mean": lo_err.mean(0).round(5).tolist(), "se": (lo_err.std(0) / np.sqrt(len(lo_err))).round(5).tolist()}}
    for kind in ("learned", "concat", "disp"):
        m = train(kind, d, args.steps, log)
        err, cos = evaluate(m, d, low)
        res[kind] = {"err_mean": err.mean(0).round(5).tolist(), "err_se": (err.std(0) / np.sqrt(len(err))).round(5).tolist(),
                     "cos_mean": round(float(cos.mean()), 4), "cos_se": round(float(cos.mean(1).std() / np.sqrt(len(cos))), 4),
                     "cos_by_h": cos.mean(0).round(4).tolist()}
        torch.save(m.state_dict(), args.results / f"f2_{kind}.pt")
    hi, lo = np.array(res["learned"]["err_mean"]), np.array(res["low"]["mean"])
    below = hi < lo
    hstar = next((int(H[k]) for k in range(len(H)) if below[k:].all() and not below[:k].any()), None)
    res["bars"] = {
        "H1": {"crossover_frames": hstar, "high_below_low": below.tolist(),
               "met": hstar is not None and 30 <= hstar <= 60},
        "H2": {"high_2s": float(hi[-1]), "low_2s": float(lo[-1]), "ratio": round(float(hi[-1] / lo[-1]), 4),
               "met": bool(hi[-1] <= 0.95 * lo[-1])},
        "H3": {"learned": res["learned"]["cos_mean"], "disp": res["disp"]["cos_mean"], "concat": res["concat"]["cos_mean"],
               "met": res["learned"]["cos_mean"] >= res["disp"]["cos_mean"]}}
    (args.results / "gate0.json").write_text(json.dumps(res, indent=1))
    log(json.dumps(res["bars"], indent=1))


if __name__ == "__main__":
    main()
