"""G1 / G1b evaluation suite (G1_PROTOCOL.md), one code path for every candidate (W1 now, W2 now, W3 later).

On the 25 held-out val episodes (+ the train episodes for the probes):
  g1b        plannable range: ranking accuracy of the true chunk vs 7 distractors per (d, H)  [g1b_range.py logic]
  corruption action-corruption ratios at K = 1, 5, 10, 20, 30: cross-episode, zero, held-first, shifts -15 / +15
  openloop   true-action rollout MSE vs copy-last at K = 1..30 (ratios)
  probes     image-only latent z_t (all dimensions, standardized), linear: arm pose R2, gripper position R2, gripper open
             AUROC, grasp onset AUROC; ridge in kernel form (dual if dim > n), alpha by grouped (by episode) 5-fold CV on the
             train episodes, refit on all train episodes, scored on val. Anchors: every 3rd frame (probe.py's).

    WM_SPLIT=split.json python g1_suite.py w1 <weights dir> --data <snapshot> --out results/<date>_g1_w1.json
    WM_SPLIT=split.json python g1_suite.py w2 <weights.pt> --data <snapshot> --features <dir> --out results/<date>_g1_w2.json
Parts: --only g1b corruption probes (default: all). --max-episodes N: smoke (not a protocol run).
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
import g1b_range as G  # noqa: E402
import probe as P  # noqa: E402

DEV = P.DEV
C, K, STRIDE = 3, 30, 3
ALPHAS = [10.0 ** e for e in range(-1, 8)]
KS = (1, 5, 10, 20, 30)


# ---------------------------------------------------------------- candidates

class W1Cand(G.W1):
    def load(self, data, tab, eps, workers):
        return G.embed_episodes(self, data, tab, eps, workers)

    def flat_at(self, Z, ep, ts):
        return Z[ep][ts].float()


class W2Cand:
    name = "W2"

    def __init__(self, weights: Path, features: Path):
        import train_w2 as W

        self.W, self.features = W, features
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        self.model = W.W2Predictor(ckpt["config"]["tokens"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(DEV).eval()

    def _tokens(self, ep):
        return np.concatenate([np.load(self.features / f"ep{ep:06d}_{v}.npy", mmap_mode="r") for v in self.W.VIEWS], axis=1)

    def load(self, data, tab, eps, workers):
        return {ep: self.W.ln(torch.from_numpy(np.asarray(self._tokens(ep)))) for ep in eps}   # (T, tokens, 384) float32

    def flat_at(self, Z, ep, ts):
        return Z[ep][ts].flatten(1).half()

    @torch.no_grad()
    def rollout(self, z_ctx, a_norm):
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            return self.model.rollout(z_ctx.to(DEV), a_norm.to(DEV), G.HMAX).float().cpu()


# ---------------------------------------------------------------- G1b + corruption + open loop

def run_rollouts(cand, Z, eps, a_norm):
    anchors, chunks = G.build_chunks(a_norm, eps, np.random.default_rng(0))
    chunks = np.concatenate([chunks, np.zeros_like(chunks[:, :1])], 1)             # chunk 8: zero (mean) actions, not ranked
    extra = {}
    dist, err, copy = G.rollout_stream(cand, Z, anchors, chunks, extra=extra)
    cells = G.cells_from(dist, anchors, k_rank=8)
    out = {"anchors": len(anchors), "g1b": {"cells": cells, "range": {f"H={h}": G.plannable_range(cells, h) for h in G.HS},
                                           "separation": G.separation_report(dist, extra)}}
    e = err.mean(0)                                                                 # (K chunks, HMAX), mean over anchors
    true = e[0]
    groups = {"cross_episode": e[1:5].mean(0), "shift_-15": e[5], "shift_+15": e[6], "held_first": e[7], "zero": e[8]}
    out["corruption"] = {str(k): {"true_mse": float(true[k - 1]), "copy_last_mse": float(copy.mean(0)[k - 1]),
                                  **{f"{n}_ratio": round(float(v[k - 1] / true[k - 1]), 3) for n, v in groups.items()}} for k in KS}
    out["corruption_min_ratio_k30"] = min(out["corruption"]["30"][f"{n}_ratio"] for n in groups)
    out["openloop_over_copy_by_k"] = [round(float(true[k] / copy.mean(0)[k]), 4) for k in range(G.HMAX)]
    out["openloop_over_copy_k30"] = out["openloop_over_copy_by_k"][-1]
    return out


# ---------------------------------------------------------------- probes

def auroc(y, s):
    return P.auroc(y, s)


def r2(y, p):
    ss_res = ((y - p) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return float((1 - ss_res / ss_tot).mean())


def ridge_path(Xa, Ya, Xb):
    """Ridge predictions of Xb for every alpha. Xa (n, d), Ya (n, k): standardized with Xa's statistics, Ya centered.
    Kernel (dual) form when d > n, primal otherwise. -> (len(ALPHAS), nb, k)."""
    home = Xa.device
    if home.type == "mps":                                                         # eigh is not implemented on MPS
        Xa, Ya, Xb = Xa.cpu(), Ya.cpu(), Xb.cpu()
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    Xa, Xb = (Xa - mu) / sd, (Xb - mu) / sd
    ym = Ya.mean(0)
    Yc = Ya - ym
    n, d = Xa.shape
    if d > n:
        s, U = torch.linalg.eigh(Xa @ Xa.T)
        s = s.clamp(min=0)
        A, B = U.T @ Yc, (Xb @ Xa.T) @ U
        return torch.stack([B @ (A / (s + a)[:, None]) + ym for a in ALPHAS]).to(home)
    s, V = torch.linalg.eigh((Xa.T @ Xa).double())
    s, V = s.clamp(min=0).float(), V.float()
    A = V.T @ (Xa.T @ Yc)
    return torch.stack([Xb @ (V @ (A / (s + a)[:, None])) + ym for a in ALPHAS]).to(home)


def probe_group(X, Y, kinds, groups, Xv, Yv, seed=0):
    """alpha per target by grouped 5-fold CV on (X, Y); refit on all; score on (Xv, Yv). kinds: per column 'r2' | 'auc'.
    Columns of the same r2 target (arm pose) share one alpha through `targets` handled by the caller (one call per target)."""
    ug = np.unique(groups)
    fold = np.random.default_rng(seed).permutation(len(ug)) % 5
    gmap = dict(zip(ug, fold))
    f = np.array([gmap[g] for g in groups])
    cv, used = torch.zeros(len(ALPHAS)), 0
    for k in range(5):
        tr, te = torch.as_tensor(f != k), torch.as_tensor(f == k)
        if not te.any() or not tr.any():
            continue                                                                # fewer than 5 groups (smoke only)
        used += 1
        pred = ridge_path(X[tr], Y[tr], X[te])
        for i in range(len(ALPHAS)):
            cv[i] += (r2(Y[te].cpu().numpy(), pred[i].cpu().numpy()) if kinds == "r2"
                      else auroc(Y[te, 0].cpu().numpy() > 0.5, pred[i][:, 0].cpu().numpy()))
    cv = cv / max(used, 1)
    best = int(torch.argmax(cv))
    pred = ridge_path(X, Y, Xv)[best]
    score = r2(Yv.cpu().numpy(), pred.cpu().numpy()) if kinds == "r2" else auroc(Yv[:, 0].cpu().numpy() > 0.5, pred[:, 0].cpu().numpy())
    return {"score": round(float(score), 4), "alpha": ALPHAS[best], "cv_scores": [round(float(c), 4) for c in cv]}


def anchor_table(tab, eps):
    rows = []
    for ep in eps:
        _, act, st = tab[ep]
        c1, o2 = P.phases(act[:, 5])
        for t in range(C - 1, len(act) - K, STRIDE):
            rows.append((ep, t, c1))
    return rows


def run_probes(cand, data, tab, split, workers, Zval=None, limit=None):
    tr_eps, va_eps = split["train_episodes"], split["val_episodes"]
    if limit:
        tr_eps, va_eps = tr_eps[:limit], va_eps[:max(1, limit // 4)]
    feats, meta = {}, {}
    for name, eps in (("train", tr_eps), ("val", va_eps)):
        rows = anchor_table(tab, eps)
        meta[name] = rows
        X = []
        for ep in eps:
            Z = cand.load(data, tab, [ep], workers) if name == "train" or Zval is None else {ep: Zval[ep]}
            ts = [t for e, t, _ in rows if e == ep]
            X.append(cand.flat_at(Z, ep, ts))
        feats[name] = torch.cat(X).to(DEV).float()
        print(f"probe features {name}: {tuple(feats[name].shape)}", flush=True)

    def labels(rows):
        st = np.stack([tab[e][2][t] for e, t, _ in rows])
        c1 = np.array([c for _, _, c in rows])
        t = np.array([t for _, t, _ in rows])
        return {"arm": st[:, :5], "grip": st[:, 5:6], "open": (st[:, 5:6] > P.THR).astype(np.float32),
                "onset": (((t >= c1 - P.ONSET) & (t < c1))[:, None]).astype(np.float32), "ok": c1 >= 0}

    ltr, lva = labels(meta["train"]), labels(meta["val"])
    gtr = np.array([e for e, _, _ in meta["train"]])
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)  # noqa: E731
    out = {"anchors": {"train": len(meta["train"]), "val": len(meta["val"])}, "feature_dim": int(feats["train"].shape[1])}
    for name, key, kind in (("arm_pose_r2", "arm", "r2"), ("gripper_pos_r2", "grip", "r2"), ("gripper_open_auroc", "open", "auc")):
        out[name] = probe_group(feats["train"], T(ltr[key]), kind, gtr, feats["val"], T(lva[key]))
        print(name, out[name]["score"], "alpha", out[name]["alpha"], flush=True)
    a, b = torch.as_tensor(ltr["ok"]), torch.as_tensor(lva["ok"])
    out["grasp_onset_auroc"] = probe_group(feats["train"][a], T(ltr["onset"][ltr["ok"]]), "auc", gtr[ltr["ok"]],
                                           feats["val"][b], T(lva["onset"][lva["ok"]]))
    print("grasp_onset_auroc", out["grasp_onset_auroc"]["score"], flush=True)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", choices=["w1", "w2"])
    ap.add_argument("weights", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--features", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", nargs="*", choices=["g1b", "probes"], default=["g1b", "probes"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-episodes", type=int)
    args = ap.parse_args()
    torch.manual_seed(0)
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    cand = W1Cand(args.weights) if args.candidate == "w1" else W2Cand(args.weights, args.features)
    tab = P.frame_table(args.data)
    a_mean, a_std = P.action_stats(args.data, split["train_episodes"])
    res = {"candidate": cand.name, "protocol": "wm_train/G1_PROTOCOL.md", "revision": split["revision"], "device": DEV,
           "smoke": args.max_episodes is not None}
    t0 = time.time()
    Zval = None
    if "g1b" in args.only:
        eps = split["val_episodes"][:args.max_episodes]
        a_norm = {ep: ((tab[ep][1] - a_mean) / a_std).astype(np.float32) for ep in eps}
        Zval = cand.load(args.data, tab, eps, args.workers)
        res |= run_rollouts(cand, Zval, eps, a_norm)
        print("\nG1b ranking accuracy (H = d):", {d: res["g1b"]["cells"][f"d{d}_Hd"]["accuracy"] for d in G.DS}, flush=True)
        print("corruption ratios k=30:", {k: v for k, v in res["corruption"]["30"].items() if k.endswith("ratio")}, flush=True)
        print("open-loop / copy-last at K=30:", res["openloop_over_copy_k30"], flush=True)
    if "probes" in args.only:
        res["probes"] = run_probes(cand, args.data, tab, split, args.workers, Zval, limit=args.max_episodes)
    res["seconds"] = round(time.time() - t0, 1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
