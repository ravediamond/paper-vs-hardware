"""Exploratory G1 diagnostics (G1_PROTOCOL.md, "Exploratory diagnostics after the G1b result"), Mac only.

  1 action-only baseline on G1b: rank the 8 chunks by |last commanded joints - measured joints at t + d| (no model, no images)
  2 regret of W1 / W2 / action-only: (selected - expert) / (random - expert) of that same end-state distance, H = d
  3 imagined-latent arm-pose probe at K = 10, 30 (W1 vs W2): ridge fit on real latents of the first --train-eps train episodes,
    applied to imagined latents; ceiling = real z_{t+K}, floor = copy-last z_t

    WM_SPLIT=split.json python g1_diag.py --data <snapshot> --w1 <w1 dir> --w2 <weights.pt> --features <dir> --out results/<date>_g1_diag.json
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
import g1_suite as S  # noqa: E402
import g1b_range as G  # noqa: E402
import probe as P  # noqa: E402

C, DS, KS = G.C, G.DS, (10, 30)


def end_state_dist(chunks, anchors, s_norm):
    """(N, 8) per d: L2 between the chunk's last commanded joints (a_{t+d-1}) and the measured joints at t + d (normalized units)."""
    out = {}
    for d in DS:
        tgt = np.stack([s_norm[ep][t + d] for ep, t in anchors])
        out[d] = torch.as_tensor(np.linalg.norm(chunks[:, :8, C + d - 2] - tgt[:, None], axis=-1), dtype=torch.float32)
    return out


def regret(score, cost, anchors, n_boot=G.N_BOOT):
    """score, cost (N, 8), chunk 0 = true. selected = argmin score; regret = (sel - expert) / (random - expert), means over anchors."""
    sel = cost[torch.arange(len(cost)), score.argmin(1)].numpy()
    exp, rnd = cost[:, 0].numpy(), cost.mean(1).numpy()
    eps = np.array([ep for ep, _ in anchors])
    uniq = np.unique(eps)
    per = np.stack([[sel[eps == u].sum(), exp[eps == u].sum(), rnd[eps == u].sum(), (eps == u).sum()] for u in uniq])
    f = lambda x: (x[0] - x[1]) / (x[2] - x[1])  # noqa: E731
    rng = np.random.default_rng(0)
    boots = [f(per[rng.integers(len(uniq), size=len(uniq))].sum(0)) for _ in range(n_boot)]
    return {"regret": round(float(f(per.sum(0))), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "selected": round(float(sel.mean()), 4), "expert": round(float(exp.mean()), 4), "random": round(float(rnd.mean()), 4)}


@torch.no_grad()
def imagined(cand, Z, anchors, true_chunks, batch=4):
    """Flat imagined latent after K steps for K in KS, from the real context with the true actions. -> {K: (N, dim)}."""
    out = {k: [] for k in KS}
    for i in range(0, len(anchors), batch):
        an = anchors[i:i + batch]
        ctx = torch.stack([Z[ep][t - C + 1:t + 1] for ep, t in an]).float()
        rolled = cand.rollout(ctx, torch.as_tensor(true_chunks[i:i + batch])).float()
        for k in KS:
            out[k].append(rolled[:, k - 1].flatten(1).cpu())
    res = {k: torch.cat(v) for k, v in out.items()}
    if not all(torch.isfinite(v).all() for v in res.values()):
        raise SystemExit("non-finite imagined latents (MPS NaN?)")
    return res


def probe_imagined(cand, data, tab, split, workers, Zval, anchors, true_chunks, train_eps):
    rows = S.anchor_table(tab, train_eps)
    X = []
    for ep in train_eps:
        Z = cand.load(data, tab, [ep], workers)
        X.append(cand.flat_at(Z, ep, [t for e, t, _ in rows if e == ep]).float())
    X = torch.cat(X).to(S.DEV)
    Y = torch.as_tensor(np.stack([tab[e][2][t][:5] for e, t, _ in rows]), dtype=torch.float32, device=S.DEV)
    groups = np.array([e for e, _, _ in rows])
    t_idx = lambda dt: np.array([t + dt for _, t in anchors])  # noqa: E731
    flat = lambda ts: torch.stack([cand.flat_at(Zval, ep, [t]).float()[0] for (ep, _), t in zip(anchors, ts)]).to(S.DEV)  # noqa: E731
    real_t = flat(t_idx(0))
    ref = S.probe_group(X, Y, "r2", groups, real_t, torch.as_tensor(np.stack([tab[ep][2][t][:5] for ep, t in anchors]), dtype=torch.float32, device=S.DEV))
    alpha = S.ALPHAS.index(ref["alpha"])
    im = imagined(cand, Zval, anchors, true_chunks)
    out = {"fit_anchors": len(rows), "fit_episodes": len(train_eps), "alpha": ref["alpha"], "feature_dim": int(X.shape[1]),
           "arm_pose_r2_real_t_vs_pose_t": ref["score"]}
    for k in KS:
        tgt = np.stack([tab[ep][2][t + k][:5] for ep, t in anchors])
        sets = {"imagined": im[k].to(S.DEV), "ceiling_real_z_t+K": flat(t_idx(k)), "floor_copy_last_z_t": real_t}
        cat = torch.cat(list(sets.values()))
        pred = S.ridge_path(X, Y, cat)[alpha].cpu().numpy()
        n = len(anchors)
        r = {name: round(S.r2(tgt, pred[i * n:(i + 1) * n]), 4) for i, name in enumerate(sets)}
        gap = r["ceiling_real_z_t+K"] - r["floor_copy_last_z_t"]
        r["fraction_of_gap_recovered"] = round((r["imagined"] - r["floor_copy_last_z_t"]) / gap, 4) if abs(gap) > 1e-6 else None
        out[f"K{k}"] = r
        print(f"  K={k}", r, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--w1", type=Path, required=True)
    ap.add_argument("--w2", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16, help="anchors per rollout batch (W2 on the Mac needs 1: attention over 33 x 128 tokens)")
    ap.add_argument("--train-eps", type=int, default=40)
    ap.add_argument("--max-episodes", type=int, help="smoke only")
    ap.add_argument("--cands", nargs="*", choices=["W1", "W2_best"], default=["W1", "W2_best"], help="run one candidate per process (memory)")
    ap.add_argument("--only", nargs="*", choices=["baseline", "regret", "probe"], default=["baseline", "regret", "probe"])
    args = ap.parse_args()
    torch.manual_seed(0)
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    eps = split["val_episodes"][:args.max_episodes]
    tab = P.frame_table(args.data)
    a_mean, a_std = P.action_stats(args.data, split["train_episodes"])
    a_norm = {ep: ((tab[ep][1] - a_mean) / a_std).astype(np.float32) for ep in eps}
    s_norm = {ep: ((tab[ep][2] - a_mean) / a_std).astype(np.float32) for ep in eps}
    anchors, chunks = G.build_chunks(a_norm, eps, np.random.default_rng(0))
    cost = end_state_dist(chunks, anchors, s_norm)
    res = {"protocol": "wm_train/G1_PROTOCOL.md (exploratory diagnostics)", "revision": split["revision"], "device": S.DEV,
           "anchors": len(anchors), "smoke": args.max_episodes is not None}
    t0 = time.time()
    if "baseline" in args.only:
        cells = G.cells_from({(d, "d"): cost[d] for d in DS}, anchors)
        res["action_only_g1b"] = cells
        print("action-only ranking accuracy (H = d):", {d: cells[f"d{d}_Hd"]["accuracy"] for d in DS}, flush=True)
    cands, Zs = {}, {}
    if set(args.only) - {"baseline"}:
        cands = {n: (S.W1Cand(args.w1) if n == "W1" else S.W2Cand(args.w2, args.features)) for n in args.cands}
        for name, cand in cands.items():
            Zs[name] = {ep: z.half() for ep, z in cand.load(args.data, tab, eps, args.workers).items()}   # half in memory (24 GB Mac)
    if "regret" in args.only:
        res["regret"] = {"action_only": {f"d{d}": regret(cost[d], cost[d], anchors) for d in DS}}
        print("action_only regret:", {f"d{d}": res["regret"]["action_only"][f"d{d}"]["regret"] for d in DS}, flush=True)
        for name, cand in cands.items():
            dist, _, _ = G.rollout_stream(cand, Zs[name], anchors, chunks[:, :8], batch=args.batch)
            res["regret"][name] = {f"d{d}": regret(dist[(d, "d")], cost[d], anchors) for d in DS}
            print(name, "regret:", {f"d{d}": res["regret"][name][f"d{d}"]["regret"] for d in DS}, flush=True)
    if "probe" in args.only:
        res["imagined_probe"] = {}
        train_eps = split["train_episodes"][:args.train_eps]
        for name, cand in cands.items():
            print("imagined-latent probe", name, flush=True)
            res["imagined_probe"][name] = probe_imagined(cand, args.data, tab, split, args.workers, Zs[name], anchors,
                                                         chunks[:, 0], train_eps if args.max_episodes is None else train_eps[:6])
    res["seconds"] = round(time.time() - t0, 1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
