"""Exploratory investigation of the G1b selection failure (G1_PROTOCOL.md, "Exploratory investigation: why model selection fails G1b").

    WM_SPLIT=split.json python g1_diag2.py --parts pair align cond corr --data <snapshot> --w2 <weights_best.pt> --features <dir> --out <json>

pair   pairwise accuracy^7 vs observed top-1 (from the committed G1 results + the action-only baseline)
align  per-episode, per-joint lag between commanded and measured joint velocity
cond   AdaLN modulation magnitudes, gradient of the prediction wrt actions vs tokens, action normalization stats (W2 best-val)
corr   corruption ratios at K = 10: training context vs context 1, whole latent vs arm tokens vs the rest (W2 best-val)
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
import g1_diag as D  # noqa: E402
import g1_suite as S  # noqa: E402
import g1b_range as G  # noqa: E402
import probe as P  # noqa: E402
import train_w2 as W  # noqa: E402

C, DS = G.C, G.DS
RESULTS = Path(__file__).resolve().parent / "results"


def setup(args):
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    tab = P.frame_table(args.data)
    a_mean, a_std = P.action_stats(args.data, split["train_episodes"])
    return split, tab, a_mean, a_std


def part_pair(args, split, tab, a_mean, a_std):
    eps = split["val_episodes"]
    a_norm = {ep: ((tab[ep][1] - a_mean) / a_std).astype(np.float32) for ep in eps}
    s_norm = {ep: ((tab[ep][2] - a_mean) / a_std).astype(np.float32) for ep in eps}
    anchors, chunks = G.build_chunks(a_norm, eps, np.random.default_rng(0))
    cost = D.end_state_dist(chunks, anchors, s_norm)
    out = {}
    base = json.loads((RESULTS / "2026-10-12_g1_diag_exploratory.json").read_text())
    for name, f in (("W1", "w1"), ("W2_best", "w2_best"), ("W2_final", "w2_final")):
        j = json.loads((RESULTS / f"2026-10-12_g1_final_{f}.json").read_text())["g1b"]
        out[name] = {f"d{d}": {"pairwise": j["separation"][f"d{d}"]["pairwise_accuracy"], "top1": j["cells"][f"d{d}_Hd"]["accuracy"]} for d in DS}
    out["action_only"] = {}
    for d in DS:
        win = (cost[d][:, :1] < cost[d][:, 1:8]).float().mean().item()
        out["action_only"][f"d{d}"] = {"pairwise": round(win, 4), "top1": base["action_only_g1b"][f"d{d}_Hd"]["accuracy"]}
    for m in out.values():
        for v in m.values():
            v["pairwise_pow7"] = round(v["pairwise"] ** 7, 4)
            v["top1_over_pow7"] = round(v["top1"] / v["pairwise_pow7"], 3)
    return out


def part_align(args, split, tab, a_mean, a_std):
    lags = np.arange(0, 31)
    per_ep, offsets = {}, []
    for ep, (_, act, st) in tab.items():
        da, ds = np.diff(act, axis=0), np.diff(st, axis=0)
        row = []
        for j in range(6):
            cs = []
            for L in lags:
                n = len(da) - L
                cs.append(np.corrcoef(da[:n, j], ds[L:L + n, j])[0, 1] if n > 10 and da[:n, j].std() > 0 and ds[L:L + n, j].std() > 0 else np.nan)
            cs = np.array(cs)
            row.append({"lag": int(lags[np.nanargmax(cs)]), "corr": round(float(np.nanmax(cs)), 3)})
        per_ep[str(ep)] = row
        offsets.append((st - act).mean(0))
    out = {"episodes": len(per_ep), "lags_searched": [0, 30], "per_episode": per_ep, "joint": {}}
    for j in range(6):
        lg = np.array([per_ep[e][j]["lag"] for e in per_ep])
        cr = np.array([per_ep[e][j]["corr"] for e in per_ep])
        out["joint"][str(j)] = {"median_lag": float(np.median(lg)), "iqr": [float(np.percentile(lg, 25)), float(np.percentile(lg, 75))],
                                "min": int(lg.min()), "max": int(lg.max()), "at_boundary": int(((lg == 0) | (lg == 30)).sum()),
                                "median_peak_corr": round(float(np.median(cr)), 3), "mean_offset_state_minus_action": round(float(np.mean([o[j] for o in offsets])), 3)}
    return out


def load_w2(weights):
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    m = W.W2Predictor(ck["config"]["tokens"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def val_tokens(features, ep, t_lo, t_hi):
    return W.ln(torch.from_numpy(np.concatenate([np.load(features / f"ep{ep:06d}_{v}.npy", mmap_mode="r")[t_lo:t_hi] for v in W.VIEWS], axis=1)))


def part_cond(args, split, tab, a_mean, a_std):
    model = load_w2(args.w2)
    rng = np.random.default_rng(0)
    eps = split["val_episodes"]
    xs, acts = [], []
    for ep in eps:
        a = ((tab[ep][1] - a_mean) / a_std).astype(np.float32)
        for t in rng.choice(np.arange(C - 1, len(a) - 1), 11, replace=False):
            xs.append(val_tokens(args.features, ep, t - C + 1, t + 1))
            acts.append(torch.from_numpy(a[t - C + 1:t + 1]))
    x, a = torch.stack(xs)[:256], torch.stack(acts)[:256]
    out = {"anchors": int(len(x)), "adaln": [], "gradient": {}, "normalization": {}}
    with torch.no_grad():
        cond = torch.nn.functional.silu(model.act(a))
        for i, blk in enumerate(model.blocks):
            m = blk.mod(cond)                                                         # (B, 3, 6d)
            row = {}
            for name, ch in zip(("shift_attn", "scale_attn", "gate_attn", "shift_mlp", "scale_mlp", "gate_mlp"), m.chunk(6, -1)):
                row[name] = {"mean_abs": round(float(ch.abs().mean()), 5), "std_over_anchors_rel": round(float(ch.std(0).mean() / (ch.abs().mean() + 1e-9)), 3)}
            out["adaln"].append({"block": i, **row})
    xg, ag = x.clone().requires_grad_(True), a.clone().requires_grad_(True)
    pred = model(xg, ag)[:, -1]
    f = ((pred - xg[:, -1]) ** 2).sum()
    f.backward()
    ga, gx = ag.grad.abs().mean().item(), xg.grad.abs().mean().item()
    out["gradient"] = {"mean_abs_grad_wrt_action_per_element": ga, "mean_abs_grad_wrt_tokens_per_element": gx, "ratio_action_over_tokens": ga / gx,
                       "predicted_change_rms": float(((pred - xg[:, -1]).detach() ** 2).mean().sqrt()), "token_rms": float(xg.detach().pow(2).mean().sqrt())}
    tr = split["train_episodes"]
    raw_tr = np.concatenate([tab[e][1] for e in tr])
    norm_va = np.concatenate([(tab[e][1] - a_mean) / a_std for e in split["val_episodes"]])
    out["normalization"] = {"train_mean": a_mean.round(3).tolist(), "train_std": a_std.round(3).tolist(), "min_train_std": float(a_std.min()),
                            "val_in_train_units_mean": norm_va.mean(0).round(3).tolist(), "val_in_train_units_std": norm_va.std(0).round(3).tolist(),
                            "frac_val_abs_gt4": round(float((np.abs(norm_va) > 4).mean()), 5), "frac_val_abs_gt10": round(float((np.abs(norm_va) > 10).mean()), 6),
                            "per_joint_frac_gt4": (np.abs(norm_va) > 4).mean(0).round(4).tolist(), "raw_train_min": raw_tr.min(0).round(2).tolist(), "raw_train_max": raw_tr.max(0).round(2).tolist()}
    return out


def arm_tokens(args, split, tab, n_keep=32, alpha=1000.0):
    eps = split["train_episodes"][:40]
    fit, score = eps[:20], eps[20:]

    def gather(es):
        rows = S.anchor_table(tab, es)
        X = torch.cat([val_tokens(args.features, ep, 0, len(tab[ep][1]))[[t for e, t, _ in rows if e == ep]] for ep in es])        # (n, 128, 384)
        Y = torch.as_tensor(np.stack([tab[e][2][t][:5] for e, t, _ in rows]), dtype=torch.float32)
        return X, Y

    Xa, Ya = gather(fit)
    Xb, Yb = gather(score)
    r2 = []
    for k in range(Xa.shape[1]):
        a, b = Xa[:, k], Xb[:, k]
        mu, sd = a.mean(0), a.std(0) + 1e-6
        a, b = (a - mu) / sd, (b - mu) / sd
        ym = Ya.mean(0)
        w = torch.linalg.solve(a.T @ a + alpha * torch.eye(a.shape[1]), a.T @ (Ya - ym))
        r2.append(S.r2(Yb.numpy(), (b @ w + ym).numpy()))
    r2 = np.array(r2)
    keep = np.argsort(-r2)[:n_keep]
    return np.sort(keep), r2


def part_corr(args, split, tab, a_mean, a_std):
    model = load_w2(args.w2).to(S.DEV)
    keep, r2 = arm_tokens(args, split, tab)
    mask = np.zeros(128, bool)
    mask[keep] = True
    print("arm tokens:", int(mask.sum()), "of which view0:", int(mask[:64].sum()), "view1:", int(mask[64:].sum()), "R2 range", float(r2[keep].min()), float(r2.max()), flush=True)
    eps = split["val_episodes"][:args.max_episodes]
    a_norm = {ep: ((tab[ep][1] - a_mean) / a_std).astype(np.float32) for ep in eps}
    anchors, chunks = G.build_chunks(a_norm, eps, np.random.default_rng(0))
    chunks = np.concatenate([chunks, np.zeros_like(chunks[:, :1])], 1)                  # chunk 8 = zero actions
    sel = np.arange(0, len(anchors), 3)
    Z = {ep: val_tokens(args.features, ep, 0, len(tab[ep][1])).half() for ep in eps}
    Ks = (1, 5, 10)
    err = {v: {k: np.zeros((len(sel), 9, 128), np.float32) for k in Ks} for v in ("ctx3", "ctx1")}
    t0 = time.time()
    with torch.no_grad():
        for n, i in enumerate(sel):
            ep, t = anchors[i]
            ctx = Z[ep][t - C + 1:t + 1].float()
            fut = Z[ep][t + 1:t + 11].float()
            a = torch.as_tensor(chunks[i]).to(S.DEV)
            for v, c in (("ctx3", ctx), ("ctx1", ctx[-1:].repeat(C, 1, 1))):
                rolled = model.rollout(c[None].repeat(9, 1, 1, 1).to(S.DEV), a[:, :C + 9], 10).float().cpu()
                for k in Ks:
                    err[v][k][n] = ((rolled[:, k - 1] - fut[k - 1]) ** 2).mean(-1).numpy()
            if n % 20 == 0:
                print(f"anchor {n}/{len(sel)} {time.time() - t0:.0f}s", flush=True)
    groups = {"cross_episode": [1, 2, 3, 4], "shift_-15": [5], "shift_+15": [6], "held_first": [7], "zero": [8]}
    out = {"anchors": int(len(sel)), "arm_tokens": keep.tolist(), "arm_token_r2_selected_min": float(r2[keep].min()), "arm_token_r2_max": float(r2.max()),
           "arm_tokens_view0": int(mask[:64].sum()), "arm_tokens_view1": int(mask[64:].sum()), "ratios": {}}
    for v in err:
        for region, m in (("whole", np.ones(128, bool)), ("arm", mask), ("rest", ~mask)):
            for k in Ks:
                e = err[v][k][:, :, m].mean(-1).mean(0)                                 # (9,) mean over anchors and region tokens
                out["ratios"][f"{v}_{region}_K{k}"] = {"true_mse": float(e[0]), **{n: round(float(e[ix].mean() / e[0]), 3) for n, ix in groups.items()}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", choices=["pair", "align", "cond", "corr"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--w2", type=Path)
    ap.add_argument("--features", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-episodes", type=int)
    args = ap.parse_args()
    torch.manual_seed(0)
    split, tab, a_mean, a_std = setup(args)
    res = {"protocol": "wm_train/G1_PROTOCOL.md (exploratory investigation 2026-10-13)", "revision": split["revision"], "device": S.DEV}
    fns = {"pair": part_pair, "align": part_align, "cond": part_cond, "corr": part_corr}
    for p in args.parts:
        t0 = time.time()
        res[p] = fns[p](args, split, tab, a_mean, a_std)
        print(p, "done", round(time.time() - t0), "s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
