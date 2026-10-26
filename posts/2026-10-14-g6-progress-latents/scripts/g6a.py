"""G6a: on-robot done / stall check with world-model encoder latents (G6A_PROTOCOL.md). One module: `extract` (features) and `analyze` (tests).

    python g6a.py extract --data <snapshot> --w1 <w1 dir> --split split.json --out <dir>
    python g6a.py analyze --data <snapshot> --feats <dir> --split split.json --out results/g6a/g6a_results.json

extract: frozen DINOv2-S/14 patch tokens (16x16 per camera at 224x224, float16, raw last_hidden_state without CLS) and the W1 192-d latent for the
selected frames: all frames of the validation episodes; every 3rd frame plus the first and last frame of every train episode. Per episode:
ep{N:06d}_idx.npy (frame indices), _pixels.npy / _wrist.npy (n, 256, 384), _w1.npy (n, 192). Resumable.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1b_range as G  # noqa: E402
import probe as P  # noqa: E402

VIEWS = ("pixels", "wrist")


def frames_for(split_name: str, T: int) -> np.ndarray:
    if split_name == "val":
        return np.arange(T)
    return np.unique(np.concatenate([np.arange(0, T, 3), [0, T - 1]]))


def extract(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from transformers import AutoModel

    split = json.loads(args.split.read_text())
    eps = [("val", e) for e in split["val_episodes"]] + [("train", e) for e in split["train_episodes"]]
    if args.max_episodes:
        eps = eps[:args.max_episodes]
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(P.DEV).eval()
    w1 = G.W1(args.w1)
    tab = P.frame_table(args.data)
    ds = LeRobotDataset(f"local/{args.data.name}", root=args.data, delta_timestamps={c: [0.0] for c in P.CAMS.values()})
    args.out.mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 0
    for name, ep in eps:
        if (args.out / f"ep{ep:06d}_w1.npy").exists():
            continue
        start, act, _ = tab[ep]
        idx = frames_for(name, len(act))
        loader = torch.utils.data.DataLoader(G.FrameSet(ds, start + idx), batch_size=64, num_workers=args.workers, shuffle=False)
        tok = {v: [] for v in VIEWS}
        z1 = []
        with torch.no_grad():
            for b in loader:
                for v in VIEWS:
                    tok[v].append(dino(pixel_values=b[v].to(P.DEV)).last_hidden_state[:, 1:].half().cpu())
                z1.append(w1.embed({v: b[v] for v in P.CAMS}))
        for v in VIEWS:
            np.save(args.out / f"ep{ep:06d}_{v}.npy", torch.cat(tok[v]).numpy())
        np.save(args.out / f"ep{ep:06d}_idx.npy", idx)
        np.save(args.out / f"ep{ep:06d}_w1.npy", torch.cat(z1).numpy())     # written last: marks the episode as done
        n += len(idx)
        print(f"{name} episode {ep}: {len(idx)} frames, {n / (time.time() - t0):.1f} frames/s so far", flush=True)
    print(f"done: {n} frames in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--w1", type=Path, required=True)
    e.add_argument("--split", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--max-episodes", type=int)
    e.set_defaults(fn=extract)
    a = sub.add_parser("analyze")
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--feats", type=Path, required=True)
    a.add_argument("--split", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(fn=lambda args: analyze(args))
    args = ap.parse_args()
    args.fn(args)


# ----------------------------------------------------------------------------------------------------------------------- analysis

STRIDE_S = 3            # train series stride (frames), 0.1 s at 30 Hz
FPS = 30
GRID_N = {8: (4, 8, 16, 32), 16: (16, 32, 64, 128)}
W_GRID = (2.0, 3.0, 5.0, 8.0)
EPS_GRID = (0.0, 0.01, 0.02, 0.05, 0.1)
N_BOOT = 2000


def ln(x):
    return torch.nn.functional.layer_norm(x.float(), x.shape[-1:])


def cells_of(tok16, grid):
    """(n, 256, 384) fp16 raw patch tokens -> (n, grid*grid, 384) float32 layer-normed (grid 8 = 2x2 average pool, as extract_dino_features --pool2)."""
    x = torch.as_tensor(np.asarray(tok16))
    if grid == 8:
        n = x.shape[0]
        x = x.float().view(n, 8, 2, 8, 2, 384).mean(dim=(2, 4)).reshape(n, 64, 384)
    return ln(x)


def cell_dist(q, b):
    """q (nq, C, D), b (nb, C, D) -> (nq, nb, C) per-cell L2."""
    d2 = (q * q).sum(-1)[:, None, :] + (b * b).sum(-1)[None] - 2 * torch.einsum("qcd,bcd->qbc", q, b)
    return d2.clamp_min(0).sqrt()


class Data:
    """Metadata, joint states, gripper phases and lazily memory-mapped features."""

    def __init__(self, args):
        import pandas as pd

        self.split = json.loads(args.split.read_text())
        self.train, self.val = self.split["train_episodes"], self.split["val_episodes"]
        self.tab = P.frame_table(args.data)
        mean, std = P.action_stats(args.data, self.train)
        self.s = {e: ((self.tab[e][2] - mean) / std).astype(np.float32) for e in self.train + self.val}
        meta = pd.concat([pd.read_parquet(p) for p in sorted((args.data / "meta/episodes").glob("*/*.parquet"))]).set_index("episode_index")
        self.task = {e: str(meta.loc[e, "tasks"]) for e in self.train + self.val}
        self.feats = args.feats
        self.c1, self.cycles = {}, {}
        for e in self.train + self.val:
            g = self.tab[e][1][:, 5]
            self.c1[e] = P.phases(g)[0]
            op = g > P.THR
            closes = np.flatnonzero((~op[1:]) & op[:-1]) + 1
            opens = np.flatnonzero(op[1:] & ~op[:-1]) + 1
            self.cycles[e] = int(sum(bool((opens > c).any()) for c in closes))   # grasp cycles: a close later followed by a release (open)
        starts = np.stack([self.s[e][0] for e in self.train])
        self.rest = np.median(starts, 0)
        self.tol = float(np.percentile(np.linalg.norm(starts - self.rest, axis=1), 95))

    def idx(self, e):
        return np.load(self.feats / f"ep{e:06d}_idx.npy")

    def tok(self, e, cam):
        return np.load(self.feats / f"ep{e:06d}_{cam}.npy", mmap_mode="r")

    def w1(self, e):
        return np.load(self.feats / f"ep{e:06d}_w1.npy")

    def at_rest(self, e, idx):
        return np.linalg.norm(self.s[e][idx] - self.rest, axis=1) <= self.tol


class Bank:
    """End frames of the train episodes (per task), per grid and camera."""

    def __init__(self, D: Data):
        self.eps = D.train
        self.task = [D.task[e] for e in self.eps]
        self.w1 = torch.as_tensor(np.stack([D.w1(e)[-1] for e in self.eps])).float()
        self.cells = {g: {c: torch.cat([cells_of(D.tok(e, c)[-1:], g) for e in self.eps]) for c in VIEWS} for g in (8, 16)}
        self.first = {g: {c: torch.cat([cells_of(D.tok(e, c)[:1], g) for e in self.eps]) for c in VIEWS} for g in (8, 16)}
        self.rank = {g: {c: torch.argsort((self.cells[g][c] - self.first[g][c]).norm(dim=-1).mean(0), descending=True) for c in VIEWS}
                     for g in (8, 16)}


def variants(sel_n):
    """name -> spec. sel_n: {(grid, cams): N} (None while N is still being chosen)."""
    out = {"W1/both": ("w1",)}
    for g in (8, 16):
        for cams in ("top", "both"):
            out[f"W2-{g}x{g}/{cams}/all"] = ("w2", g, cams, None)
            if sel_n.get((g, cams)):
                out[f"W2-{g}x{g}/{cams}/work{sel_n[(g, cams)]}"] = ("w2", g, cams, sel_n[(g, cams)])
    return out


def distances(D, bank, queries, specs, chunk=192):
    """queries: list of (episode, frame positions into the episode's extracted arrays). specs: {name: spec}.
    Returns {name: (nearest (nq,), k3 (nq,))} in query order. Bank entries of the query's own episode and of other tasks are excluded."""
    q_ep = np.array([e for e, pos in queries for _ in pos])
    q_task = np.array([D.task[e] for e in q_ep])
    allowed = torch.as_tensor(np.array(bank.task)[None, :] == q_task[:, None]) & torch.as_tensor(np.array(bank.eps)[None, :] != q_ep[:, None])
    grids = sorted({sp[1] for sp in specs.values() if sp[0] == "w2"})
    cams_used = {c for sp in specs.values() if sp[0] == "w2" for c in (("pixels",) if sp[2] == "top" else VIEWS)}
    out = {n: ([], []) for n in specs}
    flat = [(e, p) for e, pos in queries for p in pos]
    for s0 in range(0, len(flat), chunk):
        part = flat[s0:s0 + chunk]
        am = allowed[s0:s0 + chunk].to(P.DEV)
        Dm = {}
        if grids:
            for g in grids:
                for c in cams_used:
                    q = torch.cat([cells_of(D.tok(e, c)[p:p + 1], g) for e, p in part]).to(P.DEV)
                    Dm[g, c] = cell_dist(q, bank.cells[g][c].to(P.DEV))
        if any(sp[0] == "w1" for sp in specs.values()):
            qw = torch.as_tensor(np.stack([D.w1(e)[p] for e, p in part])).float().to(P.DEV)
            d_w1 = torch.cdist(qw, bank.w1.to(P.DEV))
        for n, sp in specs.items():
            if sp[0] == "w1":
                dm = d_w1
            else:
                _, g, cams, N = sp
                parts = []
                for c in ("pixels",) if cams == "top" else VIEWS:
                    dc = Dm[g, c]
                    if N:
                        dc = dc[:, :, bank.rank[g][c][:N].to(P.DEV)]
                    parts.append(dc.mean(-1))
                dm = torch.stack(parts).mean(0)
            dm = dm.masked_fill(~am, float("inf"))
            sm = torch.sort(dm, dim=1).values[:, :3]
            out[n][0].append(sm[:, 0].cpu())
            out[n][1].append(sm.mean(1).cpu())
    return {n: (torch.cat(a).numpy(), torch.cat(b).numpy()) for n, (a, b) in out.items()}


def boot_ci(f, eps_list, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        s = rng.integers(len(eps_list), size=len(eps_list))
        v = f([eps_list[i] for i in s])
        if np.isfinite(v):
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def t2_sets(D, e, pos_idx_arr):
    """positives: the last frame; negatives: at rest before the first close; secondary negatives: 30-70 % of the episode."""
    idx = D.idx(e)
    T = len(D.tab[e][1])
    last = len(idx) - 1
    neg = np.flatnonzero(D.at_rest(e, idx) & (idx < D.c1[e])) if D.c1[e] > 0 else np.array([], int)
    mid = np.flatnonzero((idx >= 0.3 * T) & (idx <= 0.7 * T))
    return last, neg, mid


def analyze(args):
    from scipy.stats import spearmanr

    t0 = time.time()
    D = Data(args)
    bank = Bank(D)
    res = {"protocol": "wm_train/G6A_PROTOCOL.md", "device": P.DEV, "rest_tol": round(D.tol, 4), "n_train_end_not_at_rest": 0,
           "cycles_hist": {str(k): int(v) for k, v in zip(*np.unique([D.cycles[e] for e in D.train + D.val], return_counts=True))},
           "bank_per_task": {t: int(sum(1 for x in bank.task if x == t)) for t in set(bank.task)}}
    res["n_train_end_not_at_rest"] = int(sum(not D.at_rest(e, D.idx(e))[-1] for e in D.train))
    res["n_val_end_not_at_rest"] = int(sum(not D.at_rest(e, D.idx(e))[-1] for e in D.val))
    print("rest tol", D.tol, "cycles", res["cycles_hist"], "train ends not at rest", res["n_train_end_not_at_rest"], "val", res["n_val_end_not_at_rest"], flush=True)

    # ---- train: choose N per (grid, cams) and the best W2 variant by leave-one-episode-out T2 AUROC
    all_sel = {(g, c): None for g in (8, 16) for c in ("top", "both")}
    cand = {}
    for g in (8, 16):
        for cams in ("top", "both"):
            for N in GRID_N[g]:
                cand[f"W2-{g}x{g}/{cams}/work{N}"] = ("w2", g, cams, N)
    base = variants({})
    qs, kind = [], []
    for e in D.train:
        idx = D.idx(e)
        neg = np.flatnonzero(D.at_rest(e, idx) & (idx < D.c1[e])) if D.c1[e] > 0 else np.array([], int)
        qs.append((e, [len(idx) - 1] + list(neg)))
        kind += [1] + [0] * len(neg)
    kind = np.array(kind)
    q_eps = np.array([e for e, pos in qs for _ in pos])
    tr = distances(D, bank, qs, {**base, **cand})
    train_auc = {n: P.auroc(kind == 1, -d[0]) for n, d in tr.items()}      # LOEO: queries exclude their own bank entry
    sel_n = {}
    for g in (8, 16):
        for cams in ("top", "both"):
            best = max(GRID_N[g], key=lambda N: (round(train_auc[f"W2-{g}x{g}/{cams}/work{N}"], 6), -N))
            sel_n[(g, cams)] = best
    specs = variants(sel_n)
    res["train_loeo_t2_auroc_all_candidates"] = {n: round(float(v), 4) for n, v in train_auc.items()}
    res["selected_N"] = {f"{g}x{g}/{c}": int(n) for (g, c), n in sel_n.items()}
    res["train_loeo_t2_auroc"] = {n: round(float(train_auc[n]), 4) for n in specs}
    best_w2 = max((n for n in specs if n.startswith("W2")), key=lambda n: train_auc[n])
    res["best_w2_variant_by_train"] = best_w2
    thr = {n: float(np.percentile(tr[n][0][kind == 1], 95)) for n in specs}
    thr3 = {n: float(np.percentile(tr[n][1][kind == 1], 95)) for n in specs}
    res["done_threshold_nearest"] = {n: round(v, 4) for n, v in thr.items()}
    print("train done; selected N", res["selected_N"], "best", best_w2, f"({time.time() - t0:.0f}s)", flush=True)

    # ---- val: distances for every frame, T1 and T2
    qv = [(e, list(range(len(D.idx(e))))) for e in D.val]
    dv = distances(D, bank, qv, specs)
    lens = [len(D.idx(e)) for e in D.val]
    offs = np.concatenate([[0], np.cumsum(lens)])
    series = {n: {e: (dv[n][0][offs[i]:offs[i + 1]], dv[n][1][offs[i]:offs[i + 1]]) for i, e in enumerate(D.val)} for n in specs}
    res["variants"] = {}
    for n in specs:
        for which, name in ((0, "nearest"), (1, "nearest3")):
            sp = {e: series[n][e][which] for e in D.val}
            rho = {e: float(spearmanr(sp[e], np.arange(len(sp[e])))[0]) for e in D.val}
            sets = {e: t2_sets(D, e, None) for e in D.val}

            def auc(es, secondary=False, sp=sp, sets=sets):
                y, s = [], []
                for e in es:
                    last, neg, mid = sets[e]
                    other = mid if secondary else neg
                    y += [1] + [0] * len(other)
                    s += [-sp[e][last]] + [-sp[e][i] for i in other]
                return P.auroc(np.array(y, bool), np.array(s))
            v = {"t1_median_spearman": round(float(np.median(list(rho.values()))), 4),
                 "t1_ci95": boot_ci(lambda es: float(np.median([rho[e] for e in es])), D.val),
                 "t2_auroc": round(auc(D.val), 4), "t2_ci95": boot_ci(auc, D.val),
                 "t2_secondary_midepisode_auroc": round(auc(D.val, True), 4), "t2_secondary_ci95": boot_ci(lambda es: auc(es, True), D.val)}
            t = (thr if which == 0 else thr3)[n]
            pos = [sp[e][sets[e][0]] <= t for e in D.val]
            neg_ = np.concatenate([[sp[e][i] <= t for i in sets[e][1]] for e in D.val]) if any(len(sets[e][1]) for e in D.val) else np.array([])
            v.update({"done_threshold": round(t, 4), "tpr_at_threshold": round(float(np.mean(pos)), 4), "fpr_at_threshold": round(float(np.mean(neg_)), 4) if len(neg_) else None,
                      "n_neg_frames": int(sum(len(sets[e][1]) for e in D.val))})
            res["variants"].setdefault(n, {})[which and "nearest3" or "nearest"] = v
        print(n, res["variants"][n]["nearest"]["t1_median_spearman"], res["variants"][n]["nearest"]["t2_auroc"], flush=True)
    print(f"val done ({time.time() - t0:.0f}s)", flush=True)

    # ---- stall rule on the best W2 variant: fit on train successes (leave-one-episode-out bank), freeze, indicative on val
    sp1 = {best_w2: specs[best_w2]}
    qt = [(e, list(range(len(D.idx(e))))) for e in D.train]
    dt_ = distances(D, bank, qt, sp1)[best_w2][0]
    toffs = np.concatenate([[0], np.cumsum([len(D.idx(e)) for e in D.train])])
    tseries = {e: dt_[toffs[i]:toffs[i + 1]] for i, e in enumerate(D.train)}
    d_ref = float(np.median([tseries[e][0] for e in D.train]))

    def smooth(d):
        k = 5
        pad = np.pad(d, (k // 2, k // 2), mode="edge")
        return np.array([np.median(pad[i:i + k]) for i in range(len(d))])

    def stalls(e, d, W, eps, stride):
        idx = D.idx(e)[::stride] if stride > 1 else D.idx(e)
        d = smooth(d[::stride] if stride > 1 else d)
        rest = D.at_rest(e, idx)
        w = int(round(W / (STRIDE_S / FPS)))
        move = np.flatnonzero(~rest)
        if not len(move):
            return np.zeros(len(d), bool)
        flags = np.zeros(len(d), bool)
        for i in range(max(move[0] + w, w), len(d) - int(round(1.0 / (STRIDE_S / FPS)))):
            flags[i] = (not rest[i]) and (d[i - w] - d[i] < eps * d_ref)
        return flags
    fit = {}
    for W in W_GRID:
        for f in EPS_GRID:
            fit[(W, f)] = np.mean([stalls(e, tseries[e], W, f, 1).any() for e in D.train])
    zero = [(W, f) for (W, f), r in fit.items() if r == 0]
    if zero:
        W_sel = min(w for w, _ in zero)
        f_sel = max(f for w, f in zero if w == W_sel)
    else:
        cand_ = [(W, f) for (W, f) in fit if W >= 5.0]
        best_r = min(fit[c] for c in cand_)
        W_sel = min(w for w, f in cand_ if fit[(w, f)] == best_r)
        f_sel = max(f for w, f in cand_ if w == W_sel and fit[(w, f)] == best_r)
    res["stall"] = {"variant": best_w2, "d_ref": round(d_ref, 4), "W_s": W_sel, "eps_frac_of_dref": f_sel, "train_false_stall_rate": float(fit[(W_sel, f_sel)]),
                    "train_grid": {f"W{W}_eps{f}": round(float(r), 4) for (W, f), r in fit.items()}}
    vs = {e: series[best_w2][e][0][::STRIDE_S] for e in D.val}
    vflags = {}
    for e in D.val:
        idx = D.idx(e)[::STRIDE_S]
        d = smooth(vs[e])
        rest = D.at_rest(e, idx)
        w = int(round(W_sel / (STRIDE_S / FPS)))
        move = np.flatnonzero(~rest)
        fl = np.zeros(len(d), bool)
        if len(move):
            for i in range(max(move[0] + w, w), len(d) - int(round(1.0 / (STRIDE_S / FPS)))):
                fl[i] = (not rest[i]) and (d[i - w] - d[i] < f_sel * d_ref)
        vflags[e] = fl
    res["stall"]["val_false_stall_rate_indicative"] = round(float(np.mean([vflags[e].any() for e in D.val])), 4)
    print("stall", res["stall"]["W_s"], res["stall"]["eps_frac_of_dref"], "train fp", res["stall"]["train_false_stall_rate"], "val fp", res["stall"]["val_false_stall_rate_indicative"], flush=True)

    res["t3_t4"] = "pending: the 81 gate rollouts are not on this machine (G6A_PROTOCOL.md); indicative only when run (8 failures with video)"
    res["t5"] = "see report (Orin)"
    res["seconds"] = round(time.time() - t0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    plots(args.out.parent, D, series, best_w2, thr[best_w2], vflags)
    print("wrote", args.out)


def plots(out_dir, D, series, name, thr, vflags):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for e in D.val:
        d = series[name][e][0]
        t = np.arange(len(d)) / FPS
        ax[0].plot(t, d, lw=0.7, alpha=0.7)
        fl = np.flatnonzero(vflags[e])
        if len(fl):
            ax[0].plot(fl * STRIDE_S / FPS, d[fl * STRIDE_S], "r.", ms=3)
        ax[1].plot(np.linspace(0, 1, len(d)), d, lw=0.7, alpha=0.7)
    for a_ in ax:
        a_.axhline(thr, color="k", ls="--", lw=1, label="done threshold (train p95)")
    ax[0].set_xlabel("time (s)")
    ax[1].set_xlabel("normalized time")
    ax[0].set_ylabel(f"distance to nearest bank end frame ({name})")
    ax[0].set_title("held-out successful demos (red: stall flags)")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "g6a_distance_vs_time_val.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
