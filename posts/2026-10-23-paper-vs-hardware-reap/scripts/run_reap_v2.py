#!/usr/bin/env python3
"""REAP_PROTOCOL.md version 2: ramp-target risk head on W1 (no onsets, no time feature), common-horizon truncation per
task, true / shuffled / zero action controls at evaluation, bTWA beside the 5/8-at-1/20 bar, and VLA-Corrector's
parameter-free GRU as the floor baseline. W1 latents are v1's cache (W1-A's encode); RGB features are computed here.

    W1=1 shared/gate/jetson_run.sh --python scripts/run_reap_v2.py --out shared/gate/data/2026-10-22_reap/monitors_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_w1_offline_screening as A  # noqa: E402
from so101_harness.episodes import TOP, WRIST, EpisodeDataset, download  # noqa: E402

H, TRAIN_STRIDE, SCORE_STRIDE, FI, WIN, WSTEP = 30, 5, 25, 1 / 20, 8, 2
CONTROLS = ("true", "shuffled", "zero")


def rgb_features(ds, ep):
    """(T, 108): per camera 112 x 112, 3x4x4 average-pooled RGB + per-channel mean + std (54 per view)."""
    out = []
    for cam in (TOP, WRIST):
        fr = []
        for _, im in ds.iter_frames(ep, cam, size=(112, 112)):
            x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
            pooled = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4)).reshape(-1)
            fr.append(torch.cat([pooled, x.mean((1, 2)), x.std((1, 2))]).numpy())
        out.append(np.stack(fr))
    n = min(len(o) for o in out)
    return np.concatenate([o[:n] for o in out], 1).astype(np.float32)


def bce_fit(model, X, y, w, epochs, lr, wd, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    X = [torch.as_tensor(x) for x in X] if isinstance(X, tuple) else torch.as_tensor(X)
    y, w = torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(w, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(*X) if isinstance(X, list) else model(X)
        torch.nn.functional.binary_cross_entropy_with_logits(out[:, 0], y, weight=w).backward()
        opt.step()
    return model


class GRUFloor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vis, self.sa = torch.nn.Linear(108, 64), torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.ReLU())
        self.gru, self.head = torch.nn.GRU(96, 64, batch_first=True), torch.nn.Linear(64, 1)

    def forward(self, v, sa):
        h, _ = self.gru(torch.cat([self.vis(v), self.sa(sa)], -1))
        return self.head(h[:, -1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="ravediamond/so101-bowls-policy-rollouts")
    ap.add_argument("--outcomes", type=Path, default=ROOT / "data/rollouts/so101-bowls-policy-rollouts_labels.csv")
    ap.add_argument("--w1a", type=Path, default=ROOT / "shared/gate/data/2026-10-19_w1_screening/w1a_summary.json")
    ap.add_argument("--latents", type=Path, default=ROOT / "outputs/reap/latents")
    ap.add_argument("--rgb", type=Path, default=ROOT / "outputs/reap/rgbfeat")
    ap.add_argument("--onsets", type=Path, default=ROOT / "shared/gate/data/2026-10-23_onsets_vlm/onsets.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from so101_harness.w1_live import W1_ACTION_MEAN, W1_ACTION_STD
    from sklearn.model_selection import StratifiedKFold

    ds = EpisodeDataset(download(args.repo_id, with_videos=True))
    oc = pd.read_csv(args.outcomes); oc = oc[oc.outcome.isin(["success", "failure"])]
    g16 = {e["episode_index"] for e in json.loads(args.w1a.read_text())["episodes"] if e["in_g16_set"]}
    on = pd.read_csv(args.onsets).groupby("episode_index").last()
    am, asd = np.array(W1_ACTION_MEAN, np.float32), np.array(W1_ACTION_STD, np.float32)
    args.rgb.mkdir(parents=True, exist_ok=True)
    E = {}
    for r in oc.itertuples():
        ep = int(r.episode_index)
        end = len(ds.states(ep)) if int(r.reset_start) < 0 else min(int(r.reset_start), len(ds.states(ep)))
        f = args.rgb / f"ep{ep:03d}.npy"
        rgb = np.load(f) if f.exists() else rgb_features(ds, ep)
        np.save(f, rgb)
        E[ep] = {"y": int(r.outcome == "failure"), "task": ds.task(ep), "end": end, "z": np.load(args.latents / f"ep{ep:03d}.npy"),
                 "a": ((ds.actions(ep) - am) / asd).astype(np.float32), "s": ((ds.states(ep) - am) / asd).astype(np.float32),
                 "rgb": rgb, "onset": int(on.loc[ep, "onset_frame"]) if ep in on.index else None}
        print(f"ep {ep}", flush=True)

    def run(eps, seeds=5):
        tasks = {}
        for e in eps:
            tasks.setdefault(E[e]["task"], []).append(e)
        Tk = {t: min(min(E[e]["end"], len(E[e]["rgb"]), len(E[e]["z"])) for e in es) for t, es in tasks.items()}
        T = {e: Tk[E[e]["task"]] for e in eps}
        rng0 = np.random.default_rng(0)
        partner = {e: rng0.choice([x for x in tasks[E[e]["task"]] if x != e]) if len(tasks[E[e]["task"]]) > 1 else e for e in eps}

        def chunk(e, t, ctl):
            src = partner[e] if ctl == "shuffled" else e
            ix = np.clip(np.arange(t, t + H), 0, T[e] - 1)
            return np.zeros(H * 6, np.float32) if ctl == "zero" else E[src]["a"][ix].reshape(-1)

        def head_X(e, ts, ctl):
            return np.stack([np.concatenate([E[e]["z"][t], chunk(e, t, ctl)]) for t in ts]).astype(np.float32)

        def gru_X(e, ts, ctl):
            src = partner[e] if ctl == "shuffled" else e
            V, SA = [], []
            for t in ts:
                ix = np.clip(np.arange(t - (WIN - 1) * WSTEP, t + 1, WSTEP), 0, T[e] - 1)
                act = np.zeros((WIN, 6), np.float32) if ctl == "zero" else E[src]["a"][ix]
                V.append(E[e]["rgb"][ix]); SA.append(np.concatenate([E[e]["s"][ix], act], 1))
            return np.stack(V), np.stack(SA)

        def ramp(e, ts):
            return np.array([0.0 if E[e]["y"] == 0 else t / (T[e] - 1) for t in ts], np.float32)

        valid = lambda e, t: not np.isnan(E[e]["z"][t]).any()  # noqa: E731
        tr_ts = {e: [t for t in range(2, T[e], TRAIN_STRIDE) if valid(e, t)] for e in eps}
        sc_ts = {e: [t for t in range(2, T[e], SCORE_STRIDE) if valid(e, t)] for e in eps}
        y_ep = np.array([E[e]["y"] for e in eps])
        mons = [f"{m}:{c}" for m in ("risk_logistic", "risk_mlp", "gru_floor") for c in CONTROLS]
        per = {m: [] for m in mons}
        for seed in range(seeds):
            held = {m: {} for m in mons}
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(eps)), y_ep):
                tre, tee = [eps[i] for i in tr], [eps[i] for i in te]
                nf = sum(E[e]["y"] for e in tre); ns = len(tre) - nf
                wcls = {1: len(tre) / (2 * max(nf, 1)), 0: len(tre) / (2 * max(ns, 1))}
                X = np.concatenate([head_X(e, tr_ts[e], "true") for e in tre]); y = np.concatenate([ramp(e, tr_ts[e]) for e in tre])
                w = np.concatenate([np.full(len(tr_ts[e]), wcls[E[e]["y"]], np.float32) for e in tre])
                mu, sd = X.mean(0), X.std(0) + 1e-6
                lin = bce_fit(torch.nn.Linear(X.shape[1], 1), (X - mu) / sd, y, w, 400, 1e-2, 1e-3, seed)
                mlp = bce_fit(torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)),
                              (X - mu) / sd, y, w, 300, 1e-3, 1e-4, seed)
                Vs, SAs = zip(*[gru_X(e, tr_ts[e], "true") for e in tre])
                gru = bce_fit(GRUFloor(), (np.concatenate(Vs), np.concatenate(SAs)), y, w, 30, 1e-3, 0.0, seed)
                with torch.no_grad():
                    def score(m, e, ts, ctl):
                        if m == "gru_floor":
                            v, sa = gru_X(e, ts, ctl); return torch.sigmoid(gru(torch.as_tensor(v), torch.as_tensor(sa))[:, 0]).numpy()
                        net = lin if m == "risk_logistic" else mlp
                        return torch.sigmoid(net(torch.as_tensor((head_X(e, ts, ctl) - mu) / sd))[:, 0]).numpy()
                    for m in ("risk_logistic", "risk_mlp", "gru_floor"):
                        tau = np.sort([score(m, e, sc_ts[e], "true").max() for e in tre if E[e]["y"] == 0])[::-1]
                        tau = float(tau[int(np.floor(FI * len(tau)))])
                        for c in CONTROLS:
                            for e in tee:
                                s = score(m, e, sc_ts[e], c); fire = np.flatnonzero(s > tau)
                                held[f"{m}:{c}"][e] = (float(s.max()), int(sc_ts[e][fire[0]]) if len(fire) else None)
            for m in mons:
                fails = [e for e in eps if E[e]["y"] == 1]; succ = [e for e in eps if E[e]["y"] == 0]
                det = [e for e in fails if held[m][e][1] is not None]
                tn = sum(held[m][e][1] is None for e in succ)
                btwa = 0.5 * (sum(1 - held[m][e][1] / T[e] for e in det) / len(fails) + tn / len(succ))
                ttw = [(E[e]["onset"] - held[m][e][1]) / 30 for e in det if E[e]["onset"] is not None]
                per[m].append({"caught": len(det), "fi": len(succ) - tn, "auroc": A.auroc(list(y_ep), [held[m][e][0] for e in eps]),
                               "btwa": btwa, "ttw_vs_gemini_s": float(np.median(ttw)) if ttw else None})
        out = {"T_common": Tk, "failures": int(y_ep.sum()), "successes": int(len(eps) - y_ep.sum())}
        for m in mons:
            r = per[m]
            out[m] = {k: float(np.median([x[k] for x in r if x[k] is not None])) if any(x[k] is not None for x in r) else None
                      for k in ("caught", "fi", "auroc", "btwa", "ttw_vs_gemini_s")}
            out[m]["per_seed"] = r
            out[m]["pass_bar"] = bool(out[m]["caught"] >= 5 and out[m]["fi"] <= out["successes"] / 20)   # 5/8 at <= 1/20
        for m in ("risk_logistic", "risk_mlp", "gru_floor"):
            out[m + ":action_conditioned"] = bool(out[f"{m}:true"]["auroc"] >= out[f"{m}:shuffled"]["auroc"] + 0.05)
        return out

    primary = sorted(e for e in E if e in g16)
    res = {"protocol": "projects/vla-jepa-world-model/jetson/REAP_PROTOCOL.md version 2", "primary_50": run(primary), "secondary_69": run(sorted(E))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=float))
    for setn in ("primary_50", "secondary_69"):
        r = res[setn]; print(setn, "T_common", r["T_common"], "fail/succ", r["failures"], r["successes"])
        for m in [k for k in r if ":" in k and not k.endswith("conditioned")]:
            v = r[m]; print(f"  {m:24s} caught {v['caught']}  FI {v['fi']}  AUROC {v['auroc']:.3f}  bTWA {v['btwa']:.3f}  pass {v['pass_bar']}")
        print("  action-conditioned:", {k: r[k] for k in r if k.endswith("conditioned")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
