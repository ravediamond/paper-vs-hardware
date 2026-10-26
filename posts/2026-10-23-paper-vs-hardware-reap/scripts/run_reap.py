#!/usr/bin/env python3
"""REAP_PROTOCOL.md: W1 L2 surprise vs a ReAP-style risk head vs elapsed time as runtime failure monitors, 5-fold by
episode x 5 seeds on W1-A's 50 episodes (primary) and all 69 (secondary). Onsets: VLM-marked (protocol deviation of
2026-10-23). W1 latents and surprise come from W1-A's own code (run_w1_offline_screening.encode_episode / anchor_curves).

    W1=1 shared/gate/jetson_run.sh --python scripts/run_reap.py --w1 <W1 weights_final.pt> \\
        --onsets shared/gate/data/2026-10-23_onsets_vlm/onsets.csv --out shared/gate/data/<date>_reap/monitors.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_w1_offline_screening as A  # noqa: E402
from so101_harness.episodes import EpisodeDataset, download  # noqa: E402

FPS, H_RISK, TRAIN_STRIDE, FI = 30, 30, 5, 1 / 20


def risk_features(z, acts_n, a, end):
    ix = np.clip(np.arange(a, a + H_RISK), 0, end - 1)
    return np.concatenate([z[a], acts_n[ix].reshape(-1), [a / FPS]]).astype(np.float32)


def fit_predict(kind, Xtr, ytr, Xte, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if kind == "logistic":
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    import torch

    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    X, y = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    pw = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1))
    for _ in range(300):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(net(X)[:, 0], y, pos_weight=pw).backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte))[:, 0]).numpy()


def tau_from(train_success_scores):
    s = np.sort(np.asarray(train_success_scores))[::-1]
    return float(s[int(np.floor(FI * len(s)))]) if len(s) else np.inf   # ≤ 1/20 of them strictly above


def evaluate(eps, info, seeds=5):
    """info[ep] = {y (1 = failure), onset_s, anchors: [(decision_s, {monitor: score})], train_rows: (X, y)}."""
    from sklearn.model_selection import StratifiedKFold

    mons = ("surprise", "risk_logistic", "risk_mlp", "elapsed")
    per_seed = {m: [] for m in mons}
    y_all = np.array([info[e]["y"] for e in eps])
    for seed in range(seeds):
        held = {m: {} for m in mons}            # ep -> (episode score, first intervention s or None)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(eps)), y_all):
            tr_eps, te_eps = [eps[i] for i in tr], [eps[i] for i in te]
            Xtr = np.concatenate([info[e]["train_X"] for e in tr_eps]); ytr = np.concatenate([info[e]["train_y"] for e in tr_eps])
            scores = {m: {} for m in mons}       # ep -> per-anchor scores (train and held-out episodes)
            for e in tr_eps + te_eps:
                scores["surprise"][e] = info[e]["surprise"]; scores["elapsed"][e] = info[e]["elapsed"]
            for kind in ("logistic", "mlp"):
                Xall = np.concatenate([info[e]["anchor_X"] for e in tr_eps + te_eps])
                p = fit_predict(kind, Xtr, ytr, Xall, seed)
                k = 0
                for e in tr_eps + te_eps:
                    n = len(info[e]["anchor_X"]); scores[f"risk_{kind}"][e] = p[k:k + n]; k += n
            for m in mons:
                tau = tau_from([scores[m][e].max() for e in tr_eps if info[e]["y"] == 0])
                for e in te_eps:
                    s = scores[m][e]
                    fire = np.flatnonzero(s > tau)
                    held[m][e] = (float(s.max()), float(info[e]["t"][m][fire[0]]) if len(fire) else None)
        for m in mons:
            fails = [e for e in eps if info[e]["y"] == 1]; succ = [e for e in eps if info[e]["y"] == 0]
            caught = [e for e in fails if held[m][e][1] is not None]
            ttw = [info[e]["onset_s"] - held[m][e][1] for e in caught if info[e]["onset_s"] is not None]
            per_seed[m].append({"caught": len(caught), "failures": len(fails),
                                "false_interventions": sum(held[m][e][1] is not None for e in succ), "successes": len(succ),
                                "auroc": A.auroc([info[e]["y"] for e in eps], [held[m][e][0] for e in eps]),
                                "time_to_warning_s_median": float(np.median(ttw)) if ttw else None})
    out = {}
    for m in mons:
        r = per_seed[m]
        med = {k: float(np.median([x[k] for x in r if x[k] is not None])) if any(x[k] is not None for x in r) else None
               for k in ("caught", "false_interventions", "auroc", "time_to_warning_s_median")}
        med.update({"failures": r[0]["failures"], "successes": r[0]["successes"], "per_seed": r,
                    "pass": bool(med["caught"] >= 5 and med["false_interventions"] <= r[0]["successes"] / 20)})
        out[m] = med
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w1", required=True)
    ap.add_argument("--onsets", type=Path, required=True)
    ap.add_argument("--repo-id", default="ravediamond/so101-bowls-policy-rollouts")
    ap.add_argument("--outcomes", type=Path, default=ROOT / "data/rollouts/so101-bowls-policy-rollouts_labels.csv")
    ap.add_argument("--w1a", type=Path, default=ROOT / "shared/gate/data/2026-10-19_w1_screening/w1a_summary.json")
    ap.add_argument("--latents", type=Path, default=ROOT / "outputs/reap/latents")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from so101_harness.w1_live import W1_ACTION_MEAN, W1_ACTION_STD, W1LiveMonitor

    ds = EpisodeDataset(download(args.repo_id, with_videos=True))
    oc = pd.read_csv(args.outcomes); oc = oc[oc.outcome.isin(["success", "failure"])]
    on = pd.read_csv(args.onsets).groupby("episode_index").last()
    g16 = {e["episode_index"] for e in json.loads(args.w1a.read_text())["episodes"] if e["in_g16_set"]}
    mon = W1LiveMonitor.from_checkpoint(args.w1)
    args.latents.mkdir(parents=True, exist_ok=True)
    am, asd = np.array(W1_ACTION_MEAN, np.float32), np.array(W1_ACTION_STD, np.float32)
    info = {}
    for r in oc.itertuples():
        ep = int(r.episode_index)
        end = len(ds.states(ep)) if int(r.reset_start) < 0 else min(int(r.reset_start), len(ds.states(ep)))
        f = args.latents / f"ep{ep:03d}.npy"
        z = np.load(f) if f.exists() else A.encode_episode(mon, ds, ep)
        np.save(f, z)
        acts = ds.actions(ep); acts_n = (acts - am) / asd
        curves = A.anchor_curves(mon, z, acts, end)
        anchors = [c["anchor"] for c in curves]
        y = int(r.outcome == "failure")
        onset = int(on.loc[ep, "onset_frame"]) if y and ep in on.index and int(on.loc[ep, "onset_frame"]) >= 0 else None
        tr = [a for a in range(2, end - 1, TRAIN_STRIDE) if not np.isnan(z[a]).any()]
        info[ep] = {"y": y, "onset_s": None if onset is None else onset / FPS,
                    "surprise": np.array([np.mean([c["l2"][k] for k in A.CHECK_K]) for c in curves]),
                    "elapsed": np.array(anchors, float) / FPS,
                    "anchor_X": np.stack([risk_features(z, acts_n, a, end) for a in anchors]),
                    "train_X": np.stack([risk_features(z, acts_n, a, end) for a in tr]),
                    "train_y": np.array([int(y and onset is not None and onset <= a + H_RISK) for a in tr]),
                    "t": {"surprise": (np.array(anchors) + A.HORIZON) / FPS, "elapsed": np.array(anchors) / FPS,
                          "risk_logistic": np.array(anchors) / FPS, "risk_mlp": np.array(anchors) / FPS}}
        print(f"ep {ep:3d} {'fail' if y else 'ok  '} anchors {len(anchors)} onset {onset}", flush=True)
    primary = sorted(e for e in info if e in g16)
    res = {"protocol": "projects/vla-jepa-world-model/jetson/REAP_PROTOCOL.md", "onsets": str(args.onsets),
           "onset_source": "VLM (gemini-3.8-flash), deviation 2026-10-23", "bar": "caught >= 5 (of 8) and false interventions <= 1/20, median over 5 seeds",
           "primary_50": evaluate(primary, info), "secondary_69": evaluate(sorted(info), info)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=float))
    for setn in ("primary_50", "secondary_69"):
        print(setn)
        for m, v in res[setn].items():
            print(f"  {m:14s} caught {v['caught']}/{v['failures']}  FI {v['false_interventions']}/{v['successes']}  "
                  f"AUROC {v['auroc']}  ttw {v['time_to_warning_s_median']}  pass {v['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
