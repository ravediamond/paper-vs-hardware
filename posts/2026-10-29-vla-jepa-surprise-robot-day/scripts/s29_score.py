#!/usr/bin/env python3
"""NEXT_ANGLES §29 step 2 scorer, frozen before rollout 1 (WS-018 step 1.2).

Five scorers on the same episodes, scored exactly as the frozen §29 bar says:
- vla_jepa:      VLA-JEPA's own world-model L1 per chunk (`vla_jepa_surprise.py replay`), running max over the
                 episode, known at t + 28 frames;
- w1_surprise:   W1 L2 surprise per anchor (W1-A / REAP code, mean over k = 15, 20, 25), known at anchor + 25;
- risk_logistic: §2's logistic risk head on W1 latent + 30 normalised commanded actions + step (REAP features,
                 refit on each training fold), at the anchor;
- elapsed:       seconds since the start;
- stall:         joint-velocity stall rule, fixed before any robot-day data: at each frame, the seconds since the
                 last frame where any joint moved faster than the stall speed (state, 0.5 s moving median of
                 |Δstate| x fps). The stall speed is derived, not chosen (owner, 2026-10-26): 1.5 x the p99 of that
                 smoothed max-joint speed over the resting segments of the 44
                 successes in the published rollouts, resting = each success's final still run (owner, 2026-10-26, option b: the
                 trailing frames whose smoothed max-joint speed is ≤ one encoder count per frame; the first rule, the
                 last 1 s, gave 51 deg/s because 6 successes end while moving); `stall-calibrate` computes it into
                 stall_calibration.json, which
                 `baselines` reads. Only the threshold τ (seconds) is learned, by the same 1/20 rule as every scorer.

Evaluation (§29): 5-fold by episode (stratified; fewer folds only if there are fewer than 5 failures or successes,
reported) x 5 seeds. Per fold, τ = the value that ≤ 1/20 of the training-fold successes' episode maxima exceed. A held-out
failure is caught if its score crosses τ strictly before the owner-marked onset; a held-out success with any crossing
is a false intervention. AUROC on the held-out episode maxima, median over seeds; 95 % CI from 2 000 episode-bootstrap
resamples (each resample: median over seeds of the AUROC on the resampled episodes; rng seed 29). Verdict branches
as written in the bar, applied mechanically by `verdict()`.

    # 1. baseline traces, in the W1 image (W1 latents cached next to --out)
    W1=1 shared/gate/jetson_run.sh --python scripts/s29_score.py baselines --w1 <weights_final.pt> \\
        --root <rollouts root or repo id> --labels <labels.csv> --out <dir>/baselines
    # 2. VLA-JEPA traces, in the VLA-JEPA image (`--episodes` = the robot-day episodes)
    python projects/vla-jepa-world-model/jetson/vla_jepa_surprise.py replay --ckpt <snapshot> --rollouts <root> \\
        --outcomes <labels.csv> --episodes <list> --out <dir>/vla_jepa
    # 3. score (CPU, numpy + scikit-learn)
    python scripts/s29_score.py score --baselines <dir>/baselines --vla-jepa <dir>/vla_jepa/replay.csv \\
        --labels <labels.csv> --out <dir>/s29_result.json

labels.csv: episode_index, outcome (success / failure), failure_mode, onset_frame (failures; -1 if none),
reset_start (-1 if none). An onset from a second file can be merged with --onsets (last row per episode wins).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FPS = 30
STALL_SMOOTH_S = 0.5
STALL_FACTOR, STALL_PCT = 1.5, 99
ONE_COUNT_DEG_S = 360 / 4096 * FPS   # one STS3215 encoder count per frame = 2.637 deg/s: the state's resolution
STALL_CAL = ROOT / "shared/gate/data/2026-10-26_s29_robot_day/stall_calibration.json"
SEEDS, BOOT, BOOT_SEED = 5, 2000, 29
SCORERS = ("vla_jepa", "w1_surprise", "risk_logistic", "elapsed", "stall")
BASELINES = SCORERS[1:]
VLA_JEPA_KNOWN_AT = 28     # frames after t (the training window's last frame)


def load_labels(path, onsets=None):
    lab = pd.read_csv(path)
    lab = lab[lab.outcome.isin(["success", "failure"])].copy()
    if "onset_frame" not in lab:
        lab["onset_frame"] = -1
    if onsets is not None:
        on = pd.read_csv(onsets).groupby("episode_index").last()["onset_frame"]
        lab["onset_frame"] = [int(on.get(e, o)) for e, o in zip(lab.episode_index, lab.onset_frame)]
    return lab.set_index("episode_index")


def smoothed_speed(states):
    """Per frame, per joint: 0.5 s trailing moving median of |Δstate| x fps (deg/s)."""
    v = np.abs(np.diff(states, axis=0, prepend=states[:1])) * FPS
    w = max(1, int(STALL_SMOOTH_S * FPS))
    return np.stack([np.median(v[max(0, i - w + 1): i + 1], axis=0) for i in range(len(v))])


def stall_trace(states, deg_s):
    """Seconds since any joint last moved faster than deg_s, per frame."""
    moving = (smoothed_speed(states) > deg_s).any(axis=1)
    out, run = np.zeros(len(v)), 0
    for i, m in enumerate(moving):
        run = 0 if m else run + 1
        out[i] = run / FPS
    return out


def cmd_baselines(a):
    import run_reap as R
    import run_w1_offline_screening as A
    from so101_harness.episodes import EpisodeDataset, download
    from so101_harness.w1_live import W1_ACTION_MEAN, W1_ACTION_STD, W1LiveMonitor

    root = Path(a.root) if Path(a.root).exists() else download(a.root, with_videos=True)
    ds = EpisodeDataset(root)
    lab = load_labels(a.labels, a.onsets)
    stall_deg_s = json.loads(STALL_CAL.read_text())["stall_deg_s"]
    mon = W1LiveMonitor.from_checkpoint(a.w1)
    am, asd = np.array(W1_ACTION_MEAN, np.float32), np.array(W1_ACTION_STD, np.float32)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lat = Path(a.latents) if a.latents else out.parent / "latents"   # per dataset: episode ids repeat across datasets
    lat.mkdir(parents=True, exist_ok=True)
    for ep, r in lab.iterrows():
        ep = int(ep)
        T = len(ds.states(ep)); end = T if int(r.reset_start) < 0 else min(int(r.reset_start), T)
        f = lat / f"ep{ep:03d}.npy"
        z = np.load(f) if f.exists() else A.encode_episode(mon, ds, ep)
        np.save(f, z)
        acts = ds.actions(ep); acts_n = (acts - am) / asd
        curves = A.anchor_curves(mon, z, acts, end)
        anchors = np.array([c["anchor"] for c in curves])
        y = int(r.outcome == "failure"); onset = int(r.onset_frame) if y and int(r.onset_frame) >= 0 else None
        tr = [t for t in range(2, end - 1, R.TRAIN_STRIDE) if not np.isnan(z[t]).any()]
        st = stall_trace(ds.states(ep)[:end], stall_deg_s)
        np.savez(out / f"ep{ep:03d}.npz", y=y, onset=-1 if onset is None else onset, end=end,
                 w1_surprise=np.array([np.mean([c["l2"][k] for k in A.CHECK_K]) for c in curves]),
                 w1_surprise_t=(anchors + A.HORIZON) / FPS,
                 anchor_t=anchors / FPS,
                 anchor_X=np.stack([R.risk_features(z, acts_n, t, end) for t in anchors]),
                 train_X=np.stack([R.risk_features(z, acts_n, t, end) for t in tr]),
                 train_y=np.array([int(y and onset is not None and onset <= t + R.H_RISK) for t in tr]),
                 stall=st, stall_t=np.arange(end) / FPS)
        print(f"ep {ep:3d} {'fail' if y else 'ok  '} anchors {len(anchors)} onset {onset}", flush=True)


def cmd_stall_calibrate(a):
    from so101_harness.episodes import EpisodeDataset, download

    ds = EpisodeDataset(download(a.repo_id, with_videos=False))
    lab = load_labels(a.labels)
    rest = []
    for ep, r in lab[lab.outcome == "success"].iterrows():
        st = ds.states(int(ep)); end = len(st) if int(r.reset_start) < 0 else min(int(r.reset_start), len(st))
        sp = smoothed_speed(st[:end]).max(axis=1)
        moving = np.flatnonzero(sp > ONE_COUNT_DEG_S + 1e-6)
        rest.append(sp[(moving[-1] + 1) if len(moving) else 0:])          # the final still run (may be empty)
    n_still = int(sum(len(r) > 0 for r in rest))
    rest = np.concatenate(rest)
    p = float(np.percentile(rest, STALL_PCT))
    res = {"rule": f"stall speed = {STALL_FACTOR} x p{STALL_PCT} of the smoothed max-joint speed over each success's final "
                   f"still run (trailing frames <= one encoder count per frame, {ONE_COUNT_DEG_S:.3f} deg/s), before end or "
                   f"reset; smoothing {STALL_SMOOTH_S} s trailing median (owner 2026-10-26, option b)",
           "successes_with_a_still_run": n_still,
           "source": a.repo_id, "labels": a.labels, "successes": int((lab.outcome == "success").sum()),
           "rest_frames": int(len(rest)), "p50_deg_s": float(np.percentile(rest, 50)), f"p{STALL_PCT}_deg_s": p,
           "stall_deg_s": STALL_FACTOR * p}
    STALL_CAL.parent.mkdir(parents=True, exist_ok=True)
    STALL_CAL.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def load_traces(baselines, vla_jepa_csv, lab):
    vj = pd.read_csv(vla_jepa_csv) if vla_jepa_csv else None
    info = {}
    for ep, r in lab.iterrows():
        ep = int(ep)
        f = Path(baselines) / f"ep{ep:03d}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        y = int(r.outcome == "failure"); onset = int(r.onset_frame)
        e = {"y": y, "onset_s": onset / FPS if y and onset >= 0 else None,
             "s": {"w1_surprise": d["w1_surprise"], "elapsed": d["anchor_t"], "stall": d["stall"]},
             "t": {"w1_surprise": d["w1_surprise_t"], "elapsed": d["anchor_t"], "risk_logistic": d["anchor_t"],
                   "stall": d["stall_t"]},
             "anchor_X": d["anchor_X"], "train_X": d["train_X"], "train_y": d["train_y"]}
        if vj is not None:
            g = vj[vj.episode == ep].sort_values("t")
            if len(g):
                e["s"]["vla_jepa"] = g.wm_l1.to_numpy(); e["t"]["vla_jepa"] = g.known_at.to_numpy() / FPS
        info[ep] = e
    return info


def first_cross(s, t, tau):
    hit = np.flatnonzero(np.maximum.accumulate(s) > tau)
    return float(t[hit[0]]) if len(hit) else None


def evaluate(eps, info, scorers):
    import run_reap as R
    import run_w1_offline_screening as A
    from sklearn.model_selection import StratifiedKFold

    y = np.array([info[e]["y"] for e in eps])
    k = int(min(5, y.sum(), (1 - y).sum()))
    if k < 2:
        return {"error": f"cannot cross-validate: {int(y.sum())} failures, {int((1 - y).sum())} successes", "folds": k}
    per_seed = {m: [] for m in scorers}
    held_max = {m: np.zeros((SEEDS, len(eps))) for m in scorers}
    for seed in range(SEEDS):
        res = {m: {} for m in scorers}
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(eps)), y):
            tr_e, te_e = [eps[i] for i in tr], [eps[i] for i in te]
            sc = {m: {e: info[e]["s"][m] for e in tr_e + te_e} for m in scorers if m != "risk_logistic"}
            if "risk_logistic" in scorers:
                Xtr = np.concatenate([info[e]["train_X"] for e in tr_e]); ytr = np.concatenate([info[e]["train_y"] for e in tr_e])
                p = R.fit_predict("logistic", Xtr, ytr, np.concatenate([info[e]["anchor_X"] for e in tr_e + te_e]), seed)
                sc["risk_logistic"], j = {}, 0
                for e in tr_e + te_e:
                    n = len(info[e]["anchor_X"]); sc["risk_logistic"][e] = p[j:j + n]; j += n
            for m in scorers:
                tau = R.tau_from([sc[m][e].max() for e in tr_e if info[e]["y"] == 0])
                for e in te_e:
                    res[m][e] = (float(sc[m][e].max()), first_cross(sc[m][e], info[e]["t"][m], tau))
        for m in scorers:
            fails = [e for e in eps if info[e]["y"]]; succ = [e for e in eps if not info[e]["y"]]
            caught = [e for e in fails if res[m][e][1] is not None and info[e]["onset_s"] is not None
                      and res[m][e][1] < info[e]["onset_s"]]
            held_max[m][seed] = [res[m][e][0] for e in eps]
            per_seed[m].append({"caught": len(caught), "caught_episodes": caught,
                                "false_interventions": sum(res[m][e][1] is not None for e in succ),
                                "auroc": A.auroc(list(y), list(held_max[m][seed]))})
    rng = np.random.default_rng(BOOT_SEED)
    boots = [rng.integers(0, len(eps), len(eps)) for _ in range(BOOT)]
    out = {"folds": k, "failures": int(y.sum()), "successes": int((1 - y).sum()),
           "failures_without_onset": [e for e in eps if info[e]["y"] and info[e]["onset_s"] is None]}
    for m in scorers:
        r = per_seed[m]
        ci = []
        for b in boots:
            if 0 < y[b].sum() < len(b):
                ci.append(np.median([A.auroc(list(y[b]), list(held_max[m][s][b])) for s in range(SEEDS)]))
        out[m] = {"caught": float(np.median([x["caught"] for x in r])),
                  "false_interventions": float(np.median([x["false_interventions"] for x in r])),
                  "auroc": float(np.median([x["auroc"] for x in r])),
                  "auroc_ci95": [float(np.percentile(ci, 2.5)), float(np.percentile(ci, 97.5))] if ci else None,
                  "per_seed": r}
    return out


def verdict(res):
    """§29 bar, verbatim branches. Needs the vla_jepa scorer and all four baselines."""
    if "error" in res:
        return {"verdict": "NOT PROVEN", "reason": res["error"]}
    nf, ns = res["failures"], res["successes"]
    j = res["vla_jepa"]; best = max(res[m]["caught"] for m in BASELINES)
    lo, hi = j["auroc_ci95"] or (None, None)
    if nf < 6:
        return {"verdict": "NOT PROVEN", "reason": f"too few failures ({nf} < 6)"}
    if j["caught"] >= 6 and j["false_interventions"] <= ns / 20 and lo is not None and lo > 0.60 and j["caught"] >= best + 2:
        return {"verdict": "PASS", "reason": "caught >= 6 at <= 1/20 FI, CI low > 0.60, >= 2 more than the best baseline"}
    fail = []
    if j["caught"] <= 3:
        fail.append("caught <= 3")
    if lo is not None and lo <= 0.5 <= hi:
        fail.append("AUROC CI includes 0.5")
    if any(res[m]["caught"] >= j["caught"] for m in BASELINES):
        fail.append("a baseline catches as many or more")
    if fail:
        return {"verdict": "FAIL", "reason": "; ".join(fail)}
    return {"verdict": "NOT PROVEN", "reason": "neither PASS nor FAIL branch"}


def cmd_score(a):
    lab = load_labels(a.labels, a.onsets)
    info = load_traces(a.baselines, a.vla_jepa, lab)
    have_vj = [e for e in info if "vla_jepa" in info[e]["s"]]
    out = {"bar": "NEXT_ANGLES §29 (frozen 2026-10-25)", "labels": a.labels, "onsets": a.onsets,
           "stall_rule": json.loads(STALL_CAL.read_text())}
    eps = sorted(have_vj) if have_vj else sorted(info)
    scorers = SCORERS if have_vj else BASELINES
    res = evaluate(eps, info, scorers)
    out["episodes"] = eps; out["result"] = res
    if have_vj:
        out["verdict"] = verdict(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    if "error" in res:
        print(res["error"])
    else:
        print(f"{len(eps)} episodes, {res['failures']} failures, {res['successes']} successes, {res['folds']} folds")
        for m in scorers:
            v = res[m]
            print(f"  {m:14s} caught {v['caught']:.0f}/{res['failures']}  FI {v['false_interventions']:.0f}/{res['successes']}"
                  f"  AUROC {v['auroc']:.3f} CI {v['auroc_ci95']}")
    if "verdict" in out:
        print("verdict:", out["verdict"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("baselines")
    p.add_argument("--w1", required=True); p.add_argument("--root", required=True); p.add_argument("--labels", required=True)
    p.add_argument("--onsets"); p.add_argument("--out", required=True)
    p.add_argument("--latents", help="W1 latent cache for THIS dataset (default: <out>/../latents)")
    p = sub.add_parser("stall-calibrate")
    p.add_argument("--repo-id", default="ravediamond/so101-bowls-policy-rollouts")
    p.add_argument("--labels", default=str(ROOT / "data/rollouts/so101-bowls-policy-rollouts_labels.csv"))
    p = sub.add_parser("score")
    p.add_argument("--baselines", required=True); p.add_argument("--vla-jepa", dest="vla_jepa")
    p.add_argument("--labels", required=True); p.add_argument("--onsets"); p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"baselines": cmd_baselines, "stall-calibrate": cmd_stall_calibrate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
