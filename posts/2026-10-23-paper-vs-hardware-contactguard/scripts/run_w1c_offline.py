"""W1-C1 and W1-C2, offline on the 69 rollouts (protocol: V1.0_REVIEW.md section 6, commit 2bcd2ff).

c1 (event-anchored grasp check, ContactGuard-style): anchor = 15 frames before the gripper command first falls below
   10 having been above 20 (episodes with no such close are counted by failure mode, not scored); context = the anchor
   and the 2 frames before it with their recorded actions (W1-A); imagine K = 30 steps over the recorded actions;
   feature = z_hat at K (192-d). Label = episode failure. Logistic probe (StandardScaler fit inside each training
   fold, L2, C = 1.0, class_weight balanced), outcome-stratified 5-fold CV x 5 seeds; baseline = the same probe on the
   current latent z at the anchor. Threshold fit inside each training fold at <= 1/20 successes flagged; held-out
   false-abort rate and recall at it. Bar: held-out AUROC >= 0.85 and >= 0.05 above the current-latent probe.
c2 (time-varying band on W1 surprise, FAIL-Detect / SAFE style): per anchor (every 25 frames, as W1-A) the long score
   (max surprise over k in {15, 20, 25}); time index = anchor number; mu_t, sigma_t over calibration successes;
   R = max_t (s_t - mu_t) / sigma_t; q = the ceil((n + 1)(1 - alpha))-th smallest R, alpha = 0.05; band = mu + q sigma,
   last value carried forward; flagged when any s_t > band_t. 22 successes calibrate, the other 22 and the 25 failures
   test, 20 seeded splits. Bar (median over splits): successes flagged <= 1/20, failures flagged >= 50 %.

Reading of the protocol where it is silent (reported in the results): c1's "held-out AUROC" = per seed, the AUROC of
the pooled out-of-fold predictions, averaged over the 5 seeds (fold AUROCs give the spread); the fold threshold is set
on the training fold's own (in-sample) probe scores. c2's mu_t / sigma_t exist where >= 2 calibration successes have an
anchor t; beyond the last such index the band is carried forward (for calibration R as for test episodes).

  W1=1 shared/gate/jetson_run.sh --python scripts/run_w1c_offline.py --w1 <W1 weights_final.pt> \
      --latents outputs/w1_screening/latents --out shared/gate/data/<date>_w1c
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ANCHOR_BEFORE_CLOSE, K_IMAGINE = 15, 30
CMD_OPEN, CMD_CLOSED = 20.0, 10.0
FOLDS, SEEDS = 5, (0, 1, 2, 3, 4)
FAR_MAX = 1 / 20
BAR_C1 = {"auroc_min": 0.85, "margin_over_current": 0.05}
LONG_K = (15, 20, 25)
ALPHA, N_SPLITS, N_CAL = 0.05, 20, 22
BAR_C2 = {"success_flagged_max": 1 / 20, "failure_flagged_min": 0.5}


# ---------------------------------------------------------------------------------------------------------------
# c1
# ---------------------------------------------------------------------------------------------------------------
def first_close(cmd: np.ndarray) -> int | None:
    """First frame the gripper command falls below 10 having been above 20 (None: never)."""
    armed = False
    for i, v in enumerate(np.asarray(cmd, float)):
        if armed and v < CMD_CLOSED:
            return i
        if v > CMD_OPEN:
            armed = True
    return None


def c1_anchor(cmd: np.ndarray) -> int | None:
    c = first_close(cmd)
    return None if c is None else c - ANCHOR_BEFORE_CLOSE


def imagine_at(mon, z: np.ndarray, actions: np.ndarray, a: int, k: int = K_IMAGINE) -> np.ndarray:
    """W1LiveMonitor's imagination from anchor a (context a-2..a, recorded actions) -> z_hat at step k."""
    import torch

    mon._ctx = torch.from_numpy(z[a - 2: a + 1]).to(mon.device, mon.dtype).unsqueeze(0)
    mon._ctx_actions = actions[a - 2: a].astype(np.float32)
    mon.imagine_chunk(None, actions[a: a + k], check_steps=(k,))
    return mon._expected_latents[k].float().cpu().numpy()


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def fold_threshold(train_scores_success: np.ndarray) -> float:
    """Smallest threshold with <= 1/20 of the training successes strictly above it."""
    s = np.sort(train_scores_success)[::-1]
    allowed = int(math.floor(FAR_MAX * len(s)))
    return float(s[allowed]) if allowed < len(s) else float("-inf")


def probe_cv(X: np.ndarray, y: np.ndarray, seeds=SEEDS, folds: int = FOLDS) -> dict[str, Any]:
    """Outcome-stratified k-fold x seeds; the scaler and the probe are fit on the training fold only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    per_seed, fold_aucs, far, rec = [], [], [], []
    oof_all, flags_all = {}, {}
    for seed in seeds:
        oof = np.full(len(y), np.nan)
        flags = np.zeros(len(y), bool)
        fa = fs = ra = rs = 0
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=10000))
            m.fit(X[tr], y[tr])
            s_te = m.decision_function(X[te])
            oof[te] = s_te
            if len(set(y[te])) == 2:
                fold_aucs.append(auroc(y[te], s_te))
            tau = fold_threshold(m.decision_function(X[tr])[y[tr] == 0])
            flags[te] = s_te > tau
            fa += int((s_te[y[te] == 0] > tau).sum()); fs += int((y[te] == 0).sum())
            ra += int((s_te[y[te] == 1] > tau).sum()); rs += int((y[te] == 1).sum())
        per_seed.append(auroc(y, oof))
        far.append(fa / fs if fs else None)
        rec.append(ra / rs if rs else None)
        oof_all[seed], flags_all[seed] = oof, flags
    return {"auroc_oof_per_seed": [round(a, 4) for a in per_seed], "auroc": round(float(np.mean(per_seed)), 4),
            "auroc_fold": {"mean": round(float(np.mean(fold_aucs)), 4), "std": round(float(np.std(fold_aucs)), 4),
                           "min": round(float(np.min(fold_aucs)), 4), "max": round(float(np.max(fold_aucs)), 4),
                           "n": len(fold_aucs)},
            "held_out_false_abort_rate_per_seed": far, "held_out_recall_per_seed": rec,
            "held_out_false_abort_rate": round(float(np.median(far)), 4),
            "held_out_recall": round(float(np.median(rec)), 4), "oof": oof_all, "flags": flags_all}


def recall_by_mode(cv: dict[str, Any], y: np.ndarray, modes: list[str]) -> dict[str, Any]:
    """Per failure mode: held-out recall at the fold threshold (mean over seeds) and the AUROC of its out-of-fold
    scores against all successes."""
    out = {}
    succ = y == 0
    for m in sorted(set(modes[i] for i in range(len(y)) if y[i] == 1)):
        idx = np.array([y[i] == 1 and modes[i] == m for i in range(len(y))])
        aucs = [auroc(np.r_[np.zeros(succ.sum()), np.ones(idx.sum())], np.r_[o[succ], o[idx]])
                for o in cv["oof"].values()]
        out[m] = {"n": int(idx.sum()), "recall_at_fold_threshold": round(float(np.mean(
                  [f[idx].mean() for f in cv["flags"].values()])), 4),
                  "auroc_vs_successes": round(float(np.mean(aucs)), 4)}
    return out


# ---------------------------------------------------------------------------------------------------------------
# c2
# ---------------------------------------------------------------------------------------------------------------
def fit_band(cal: list[np.ndarray], alpha: float = ALPHA) -> dict[str, Any]:
    """One-sided functional CP band from calibration score series (one per success, ragged lengths)."""
    L = max(len(s) for s in cal)
    mu, sd = [], []
    for t in range(L):
        v = np.array([s[t] for s in cal if len(s) > t])
        if len(v) < 2:
            break
        mu.append(float(v.mean()))
        sd.append(float(v.std(ddof=1)))
    mu, sd = np.array(mu), np.array(sd)
    R = np.array([band_norm_max(s, mu, sd) for s in cal])
    n = len(cal)
    rank = math.ceil((n + 1) * (1 - alpha))
    q = float(np.sort(R)[min(rank, n) - 1]) if rank <= n else float("inf")
    return {"mu": mu, "sd": sd, "q": q, "R": R, "rank": rank, "n": n}


def _carry(arr: np.ndarray, t: int) -> float:
    return float(arr[min(t, len(arr) - 1)])


def band_norm_max(s: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> float:
    return max((s[t] - _carry(mu, t)) / max(_carry(sd, t), 1e-12) for t in range(len(s)))


def flagged(s: np.ndarray, band: dict[str, Any]) -> bool:
    return any(s[t] > _carry(band["mu"], t) + band["q"] * _carry(band["sd"], t) for t in range(len(s)))


def band_splits(success: list[np.ndarray], failure: list[np.ndarray], fail_modes: list[str],
                n_splits: int = N_SPLITS, n_cal: int = N_CAL, alpha: float = ALPHA) -> dict[str, Any]:
    fpr, rec, per_mode = [], [], {}
    for seed in range(n_splits):
        perm = np.random.default_rng(seed).permutation(len(success))
        cal, test = [success[i] for i in perm[:n_cal]], [success[i] for i in perm[n_cal:]]
        band = fit_band(cal, alpha)
        fpr.append(float(np.mean([flagged(s, band) for s in test])))
        fl = [flagged(s, band) for s in failure]
        rec.append(float(np.mean(fl)))
        for m, f in zip(fail_modes, fl):
            per_mode.setdefault(m, []).append(f)
    q = lambda v: {"median": round(float(np.median(v)), 4), "min": round(float(np.min(v)), 4),  # noqa: E731
                   "max": round(float(np.max(v)), 4), "iqr": [round(float(np.percentile(v, 25)), 4),
                                                              round(float(np.percentile(v, 75)), 4)]}
    return {"success_flagged": q(fpr), "failure_flagged": q(rec), "per_split": {"fpr": fpr, "recall": rec},
            "recall_by_mode": {m: {"n": fail_modes.count(m), "flag_rate": round(float(np.mean(v)), 4)}
                               for m, v in sorted(per_mode.items())}}


# ---------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w1", required=True)
    ap.add_argument("--repo-id", default="ravediamond/so101-bowls-policy-rollouts")
    ap.add_argument("--outcomes", type=Path, default=ROOT / "data/rollouts/so101-bowls-policy-rollouts_labels.csv")
    ap.add_argument("--latents", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from run_w1_offline_screening import anchor_curves, encode_episode
    from so101_harness.episodes import EpisodeDataset, download
    from so101_harness.w1_live import W1LiveMonitor

    ds = EpisodeDataset(download(args.repo_id, with_videos=True))
    o = pd.read_csv(args.outcomes)
    o = o[o.outcome.isin(["success", "failure"])]
    mon = W1LiveMonitor.from_checkpoint(args.w1)
    args.latents.mkdir(parents=True, exist_ok=True)

    rows, no_anchor, series = [], [], []
    for r in o.itertuples():
        ep = int(r.episode_index)
        f = args.latents / f"ep{ep:03d}.npy"
        if not f.exists():
            np.save(f, encode_episode(mon, ds, ep))
        z, acts = np.load(f), ds.actions(ep)
        T = len(acts)
        end = T if int(r.reset_start) < 0 else min(int(r.reset_start), T)
        mode = r.failure_mode if isinstance(r.failure_mode, str) and r.failure_mode else (
            "none" if r.outcome == "failure" else "")
        # c2 series (same anchors and imagination as W1-A / G11-B-cal)
        curves = anchor_curves(mon, z, acts, end)
        series.append({"ep": ep, "outcome": r.outcome, "mode": mode,
                       "long": np.array([max(c["l2"][k] for k in LONG_K) for c in curves])})
        # c1
        a = c1_anchor(acts[:end, 5])
        if a is None or a < 2 or a + K_IMAGINE > T or np.isnan(z[a - 2: a + 1]).any():
            no_anchor.append({"episode_index": ep, "outcome": r.outcome, "failure_mode": mode,
                              "reason": "no close" if a is None else "anchor out of range"})
            continue
        rows.append({"ep": ep, "outcome": r.outcome, "mode": mode, "anchor": a,
                     "z_hat": imagine_at(mon, z, acts, a), "z": z[a].astype(np.float64)})
        print(f"  ep {ep:3d} {r.outcome:7s} anchor {a}", flush=True)

    y = np.array([int(x["outcome"] == "failure") for x in rows])
    X_hat, X_cur = np.stack([x["z_hat"] for x in rows]), np.stack([x["z"] for x in rows])
    modes = [x["mode"] for x in rows]
    imag, cur = probe_cv(X_hat, y), probe_cv(X_cur, y)
    margin = round(imag["auroc"] - cur["auroc"], 4)
    c1_pass = imag["auroc"] >= BAR_C1["auroc_min"] and margin >= BAR_C1["margin_over_current"]
    c1 = {"bar": BAR_C1, "scored_episodes": len(rows), "successes": int((y == 0).sum()), "failures": int(y.sum()),
          "not_scored": no_anchor,
          "not_scored_by_mode": pd.Series([n["failure_mode"] or n["outcome"] for n in no_anchor]).value_counts().to_dict(),
          "imagined": {k: v for k, v in imag.items() if k not in ("oof", "flags")},
          "current": {k: v for k, v in cur.items() if k not in ("oof", "flags")}, "margin": margin,
          "imagined_by_mode": recall_by_mode(imag, y, modes),
          "current_by_mode": recall_by_mode(cur, y, modes),
          "verdict": "PASS: register a live shadow check at the grasp" if c1_pass else "FAIL: W1 stays a logged signal"}

    succ = [s["long"] for s in series if s["outcome"] == "success"]
    fail = [s["long"] for s in series if s["outcome"] == "failure"]
    fmodes = [s["mode"] for s in series if s["outcome"] == "failure"]
    c2r = band_splits(succ, fail, fmodes)
    c2_pass = (c2r["success_flagged"]["median"] <= BAR_C2["success_flagged_max"]
               and c2r["failure_flagged"]["median"] >= BAR_C2["failure_flagged_min"])
    full = fit_band(succ[:N_CAL])  # one band, for its shape in the report (split 0 uses a permutation)
    c2 = {"bar": BAR_C2, "alpha": ALPHA, "splits": N_SPLITS, "calibration_successes": N_CAL,
          "test_successes": len(succ) - N_CAL, "failures": len(fail),
          "q_rank": f"ceil((n+1)(1-alpha)) = {full['rank']} of n = {full['n']}",
          "band_length_anchors": int(len(full["mu"])), **c2r,
          "verdict": "PASS: register the band for G11-B" if c2_pass else "FAIL: shadow mode stays"}

    args.out.mkdir(parents=True, exist_ok=True)
    res = {"protocol": "V1.0_REVIEW.md section 6, W1-C1 / W1-C2 (commit 2bcd2ff)", "w1": args.w1, "c1": c1, "c2": c2}
    (args.out / "w1c_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({"c1": {k: c1[k] for k in ("scored_episodes", "not_scored_by_mode", "margin", "verdict")}
                      | {"imagined_auroc": imag["auroc"], "current_auroc": cur["auroc"]},
                      "c2": {k: c2[k] for k in ("success_flagged", "failure_flagged", "verdict")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
