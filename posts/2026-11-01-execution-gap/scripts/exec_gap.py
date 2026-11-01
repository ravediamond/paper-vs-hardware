#!/usr/bin/env python3
"""WS-088, PROTOCOL.md (frozen 5ca5880): servo dead zone (C1) and command-vs-follower gap (C2) on the SO-101.

    python exec_gap.py --demos DEMOS_ROOT --rollouts ROLLOUTS_ROOT --chunks outputs/smooth/chunks \
        --onsets onsets.csv --g16 g16b_episodes.csv --out results/
Command side = `action` (leader in demos, policy in rollouts); follower side = `observation.state`.
"""
import argparse, csv, glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM, GRIP = list(range(5)), 5
HOLD_STEP, HOLD_LEN, MAXLAG, B, SEED = 0.05, 15, 10, 2000, 0
BINS = [(0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8), (0.8, 1e9)]


def load(root):
    d = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action", "observation.state"])
                   for f in sorted(glob.glob(f"{root}/data/*/*.parquet"))])
    out = {}
    for ep, g in d.groupby("episode_index"):
        g = g.sort_values("frame_index")
        out[int(ep)] = (np.stack(g.action.to_numpy()).astype(float), np.stack(g["observation.state"].to_numpy()).astype(float))
    return out


def lags(eps):
    L = []
    for j in range(6):
        best, bl = -2, 0
        for lag in range(MAXLAG + 1):
            x = np.concatenate([np.diff(a[:, j])[: len(a) - 1 - lag] for a, s in eps.values()])
            y = np.concatenate([np.diff(s[:, j])[lag:] for a, s in eps.values()])
            c = np.corrcoef(x, y)[0, 1]
            if c > best:
                best, bl = c, lag
        L.append(bl)
    return L


def holds(a, j):
    """End indices of hold windows: >= HOLD_LEN consecutive frames with |Δa_j| <= HOLD_STEP."""
    still = np.abs(np.diff(a[:, j])) <= HOLD_STEP
    ends, run = [], 0
    for t, s in enumerate(still):
        run = run + 1 if s else 0
        if run == HOLD_LEN:
            ends.append(t + 1)
    return ends


def dead_zone(eps, L):
    per = {j: {} for j in range(6)}
    for ep, (a, s) in eps.items():
        for j in range(6):
            v = [abs(s[e + L[j], j] - a[e, j]) for e in holds(a, j) if e + L[j] < len(s)]
            if v:
                per[j][ep] = v
    res = {}
    for j in range(6):
        keys = list(per[j]); allv = np.concatenate([per[j][k] for k in keys]) if keys else np.array([])
        rng = np.random.default_rng(SEED); bs = []
        for _ in range(B):
            pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            bs.append(np.median(np.concatenate([per[j][k] for k in pick])))
        res[JOINTS[j]] = {"median_deg": float(np.median(allv)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                          "n_holds": int(len(allv)), "episodes": len(keys)}
    return res


def substep(eps, L):
    out = {}
    for j in ARM:
        cnt = {b: [0, 0] for b in BINS}
        for a, s in eps.values():
            for e in holds(a, j):
                t = e  # first command change after the hold
                if t + 1 >= len(a) or t + 1 + L[j] + 5 >= len(s):
                    continue
                step = a[t + 1, j] - a[t, j]
                if abs(step) <= HOLD_STEP:
                    continue
                moved = abs(s[t + 1 + L[j] + 5, j] - s[t + L[j], j])
                for b in BINS:
                    if b[0] <= abs(step) < b[1]:
                        cnt[b][1] += 1; cnt[b][0] += int(moved < abs(step) / 2)
        out[JOINTS[j]] = {f"{lo}-{hi if hi < 1e8 else 'inf'}": {"n": c[1], "share_under_half": (c[0] / c[1]) if c[1] else None} for (lo, hi), c in cnt.items()}
    return out


def corrections(chunk_dir, dz):
    import smooth_chunking as S
    aff = json.load(open(f"{chunk_dir}/affine.json")); sc, off = np.array(aff["scale"]), np.array(aff["offset"])
    res = {}
    for kind in ("S2", "S3"):
        diffs = []
        for f in sorted(Path(chunk_dir).glob("ep*.npz")):
            z = np.load(f); d = {k: z[k] for k in z.files}
            a0, _, _ = S.stream("S0", 50, d); ak, _, _ = S.stream(kind, 50, d)
            diffs.append(np.abs((ak - a0) * sc + off - off)[:, ARM])
        D = np.concatenate(diffs)
        res[kind] = {JOINTS[j]: {"median_change_deg": float(np.median(D[:, j])),
                                 "share_below_dead_zone": float((D[:, j] < dz[JOINTS[j]]["median_deg"]).mean())} for j in ARM}
    return res


def grasp_window(a):
    g = a[:, GRIP]; th = (np.percentile(g, 5) + np.percentile(g, 95)) / 2
    above = g > th
    x = np.where(above[:-1] & ~above[1:])[0]  # command closing: crosses below θ
    if not len(x):
        return None
    t = int(x[0]) + 1
    return max(0, t - 15), min(len(a), t + 30)


def ratio_boot(f, s):
    rng = np.random.default_rng(SEED); v = []
    for _ in range(B):
        v.append(np.median(f[rng.integers(0, len(f), len(f))]) / np.median(s[rng.integers(0, len(s), len(s))]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def mannwhitney_p(x, y):
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)


def norm_residual(a, s, chunk=50):
    num = den = 0.0
    for t in range(len(a) - 1):
        num += np.linalg.norm(s[t + 1, ARM] - a[t, ARM])
        den += np.linalg.norm(a[t, ARM] - s[(t // chunk) * chunk, ARM])
    return num / den if den else None


def main():
    ap = argparse.ArgumentParser()
    for k in ("demos", "rollouts", "chunks", "onsets", "g16", "labels", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    demos, rolls = load(a.demos), load(a.rollouts)
    lab = {int(r["episode_index"]): r for r in csv.DictReader(open(a.labels))}
    rolls = {e: v for e, v in rolls.items() if lab.get(e, {}).get("outcome") in ("success", "failure")}
    L = lags(demos)
    out = {"lags_frames": dict(zip(JOINTS, L)), "n_demos": len(demos), "n_rollouts": len(rolls)}
    # C1
    out["c1_dead_zone_rollouts"] = dead_zone(rolls, L); out["c1_dead_zone_demos"] = dead_zone(demos, L)
    out["c1_substep_rollouts"] = substep(rolls, L)
    out["c1_corrections"] = corrections(a.chunks, out["c1_dead_zone_rollouts"])
    big = [j for j in ARM if out["c1_dead_zone_rollouts"][JOINTS[j]]["ci95"][0] >= 0.3]
    out["c1_joints_dead_zone_ge_0.3"] = [JOINTS[j] for j in big]
    out["c1_reading"] = "sub-servo" if len(big) >= 3 else "not sub-servo"
    # C2
    per = {"failure": {"grip": [], "wrist": []}, "success": {"grip": [], "wrist": []}}; excl = {"failure": 0, "success": 0}
    pre = []
    on = {int(r["episode_index"]): int(r["onset_frame"]) for r in csv.DictReader(open(a.onsets))
          if r.get("onset_frame", "").strip().isdigit()}
    for e, (act, st) in rolls.items():
        o = lab[e]["outcome"]; w = grasp_window(act)
        gap = np.abs(act[: len(act) - max(L)] - np.stack([st[L[j]: L[j] + len(act) - max(L), j] for j in range(6)], 1))
        if w is None:
            excl[o] += 1
        else:
            per[o]["grip"].append(np.median(gap[w[0]:w[1], GRIP])); per[o]["wrist"].append(np.median(gap[w[0]:w[1], 3:5].mean(1)))
        if o == "failure" and e in on and on[e] >= 30:
            pre.append(float(np.median(gap[on[e] - 30:on[e], GRIP])))
    f, s = np.array(per["failure"]["grip"]), np.array(per["success"]["grip"])
    r = float(np.median(f) / np.median(s)); ci = ratio_boot(f, s)
    fw, sw = np.array(per["failure"]["wrist"]), np.array(per["success"]["wrist"])
    out["c2"] = {"excluded_no_grasp": excl, "n_failure": len(f), "n_success": len(s),
                 "gripper_median_failure": float(np.median(f)), "gripper_median_success": float(np.median(s)),
                 "gripper_ratio": r, "gripper_ratio_ci95": ci, "gripper_mannwhitney_p": mannwhitney_p(f, s),
                 "wrist_ratio": float(np.median(fw) / np.median(sw)), "wrist_ratio_ci95": ratio_boot(fw, sw),
                 "wrist_mannwhitney_p": mannwhitney_p(fw, sw),
                 "pre_onset_gripper_median": float(np.median(pre)) if pre else None, "pre_onset_n": len(pre)}
    g16 = {int(r["episode_index"]) for r in csv.DictReader(open(a.g16)) if r["in_g16_set"] == "True"}
    sub = {"failure": [], "success": []}
    for e, (act, st) in rolls.items():
        w = grasp_window(act)
        if e in g16 and w is not None:
            gap = np.abs(act[: len(act) - max(L), GRIP] - st[L[GRIP]: L[GRIP] + len(act) - max(L), GRIP])
            sub[lab[e]["outcome"]].append(np.median(gap[w[0]:w[1]]))
    out["c2_g16_subset"] = {"n_failure": len(sub["failure"]), "n_success": len(sub["success"]),
                            "gripper_ratio": float(np.median(sub["failure"]) / np.median(sub["success"])) if sub["failure"] and sub["success"] else None}
    out["c2_reading"] = "execution matters" if r >= 2 and ci[0] > 1 else ("not seen" if ci[0] <= 1 <= ci[1] else "partial")
    # 2609.37334's normalised residual, and the demos' leader-follower gap (CSDW side)
    nr = {e: norm_residual(*v) for e, v in rolls.items()}
    pol = {}
    for e, v in nr.items():
        pol.setdefault(lab[e]["policy"], []).append(v)
    out["normalised_residual_rollouts"] = {"mean": float(np.mean([v for v in nr.values() if v is not None])),
                                           "per_policy": {p: float(np.mean(v)) for p, v in pol.items()},
                                           "by_outcome": {o: float(np.mean([nr[e] for e in rolls if lab[e]["outcome"] == o])) for o in ("success", "failure")}}
    out["normalised_residual_demos_leader_follower"] = float(np.mean([norm_residual(*v) for v in demos.values()]))
    Path(a.out).mkdir(parents=True, exist_ok=True); (Path(a.out) / "exec_gap.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
