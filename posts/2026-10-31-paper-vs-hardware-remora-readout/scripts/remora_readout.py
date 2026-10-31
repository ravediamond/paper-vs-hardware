#!/usr/bin/env python3
"""WS-065 (projects/remora-readout/PROTOCOL.md, frozen 2026-10-31): REMORA's model-free parts on the 69 SO-101 rollouts —
the error-aware progress target (paper §3.2, Eq. 2) and the causal nearest-peak readout (Eq. 3) — on G15's evidence.

Evidence: G15's `extract_features` (proprio + top-camera blobs, a query every 3 frames), cached per episode.
p_t: two logistic regressions fitted on G15's dev half only, P(grasped | x_t) and P(placed | x_t), with
x_t = [gripper, shoulder lift, blue-blob shift from its first position, blue–pink distance] (standardised on the dev
half; a missing blob → shift 0, distance = the dev maximum); labels at each query: grasped = frame ≥ the grasp label,
placed = frame ≥ the place label (−1 = never). p_t over the atoms {0, 0.5, 1}: reach = (1−g)(1−p), hold = g(1−p),
placed = p, renormalised.

Readouts (per query):
  R0  G15's monotone stage machine as is (mode `both`, thresholds from g15_summary.json): 0 → 0.5 at its grasp fire,
      → 1 at its place fire.
  R1  per-frame argmax of p_t.          R2  per-frame median of p_t over the atoms.
  R3  causal nearest peak: peaks = atoms that are local maxima of p_t with p ≥ 0.2; ŷ_t = the peak nearest ŷ_{t−1},
      ties → the lower atom; ŷ_0 = 0 (before the first query). No peak → keep ŷ_{t−1}.
Event fires for R1–R3: grasp = the first query with ŷ ≥ 0.5, place = the first query with ŷ = 1; scored with G15's
score_event (premature = more than 5 frames early or fires when absent; missed = never or more than 30 frames late).

Target ỹ (per frame): stages reach [0, grasp) 0 → 0.5, hold [grasp, place) 0.5 → 1, then flat at 1; with no place
label flat at 0.5 after the grasp, with no grasp label flat at 0. Inside each G14 window [a, c) (the episode's longest
reversal run from g14_episodes.csv), Eq. 2: β = min(α, v/L) per frame down from v = ỹ_a, α = the slope of the stage
before a, then the slope α⁺ = (B_{j+1} − (v − βL)) / (d_j − c) to the stage end d_j. A window in a flat stretch
(α = 0) gets β = 0 by the paper's own rule; check 2 therefore asks for a decrease inside the windows where α > 0 and
reports the flat ones.

Metrics (test half, G15's split): premature / missed per readout; flips per episode (changes of ŷ); regression recall
(G14 windows of test episodes inside which ŷ decreases at least once) and false regressions (test episodes without a
window in which ŷ decreases at least once); MAE and curvature S = mean |Δ²ŷ|, J = mean |Δ²ŷ − Δ²ỹ| (paper Eq. 5, 7) on
the queries inside the labelled span (first frame to the last label, or the episode end).
Checks: (1) R0 on the recomputed evidence reproduces G15's dev-half premature and missed rates (mode `both`) within
2 points; (2) the target is in [0, 1], non-decreasing outside windows, decreasing inside windows with α > 0;
(3) R3 on a synthetic p_t with two peaks (atoms 0 and 1) whose larger one alternates does not flip.

    IMAGE=lerobot-jetson:rc LEROBOT_SRC=image shared/gate/jetson_run.sh --python scripts/remora_readout.py features \\
        --cache outputs/remora/features
    … remora_readout.py fit --cache outputs/remora/features --out projects/remora-readout/results
    … remora_readout.py target --out projects/remora-readout/results
    … remora_readout.py checks --cache outputs/remora/features --out projects/remora-readout/results
    … remora_readout.py test --cache outputs/remora/features --out projects/remora-readout/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
G15 = ROOT / "shared/gate/data/2026-10-19_g15"
G14 = ROOT / "shared/gate/data/2026-10-18_g14/g14_episodes.csv"
OUTCOMES = ROOT / "data/rollouts/so101-bowls-policy-rollouts_labels.csv"
REPO = "ravediamond/so101-bowls-policy-rollouts"
ATOMS = np.array([0.0, 0.5, 1.0])
PEAK_MIN, N_BOOT = 0.2, 2000


def summary():
    return json.loads((G15 / "g15_summary.json").read_text())


def labels():
    return pd.read_csv(G15 / "transition_labels.csv").set_index("episode_index")


def load_feats(cache):
    import run_g15_pace_offline as G
    out = {}
    for f in sorted(Path(cache).glob("ep*_q3.npz")):
        z = np.load(f); e = int(f.stem[2:5])
        out[e] = G.Features(e, z["frames"], z["gripper"], z["shoulder"], z["blue"], z["pink"])
    return out


def cmd_features(a):
    from dataclasses import asdict

    import run_g15_pace_offline as G
    from so101_harness.episodes import EpisodeDataset, download
    ds = EpisodeDataset(download(REPO, with_videos=True))
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    for ep in labels().index:
        p = Path(a.cache) / f"ep{int(ep):03d}_q3.npz"
        if not p.exists():
            f = G.extract_features(ds, int(ep), 3)
            np.savez(p, **{k: v for k, v in asdict(f).items() if k != "episode"})
            print("features", ep, flush=True)


def xmat(f, stats=None):
    first = f.blue[~np.isnan(f.blue[:, 0])]
    shift = np.nan_to_num(np.hypot(*(f.blue - first[0]).T), nan=0.0) if len(first) else np.zeros(len(f.frames))
    dist = np.hypot(*(f.blue - f.pink).T)
    x = np.stack([f.gripper, f.shoulder, shift, dist], 1)
    if stats is None:
        return x
    x[:, 3] = np.nan_to_num(x[:, 3], nan=stats["dist_fill"])
    return (x - np.array(stats["mean"])) / np.array(stats["std"])


def logreg(x, y, l2=1e-3, iters=50):
    x1 = np.c_[x, np.ones(len(x))]; w = np.zeros(x1.shape[1])
    for _ in range(iters):                                                  # IRLS / Newton
        p = 1 / (1 + np.exp(-x1 @ w)); g = x1.T @ (p - y) + l2 * w
        h = (x1 * (p * (1 - p))[:, None]).T @ x1 + l2 * np.eye(len(w))
        w -= np.linalg.solve(h, g)
    return w


def cmd_fit(a):
    feats, lab, s = load_feats(a.cache), labels(), summary()
    raw = np.concatenate([xmat(feats[e]) for e in s["dev_episodes"]])
    fill = float(np.nanmax(raw[:, 3]))
    raw[:, 3] = np.nan_to_num(raw[:, 3], nan=fill)
    stats = {"mean": raw.mean(0).tolist(), "std": (raw.std(0) + 1e-9).tolist(), "dist_fill": fill,
             "features": ["gripper", "shoulder_lift", "blue_shift_px", "blue_pink_dist_px"]}
    X, yg, yp = [], [], []
    for e in s["dev_episodes"]:
        f = feats[e]; r = lab.loc[e]
        X.append(xmat(f, stats))
        yg.append((r.grasp_frame >= 0) & (f.frames >= r.grasp_frame)); yp.append((r.place_frame >= 0) & (f.frames >= r.place_frame))
    X = np.concatenate(X)
    res = {"stats": stats, "w_grasped": logreg(X, np.concatenate(yg).astype(float)).tolist(),
           "w_placed": logreg(X, np.concatenate(yp).astype(float)).tolist(), "dev_episodes": s["dev_episodes"],
           "queries": int(len(X))}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "p_t_fit.json").write_text(json.dumps(res, indent=1)); print("fit on", len(X), "dev queries")


def p_t(f, fit):
    x1 = np.c_[xmat(f, fit["stats"]), np.ones(len(f.frames))]
    g = 1 / (1 + np.exp(-x1 @ np.array(fit["w_grasped"]))); p = 1 / (1 + np.exp(-x1 @ np.array(fit["w_placed"])))
    pt = np.stack([(1 - g) * (1 - p), g * (1 - p), p], 1)
    return pt / pt.sum(1, keepdims=True)


def r3(pt):
    out, prev = [], 0.0
    for row in pt:
        pk = [k for k in range(3) if row[k] >= PEAK_MIN and all(row[k] >= row[j] for j in (k - 1, k + 1) if 0 <= j < 3)]
        if pk:
            d = [abs(ATOMS[k] - prev) for k in pk]
            prev = float(ATOMS[pk[int(np.argmin(d))]])                       # argmin keeps the first (lower) on ties
        out.append(prev)
    return np.array(out)


def readouts(f, fit, params):
    import run_g15_pace_offline as G
    pt = p_t(f, fit)
    g, p = G.evidence(f, params, "both")
    fg, fp = G.run_stage_machine(g, p, f.frames)
    r0 = np.zeros(len(f.frames))
    if fg is not None:
        r0[f.frames >= fg] = 0.5
    if fp is not None:
        r0[f.frames >= fp] = 1.0
    r1 = ATOMS[pt.argmax(1)]
    cdf = np.cumsum(pt, 1); r2 = ATOMS[(cdf >= 0.5).argmax(1)]
    return {"R0": r0, "R1": r1, "R2": r2, "R3": r3(pt)}, pt


def fires(y, frames):
    gi = np.flatnonzero(y >= 0.5); pi = np.flatnonzero(y >= 1.0)
    return (int(frames[gi[0]]) if len(gi) else None), (int(frames[pi[0]]) if len(pi) else None)


def windows():
    w = pd.read_csv(G14)
    return {int(r.episode_index): (int(r.run_start_frame), int(r.run_end_frame)) for _, r in w.iterrows() if r.oscillating}


def target(T, grasp, place, win):
    """Per-frame ỹ for one episode (Eq. 2 inside the G14 window)."""
    y = np.zeros(T); bounds = [(0, 0.0)]
    if grasp >= 0:
        bounds.append((grasp, 0.5))
        if place >= 0:
            bounds.append((place, 1.0))
    segs = []
    for k, (b, v) in enumerate(bounds):
        d = bounds[k + 1][0] if k + 1 < len(bounds) else T
        vend = bounds[k + 1][1] if k + 1 < len(bounds) else v
        segs.append((b, d, v, vend))
        y[b:d] = v + (vend - v) * (np.arange(d - b) / max(d - b, 1))
    info = {"window": None, "alpha": None}
    if win:
        a, c = win
        for b, d, v0, v1 in segs:
            if b <= a < d:
                c = min(c, d - 1); L = max(c - a, 1)
                alpha = (v1 - v0) / max(d - b, 1)
                v = y[a]; beta = min(alpha, v / L)
                y[a:c] = v - beta * np.arange(c - a)
                vc = v - beta * L
                if c < d:
                    aplus = (v1 - vc) / max(d - c, 1) if d < T or v1 != v0 else 0.0
                    y[c:d] = vc + aplus * np.arange(d - c)
                info = {"window": [a, c], "alpha": alpha, "beta": beta}
                break
    return np.clip(y, 0, 1), info


def cmd_target(a):
    lab, wins, oc = labels(), windows(), pd.read_csv(OUTCOMES).set_index("episode_index")
    with open(Path(a.out) / "target.jsonl", "w") as fo:
        for e in lab.index:
            T = int(oc.loc[e, "length"]); r = lab.loc[e]
            y, info = target(T, int(r.grasp_frame), int(r.place_frame), wins.get(int(e)))
            fo.write(json.dumps({"episode": int(e), "T": T, "target": np.round(y, 6).tolist(), **info}) + "\n")
    print("target for", len(lab), "episodes")


def load_targets(out):
    return {d["episode"]: d for d in map(json.loads, open(Path(out) / "target.jsonl"))}


def cmd_checks(a):
    import run_g15_pace_offline as G
    feats, s, lab = load_feats(a.cache), summary(), labels()
    params = s["modes"]["both"]["params"]
    r = G.evaluate(feats, lab, params, "both", s["dev_episodes"])
    ref = s["modes"]["both"]["dev"]
    c1 = abs(r["premature_rate"] - ref["premature_rate"]) <= 0.02 and abs(r["missed_rate"] - ref["missed_rate"]) <= 0.02
    tg, wins = load_targets(a.out), windows()
    c2, flat = True, []
    for e, d in tg.items():
        y = np.array(d["target"]); dy = np.diff(y)
        if not (0 <= y.min() and y.max() <= 1):
            c2 = False
        mask = np.ones(len(dy), bool)
        if d["window"]:
            a_, c_ = d["window"]; mask[max(a_ - 1, 0): c_] = False
            if d["alpha"] and d["alpha"] > 0:
                c2 &= bool((dy[a_: c_ - 1] < 0).all()) if c_ - 1 > a_ else True
            else:
                flat.append(e)
        c2 &= bool((dy[mask] >= -1e-9).all())
    syn = np.array([[0.46, 0.08, 0.46], [0.44, 0.08, 0.48]] * 10)            # two peaks (0 and 1), the larger alternating
    y3 = r3(syn); c3 = int((np.diff(y3) != 0).sum()) <= 1
    res = {"check1": {"R0_dev_premature": r["premature_rate"], "R0_dev_missed": r["missed_rate"],
                      "g15_dev_premature": ref["premature_rate"], "g15_dev_missed": ref["missed_rate"], "ok": bool(c1)},
           "check2": {"ok": bool(c2), "windows_in_flat_stretch": flat},
           "check3": {"flips_on_alternating_equal_peaks": int((np.diff(y3) != 0).sum()), "ok": bool(c3)}}
    (Path(a.out) / "checks.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    if not (c1 and c2 and c3):
        raise SystemExit("a check failed: stop and report")


def boot(x, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    b = [rng.choice(x, len(x)).mean() for _ in range(N_BOOT)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def cmd_test(a):
    import run_g15_pace_offline as G
    feats, s, lab = load_feats(a.cache), summary(), labels()
    fit = json.loads((Path(a.out) / "p_t_fit.json").read_text()); params = s["modes"]["both"]["params"]
    tg, wins = load_targets(a.out), windows()
    rows, per = [], {k: [] for k in ("R0", "R1", "R2", "R3")}
    with open(Path(a.out) / "readouts.jsonl", "w") as fo:
        for e in s["test_episodes"]:
            f = feats[e]; r = lab.loc[e]; ro, pt = readouts(f, fit, params)
            y = np.array(tg[e]["target"]); fr = f.frames
            labs = [x for x in (int(r.grasp_frame), int(r.place_frame)) if x >= 0]
            span = fr <= (max(labs) if labs else len(y) - 1)
            fo.write(json.dumps({"episode": int(e), "frames": fr.tolist(), "p_t": np.round(pt, 4).tolist(),
                                 **{k: v.tolist() for k, v in ro.items()}}) + "\n")
            for k, yh in ro.items():
                fg, fp = fires(yh, fr)
                if k == "R0":
                    g_, p_ = G.evidence(f, params, "both"); fg, fp = G.run_stage_machine(g_, p_, fr)
                ev = {"grasp": G.score_event(fg, int(r.grasp_frame)), "place": G.score_event(fp, int(r.place_frame))}
                yt = y[np.minimum(fr, len(y) - 1)]
                dec = np.diff(yh) < 0
                w = wins.get(int(e))
                inside = (fr[1:] >= w[0]) & (fr[1:] < w[1]) if w else None
                d2h = np.diff(yh[span], 2); d2t = np.diff(yt[span], 2)
                per[k].append({"episode": int(e), "premature": sum(v == "premature" for v in ev.values()),
                               "missed": sum(v == "missed" for v in ev.values()),
                               "present": int(r.grasp_frame >= 0) + int(r.place_frame >= 0),
                               "flips": int((np.diff(yh) != 0).sum()),
                               "window": w is not None, "regressed_in_window": bool((dec & inside).any()) if w else None,
                               "decreases": bool(dec.any()),
                               "mae": float(np.abs(yh[span] - yt[span]).mean()) if span.any() else None,
                               "S": float(np.abs(d2h).mean()) if len(d2h) else None,
                               "J": float(np.abs(d2h - d2t).mean()) if len(d2h) else None})
    res = {"test_episodes": len(s["test_episodes"])}
    for k, rs in per.items():
        n_ev = 2 * len(rs); pres = sum(r["present"] for r in rs)
        wn = [r for r in rs if r["window"]]; nw = [r for r in rs if not r["window"]]
        res[k] = {"premature_rate": sum(r["premature"] for r in rs) / n_ev,
                  "missed_rate": sum(r["missed"] for r in rs) / max(pres, 1),
                  "flips_mean": float(np.mean([r["flips"] for r in rs])),
                  "regression_recall": (sum(r["regressed_in_window"] for r in wn) / len(wn)) if wn else None,
                  "windows": len(wn),
                  "false_regressions": (sum(r["decreases"] for r in nw) / len(nw)) if nw else None,
                  "MAE": float(np.mean([r["mae"] for r in rs if r["mae"] is not None])),
                  "S": float(np.mean([r["S"] for r in rs if r["S"] is not None])),
                  "J": float(np.mean([r["J"] for r in rs if r["J"] is not None]))}
    d_prem = [a3["premature"] - a1["premature"] for a3, a1 in zip(per["R3"], per["R1"])]
    d_miss = [a3["missed"] - a1["missed"] for a3, a1 in zip(per["R3"], per["R1"])]
    flips_ratio = [(a3["flips"], a1["flips"]) for a3, a1 in zip(per["R3"], per["R1"])]
    rng = np.random.default_rng(0)
    br = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(flips_ratio), len(flips_ratio))
        s1 = sum(flips_ratio[i][1] for i in idx)
        br.append(sum(flips_ratio[i][0] for i in idx) / s1 if s1 else np.nan)
    r31 = sum(x for x, _ in flips_ratio) / max(sum(y for _, y in flips_ratio), 1)
    p1_pass = res["R3"]["premature_rate"] <= res["R1"]["premature_rate"] and res["R3"]["missed_rate"] <= res["R1"]["missed_rate"] + 0.02
    p1_fail = res["R3"]["missed_rate"] > res["R1"]["missed_rate"] + 0.05
    p2_pass = r31 <= 0.5 and np.nanpercentile(br, 97.5) <= 0.7
    rr, fr_ = res["R3"]["regression_recall"], res["R3"]["false_regressions"]
    p3_pass = rr is not None and rr >= 0.5 and fr_ is not None and fr_ <= 0.2
    res["paired"] = {"premature_R3_minus_R1_mean": float(np.mean(d_prem)), "premature_ci95": boot(d_prem),
                     "missed_R3_minus_R1_mean": float(np.mean(d_miss)), "missed_ci95": boot(d_miss),
                     "flips_R3_over_R1": r31, "flips_ratio_ci95": [float(np.nanpercentile(br, 2.5)), float(np.nanpercentile(br, 97.5))]}
    res["verdicts"] = {"P1": "pass" if p1_pass else "fail" if p1_fail else "not proven",
                       "P2": "pass" if p2_pass else "fail", "P3": "pass" if p3_pass else "fail",
                       "R0_check_vs_g15_test": {"premature_rate": s["modes"]["both"]["test"]["premature_rate"],
                                                "missed_rate": s["modes"]["both"]["test"]["missed_rate"]}}
    (Path(a.out) / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("features", "fit", "checks", "test"):
        p = sub.add_parser(name); p.add_argument("--cache", required=True)
        if name != "features":
            p.add_argument("--out", required=True)
    t = sub.add_parser("target"); t.add_argument("--out", required=True)
    a = ap.parse_args()
    {"features": cmd_features, "fit": cmd_fit, "target": cmd_target, "checks": cmd_checks, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
