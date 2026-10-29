"""WS-047 gate 0, step 2: R²(h→z), anisotropy, ε, δ, and bars B1–B3 (PROTOCOL.md + amendment 1). numpy only.

    python3 gate0_analyze.py <out dir> <results dir>   # reads <out>/latents.npz, writes <results>/gate0.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ALPHAS = [10.0 ** k for k in range(-3, 4)]
G1B = {"W1": 0.6046, "W2p": 0.6458}  # amendment 1: best G1b cell, same G1 protocol
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def ridge_fit(x, y, a):
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    w = np.linalg.solve(xc.T @ xc + a * np.eye(x.shape[1]), xc.T @ yc)
    return w, mx, my


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def choose_alpha(x, y, ep, k=5, seed=0):
    eps = np.unique(ep)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(eps), k)
    score = {}
    for a in ALPHAS:
        s = []
        for f in folds:
            te = np.isin(ep, f)
            w, mx, my = ridge_fit(x[~te], y[~te], a)
            s.append(r2(y[te], (x[te] - mx) @ w + my).mean())
        score[a] = float(np.mean(s))
    return max(score, key=score.get), score


def diagnostics(h, ep):
    c = np.cov(h, rowvar=False)
    n = h.shape[1]
    eps_ = float(np.linalg.norm(c - np.eye(n)) / np.sqrt(n))
    same = ep[1:] == ep[:-1]  # consecutive frames of one episode only
    d = h[1:][same] - h[:-1][same]
    hc = h - h.mean(0)
    lag = (hc[1:][same] * hc[:-1][same]).mean(0) / hc.var(0)
    rho = float(np.mean(lag))
    delta = float((d ** 2).sum(1).mean() - 2 * (1 - rho) * np.trace(c))
    return eps_, rho, delta


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else (1.0 if (a[0] - a[1]) * (b[0] - b[1]) > 0 else -1.0)


def main():
    out = Path(sys.argv[1])
    d = np.load(out / "latents.npz")
    z, ep, val = d["z"].astype(np.float64), d["episode"], d["is_val"].astype(bool)
    res = {"frames": {"train": int((~val).sum()), "val": int(val.sum())}, "models": {}}
    for name, key in (("W1", "h_w1"), ("W2p", "h_w2p")):
        h = d[key].astype(np.float64)
        a, cv = choose_alpha(h[~val], z[~val], ep[~val])
        w, mx, my = ridge_fit(h[~val], z[~val], a)
        per = r2(z[val], (h[val] - mx) @ w + my)
        e_, rho, delta = diagnostics(h[val], ep[val])
        res["models"][name] = {"alpha": a, "cv_mean_r2": {str(k): round(v, 4) for k, v in cv.items()},
                               "r2_per_joint": dict(zip(JOINTS, [round(float(x), 4) for x in per])),
                               "mean_r2": round(float(per.mean()), 4),
                               "anisotropy": round(float(per.max() - per.min()), 4),
                               "epsilon": round(e_, 4), "rho_hat": round(rho, 4), "delta": round(delta, 4),
                               "g1b_top1": G1B[name], "dim": int(h.shape[1])}
    m = res["models"]
    names = list(m)
    sp = spearman([m[n]["mean_r2"] for n in names], [m[n]["g1b_top1"] for n in names])
    top_r2 = max(names, key=lambda n: m[n]["mean_r2"])
    top_g = max(names, key=lambda n: m[n]["g1b_top1"])
    res["bars"] = {
        "B1": {"value": m["W1"]["mean_r2"], "bar": "< 0.80", "met": m["W1"]["mean_r2"] < 0.80},
        "B2": {"value": m["W1"]["anisotropy"], "bar": ">= 0.20", "met": m["W1"]["anisotropy"] >= 0.20},
        "B3": {"spearman": sp, "top_r2": top_r2, "top_g1b": top_g, "n": len(names),
               "met": sp > 0 and top_r2 == top_g, "note": "n = 2: a sighting, not a test"}}
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else out
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "gate0.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
