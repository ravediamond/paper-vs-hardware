#!/usr/bin/env python3
"""WS-086, PROTOCOL.md (frozen 7e29306): is SmolVLA's chunk error concentrated near sub-skill transitions?

    python3 projects/race-diagnostic/race_diag.py   # reads outputs/smooth/chunks, writes results/race.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CH = ROOT / "outputs/smooth/chunks"
OUT = Path(__file__).resolve().parent / "results"
HS, NEAR, FAR, MERGE, B, SEED = (10, 25, 50), 5, 15, 15, 2000, 0


def transitions(g):
    th = (np.percentile(g, 5) + np.percentile(g, 95)) / 2
    s = g > th
    x = list(np.where(s[1:] != s[:-1])[0] + 1)
    out = []
    for t in x:
        if not out or t - out[-1] >= MERGE:
            out.append(t)
    return out


def cells(z, hold=False):
    """Rows (i, delta, err_arm, err_grip) for every chunk cell inside the episode."""
    rec, T = z["recorded"], int(z["T"]); tr = transitions(rec[:, 5])
    if not tr:
        return None
    rows = []
    for k, f in enumerate(z["frames"]):
        c = np.repeat(rec[f][None], 50, 0) if hold else z["chunks_deg"][k]
        for i in range(50):
            t = f + i
            if t >= T:
                break
            d = min(abs(t - x) for x in tr)
            rows.append((i, d, np.abs(c[i, :5] - rec[t, :5]).mean(), abs(c[i, 5] - rec[t, 5])))
    return np.array(rows)


def ratio(eps, H, col=2):
    num = [e[(e[:, 0] < H) & (e[:, 1] <= NEAR), col] for e in eps]
    den = [e[(e[:, 0] < H) & (e[:, 1] > FAR), col] for e in eps]
    n, d = np.concatenate(num), np.concatenate(den)
    return float(n.mean() / d.mean()) if len(n) and len(d) else None


def boot(eps, fn):
    rng = np.random.default_rng(SEED); v = []
    for _ in range(B):
        s = [eps[j] for j in rng.integers(0, len(eps), len(eps))]
        x = fn(s)
        if x is not None:
            v.append(x)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    files = sorted(CH.glob("ep*.npz"))
    zs = [np.load(f) for f in files]
    pol = [cells(z) for z in zs]; hol = [cells(z, hold=True) for z in zs]
    keep = [k for k, c in enumerate(pol) if c is not None]
    pol = [pol[k] for k in keep]; hol = [hol[k] for k in keep]
    r = {"episodes": len(files), "excluded_no_transition": len(files) - len(keep),
         "transitions_per_episode": [len(transitions(zs[k]["recorded"][:, 5])) for k in keep]}
    for name, eps in (("smolvla", pol), ("hold", hol)):
        r[name] = {str(H): {"R": ratio(eps, H), "ci95": boot(eps, lambda s, H=H: ratio(s, H))} for H in HS}
        r[name]["rise_50_minus_10"] = {"value": ratio(eps, 50) - ratio(eps, 10),
                                       "ci95": boot(eps, lambda s: ratio(s, 50) - ratio(s, 10))}
    both = list(zip(pol, hol))
    r["excess_vs_hold"] = {str(H): {"value": ratio(pol, H) - ratio(hol, H),
                                    "ci95": boot(both, lambda s, H=H: ratio([p for p, _ in s], H) - ratio([h for _, h in s], H))} for H in HS}
    r["by_position"] = {f"{a}-{a + 9}": ratio([e[(e[:, 0] >= a) & (e[:, 0] < a + 10)] for e in pol], 50) for a in range(0, 50, 10)}
    r["gripper"] = {str(H): ratio(pol, H, col=3) for H in HS}
    r["near_far_mean_deg_H50"] = {"near": float(np.concatenate([e[(e[:, 1] <= NEAR), 2] for e in pol]).mean()),
                                  "far": float(np.concatenate([e[(e[:, 1] > FAR), 2] for e in pol]).mean())}
    s = r["smolvla"]; R50, rise = s["50"], s["rise_50_minus_10"]
    first = R50["R"] >= 2 and R50["ci95"][0] > 1
    second = s["10"]["R"] < s["25"]["R"] < R50["R"] and rise["ci95"][0] > 0
    r["verdict"] = "concentrated as RACE says" if first and second else ("partly" if first else "not seen")
    OUT.mkdir(exist_ok=True); (OUT / "race.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
