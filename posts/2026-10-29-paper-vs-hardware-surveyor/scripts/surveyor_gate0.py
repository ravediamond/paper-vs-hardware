"""WS-050 gate 0: τ on real-robot latents (PROTOCOL.md). numpy only.

    python3 surveyor_gate0.py <latents dir with epNNN.npy> <results dir>   # reads <results>/judge.csv, writes gate0.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def rel(a, b):
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(a, axis=-1)


def q(x, p):
    return round(float(np.percentile(x, p)), 4)


def main():
    lat, out = Path(sys.argv[1]), Path(sys.argv[2])
    j = pd.read_csv(out / "judge.csv")
    j = j[~j.invalid]
    Z = {int(e): np.load(lat / f"ep{int(e):03d}.npy") for e in j.episode}
    for r in j.itertuples():
        assert len(Z[r.episode]) == r.n_frames, (r.episode, len(Z[r.episode]), r.n_frames)
    zj = {int(r.episode): Z[r.episode][r.judge_frame] for r in j.itertuples()}
    zmid = {e: z[len(z) // 2] for e, z in Z.items()}
    succ = [int(e) for e in j[j.verdict == "success"].episode]
    fail = [int(e) for e in j[j.verdict == "failure"].episode]
    eq = np.array([rel(zj[a], zj[b]) for a in succ for b in succ if a != b])
    ne_mid = np.array([rel(zj[a], zmid[b]) for a in succ for b in Z if b != a])
    ne_fail = np.array([rel(zj[a], zj[b]) for a in succ for b in fail])
    ne = np.concatenate([ne_mid, ne_fail])
    tau = float(np.median(eq))
    res = {"episodes": {"success": len(succ), "failure": len(fail), "invalid": 0},
           "equivalent": {"n": len(eq), "median": round(tau, 4), "p10": q(eq, 10), "p90": q(eq, 90)},
           "non_equivalent_mid": {"n": len(ne_mid), "median": q(ne_mid, 50), "p10": q(ne_mid, 10), "p90": q(ne_mid, 90)},
           "non_equivalent_fail_end": {"n": len(ne_fail), "median": q(ne_fail, 50), "p10": q(ne_fail, 10), "p90": q(ne_fail, 90)},
           "non_equivalent_pooled": {"n": len(ne), "median": q(ne, 50), "p10": q(ne, 10), "p90": q(ne, 90)},
           "auroc_eq_vs_ne": round(float((ne[None, :] > eq[:, None]).mean() + 0.5 * (ne[None, :] == eq[:, None]).mean()), 4),
           "paper_tau": 0.20}
    res["bars"] = {"S1": {"tau": round(tau, 4), "ne_p10": q(ne, 10), "met": bool(tau < np.percentile(ne, 10))},
                   "S2": {"tau": round(tau, 4), "bar": 0.40, "met": bool(tau <= 0.40)}}
    (out / "gate0.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
