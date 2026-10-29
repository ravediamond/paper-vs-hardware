"""WS-050 gate 1: the accept rule as a monitor (PROTOCOL.md). numpy only.

    python3 surveyor_gate1.py <repo root> <results dir>
Reads outputs/reap/latents, projects/surveyor/out/imagine_S{10,30}.npz, results/gate0.json (τ), the W1-A summary
(primary 50 = in_g16_set), the human outcomes and the Gemini onsets. Writes <results>/gate1.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def rel(a, b):
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(a, axis=-1)


def auroc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def score(rows, tau, onsets):
    fail = [r for r in rows if r["outcome"] == "failure"]
    succ = [r for r in rows if r["outcome"] == "success"]
    ttw = [(onsets[r["ep"]] - r["fire"]) / 30 for r in fail if r["fire"] is not None and r["ep"] in onsets]
    return {"caught": sum(r["fire"] is not None for r in fail), "failures": len(fail),
            "fi": sum(r["fire"] is not None for r in succ), "successes": len(succ),
            "auroc": round(auroc([r["max"] for r in fail], [r["max"] for r in succ]), 4),
            "ttw_vs_gemini_s_median": (round(float(np.median(ttw)), 2) if ttw else None)}


def main():
    root, res_dir = Path(sys.argv[1]), Path(sys.argv[2])
    tau = json.loads((res_dir / "gate0.json").read_text())["bars"]["S1"]["tau"]
    w1a = json.loads((root / "shared/gate/data/2026-10-19_w1_screening/w1a_summary.json").read_text())["episodes"]
    meta = {int(e["episode_index"]): e for e in w1a}
    onsets = {int(r.episode_index): int(r.onset_frame) for r in
              pd.read_csv(root / "shared/gate/data/2026-10-23_onsets_vlm/onsets.csv").itertuples()}
    out = {"tau": tau, "drafter": "W1's own imagination (stand-in for FF-JEPA's GDM drafter)"}
    for step in (10, 30):
        im = np.load(root / f"projects/surveyor/out/imagine_S{step}.npz")
        eps = sorted(int(k[1:]) for k in im.files if k.startswith("b"))
        Z = {e: np.load(root / f"outputs/reap/latents/ep{e:03d}.npy") for e in eps}
        D = {}
        for e in eps:
            b, w = im[f"b{e}"], im[f"w{e}"]
            D[e] = rel(Z[e][b + step], w) if len(b) else np.zeros(0)

        def rows_for(sel, dmap):
            out_rows = []
            for e in sel:
                d = dmap[e]
                hit = np.flatnonzero(d > tau)
                b = im[f"b{e}"]
                out_rows.append({"ep": e, "outcome": meta[e]["outcome"], "max": float(d.max()) if len(d) else 0.0,
                                 "fire": (int(b[hit[0]] + step) if len(hit) else None)})
            return out_rows

        prim = [e for e in eps if meta[e]["in_g16_set"]]
        allv = [e for e in eps if meta[e]["outcome"] in ("success", "failure")]
        rng = np.random.default_rng(1234)
        perm = list(prim)
        while any(a == b for a, b in zip(prim, perm)):
            perm = list(rng.permutation(prim))
        Dsh = {}
        for e, f in zip(prim, perm):
            n = min(len(im[f"b{e}"]), len(im[f"b{f}"]))
            Dsh[e] = rel(Z[e][im[f"b{e}"][:n] + step], im[f"w{f}"][:n])
        p_true, p_shuf = score(rows_for(prim, D), tau, onsets), score(rows_for(prim, Dsh), tau, onsets)
        out[f"S{step}"] = {"primary_50": p_true, "primary_50_shuffled": p_shuf, "all_69": score(rows_for(allv, D), tau, onsets),
                           "delta_success_median": round(float(np.median(np.concatenate([D[e] for e in prim if meta[e]["outcome"] == "success"]))), 4)}
        out[f"S{step}"]["bars"] = {
            "S3": {"met": bool(p_true["caught"] >= 5 and p_true["fi"] <= 2)},
            "S4": {"true": p_true["auroc"], "shuffled": p_shuf["auroc"], "met": bool(p_true["auroc"] >= p_shuf["auroc"] + 0.05)}}
    (res_dir / "gate1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
