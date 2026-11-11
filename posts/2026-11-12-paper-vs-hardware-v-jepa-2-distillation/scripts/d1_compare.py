"""WS-110 D1 decision rows (D1_PROTOCOL.md §3, §7 G3-G4): the §30 term of the D1 student vs W1-scratch, paired by validation episode.

Same code path as gap5_compare.s30 (id_probe.py: cell both views, k = 5, ridge fit on teleop train, evaluated on teleop val; term = arm
(ii) - max((i), (iii)), the max inside each resample; 2,000 episode resamples, seed 0, one resample shared by all rows, percentile
2.5 / 97.5). Rows: W1 (stored §30 rows), W1_random_init (floor, reported), D1_best (best-val checkpoint, the decision row, GAP5
§2 / owner F6 / O2) and D1_final (final-step checkpoint, reported). "Beats" = lower CI bound > 0 (B1 wording "with CI > 0"; no margin).
G1b is not computed (G3); B1's random-teacher clause is not testable in this step (G4): the JSON says so.

    python d1_compare.py --id30 <W1 rows dir> --student-id30 <D1 rows dir> --out <json> [--rows csv] [--preds npz]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vla-jepa-world-model" / "wm_train"))
sys.path.insert(1, str(Path(__file__).resolve().parent))
import id_probe as I  # noqa: E402

N_BOOT = 2000
BASE = ("W1", "W1_random_init")


def term_of(per_enc: dict, sd, pick) -> float:
    a = {w: I.summarise(per_enc[w], sd, pick)["arm_residual_r2"] for w in ("i", "ii", "iii")}
    return a["ii"] - max(a["i"], a["iii"])


def pairs_for(encs) -> list:
    return [(e, b) for e in encs if e not in BASE for b in BASE] + [("W1", "W1_random_init")]


def pooled(per: dict, sd, pairs) -> dict:
    eps = sorted(per["W1"]["ii"])
    rng = np.random.default_rng(0)
    diff = {p: [] for p in pairs}
    for _ in range(N_BOOT):
        pick = [eps[i] for i in rng.integers(0, len(eps), len(eps))]
        t = {enc: term_of(per[enc], sd, pick) for enc in per}
        for a, b in pairs:
            diff[(a, b)].append(t[a] - t[b])
    return {f"{a}_minus_{b}": {"point": term_of(per[a], sd, eps) - term_of(per[b], sd, eps),
                               "ci95": [float(np.percentile(diff[(a, b)], 2.5)), float(np.percentile(diff[(a, b)], 97.5))]}
            for a, b in pairs}


def s30(id30: Path, student: Path, tags, rows=None, preds=None, prefix="D1") -> dict:
    I.ENCODERS = BASE
    data = {s: I.load_set(id30 / s) for s in ("teleop_train", "teleop_val")}
    students = [f"{prefix}_{t}" for t in tags]
    for s in data:
        eps, meta, lat = data[s]
        for name in students:
            z = np.load(student / s / f"{name}.npz")
            lat[name] = {}
            for e in eps:
                top, wr = z[f"ep{e:03d}_top"].astype(np.float32), z[f"ep{e:03d}_wrist"].astype(np.float32)
                assert len(top) == len(meta[e]["frame"]), (s, e, len(top), len(meta[e]["frame"]))
                lat[name][e] = {"top": top, "wrist": wr, "both": np.concatenate([top, wr], 1)}
    ends = {s: {e: len(d[1][e]["frame"]) for e in d[0]} for s, d in data.items()}
    anc = {s: I.anchors(d[1], d[0], ends[s]) for s, d in data.items()}
    sd = np.concatenate([m["action"] for m in data["teleop_train"][1].values()]).std(0) + 1e-9
    groups = np.array([e for e, _ in anc["teleop_train"]])
    k, out, per = 5, {}, {}
    for enc in (*BASE, *students):
        per[enc] = {}
        for which in ("i", "ii", "iii", "iv", "v"):
            if enc != "W1" and which in ("iv", "v"):
                continue
            Z = lambda s: {e: data[s][2][enc][e]["both"] for e in data[s][0]}  # noqa: E731
            Xtr, Atr, _ = I.build(anc["teleop_train"], data["teleop_train"][1], Z("teleop_train"), ends["teleop_train"], k, which,
                                  np.random.default_rng(I.SEED))
            pred, _ = I.fit_ridge(Xtr, Atr, groups)
            X, A, Q = I.build(anc["teleop_val"], data["teleop_val"][1], Z("teleop_val"), ends["teleop_val"], k, which,
                              np.random.default_rng(I.SEED))
            per[enc][which] = I.per_episode_metrics(pred(X), A, Q, np.array([e for e, _ in anc["teleop_val"]]), sd)
        arm = {w: I.summarise(p, sd, sorted(p))["arm_residual_r2"] for w, p in per[enc].items()}
        out[enc] = {"arm": arm, "term": arm["ii"] - max(arm["i"], arm["iii"])}
    out["W1"]["check_v_minus_iv"] = out["W1"]["arm"]["v"] - out["W1"]["arm"]["iv"]       # EXPERIMENTS line 128: +0.007
    eps = sorted(per["W1"]["ii"])
    if rows:
        with open(rows, "w") as f:
            f.write("encoder,input,episode,n,arm_residual_r2\n")
            for enc, p in per.items():
                for w, pe in p.items():
                    for e in sorted(pe):
                        f.write(f"{enc},{w},{e},{pe[e]['n']},{I.summarise(pe, sd, [e])['arm_residual_r2']:.6f}\n")
    if preds:
        np.savez_compressed(preds, sd=sd, **{f"{enc}__{w}__{f}": np.concatenate([pe[e][f] for e in sorted(pe)]) if f != "episode" else
                                             np.concatenate([np.full(pe[e]["n"], e) for e in sorted(pe)])
                                             for enc, p in per.items() for w, pe in p.items() for f in ("pred", "A", "Q", "episode")})
    out["pooled_pairs"] = pooled(per, sd, pairs_for(students))
    out["val_episodes"] = len(eps)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id30", type=Path, required=True)
    ap.add_argument("--student-id30", type=Path, required=True)
    ap.add_argument("--tags", nargs="+", default=["best", "final"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=Path)
    ap.add_argument("--preds", type=Path)
    ap.add_argument("--prefix", default="D1")
    a = ap.parse_args()
    s = s30(a.id30, a.student_id30, a.tags, a.rows, a.preds, a.prefix)
    pp = s["pooled_pairs"]
    res = {"protocol": "projects/jepa-distill-edge/D1_PROTOCOL.md", "s30_both_k5_ridge": s,
           "g1b": "not measured (D1_PROTOCOL §7 G3)",
           "decision": {"decision_row": f"{a.prefix}_best", "beats_W1": {t: pp[f"{a.prefix}_{t}_minus_W1"]["ci95"][0] > 0 for t in a.tags},
                        "beats_random_init_W1": {t: pp[f"{a.prefix}_{t}_minus_W1_random_init"]["ci95"][0] > 0 for t in a.tags},
                        "B1": "not proven in this step: the random-teacher student did not run (G4)"}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"terms": {e: s[e]["term"] for e in s if isinstance(s[e], dict) and "term" in s[e]}, "pairs": pp,
                      "W1_check_v_minus_iv": s["W1"]["check_v_minus_iv"], "decision": res["decision"]}, default=float))


if __name__ == "__main__":
    main()
