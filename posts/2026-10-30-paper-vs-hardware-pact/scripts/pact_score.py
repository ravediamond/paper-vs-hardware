#!/usr/bin/env python3
"""WS-059 gate B score: rows.jsonl (pact_flowres.py) -> gate_b.json. Means are per-episode means averaged over the 20
episodes; paired differences per episode; 95 % CIs by episode bootstrap (2,000 resamples, seed 0). Reconstructed for
publication on 2026-10-31 (the original was computed inline): the means and differences reproduce expected/gate_b.json
and the CIs exactly (all 34 numbers, checked cold on 2026-10-31).

    python scripts/pact_score.py out/pact/rows.jsonl out/pact/gate_b.json
"""
import collections, json, sys
import numpy as np

SEL = ("single", "random", "old_flowres", "pact", "mean_of_n", "oracle")
rows = [json.loads(l) for l in open(sys.argv[1])]
res = {"episodes": len({r["episode"] for r in rows}), "boundaries": {}}
for k in sorted({r["k"] for r in rows}):
    ep = collections.defaultdict(list)
    for r in rows:
        if r["k"] == k:
            ep[r["episode"]].append(r)
    res["boundaries"][str(k)] = sum(len(v) for v in ep.values())
    per = {e: {s: float(np.mean([x[s] for x in v])) for s in SEL} for e, v in sorted(ep.items())}
    keys = list(per)
    out = {"mean_error_deg": {s: round(float(np.mean([per[e][s] for e in keys])), 4) for s in SEL}}
    for name, a, b in (("pact_minus_random", "pact", "random"), ("pact_minus_old", "pact", "old_flowres"),
                       ("old_minus_random", "old_flowres", "random")):
        d = np.array([per[e][a] - per[e][b] for e in keys]); rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        out[name] = {"mean": round(float(d.mean()), 4), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res[f"k{k}"] = out
k8 = res["k8"]
res["B-P1"] = "pass" if k8["pact_minus_random"]["ci95"][1] < 0 else "fail"
res["B-P2"] = "pass" if k8["pact_minus_old"]["ci95"][1] < 0 else "fail"
res["sanity_recorded_below_median_frac"] = round(sum(r["rec_below_median"] for r in rows) / len(rows), 3)
open(sys.argv[2], "w").write(json.dumps(res, indent=1)); print(json.dumps({k: res[k] for k in ("B-P1", "B-P2")}))
