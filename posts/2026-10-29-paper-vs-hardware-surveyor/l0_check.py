"""L0: recompute the four bars (S1-S4) from expected/gate0.json and gate1.json. Standard library only."""
import json, sys
from pathlib import Path

E = Path(__file__).resolve().parent / "expected"
g0, g1 = (json.loads((E / f).read_text()) for f in ("gate0.json", "gate1.json"))
fails = 0


def check(name, got, want, tol=0.0005):
    global fails
    ok = abs(got - want) <= tol
    fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name}: {got:.4f} (post: {want}, tol {tol})")


tau, ne10 = g0["equivalent"]["median"], g0["non_equivalent_pooled"]["p10"]
check("tau = median rel of equivalent pairs (read)", tau, 1.249)
check("S1: tau < p10 of non-equivalent (recomputed; 0 = fail)", float(tau < ne10), 0, 0)
check("S2: tau <= 0.40 (recomputed; 0 = fail)", float(tau <= 0.40), 0, 0)
for s in ("S10", "S30"):
    p = g1[s]["primary_50"]
    check(f"S3 at {s}: >= 5/8 at <= 2/42 (recomputed; 0 = fail)", float(p["caught"] >= 5 and p["fi"] <= 2), 0, 0)
check("S4 at S30: true AUROC - shuffled >= 0.05 (recomputed; 1 = met)",
      float(g1["S30"]["primary_50"]["auroc"] >= g1["S30"]["primary_50_shuffled"]["auroc"] + 0.05), 1, 0)
check("S4 at S10 (recomputed; 0 = not met)",
      float(g1["S10"]["primary_50"]["auroc"] >= g1["S10"]["primary_50_shuffled"]["auroc"] + 0.05), 0, 0)
print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
