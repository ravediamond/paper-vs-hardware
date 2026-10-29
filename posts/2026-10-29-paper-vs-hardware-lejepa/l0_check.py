"""L0: recompute B1-B3 from the per-joint R2 in expected/gate0.json (the corrected run). Standard library only."""
import json, sys
from pathlib import Path

g = json.loads((Path(__file__).resolve().parent / "expected/gate0.json").read_text())
fails = 0


def check(name, got, want, tol=0.0005):
    global fails
    ok = abs(got - want) <= tol
    fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name}: {got:.4f} (post: {want}, tol {tol})")


for m, want_r2, want_an in (("W1", 0.565, 0.639), ("W2p", 0.867, 0.301)):
    r = g["models"][m]["r2_per_joint"].values()
    check(f"{m} mean R2 (recomputed from 6 joints)", sum(r) / len(r), want_r2)
    check(f"{m} anisotropy max - min (recomputed)", max(r) - min(r), want_an)
w1 = g["models"]["W1"]
check("B1: W1 mean R2 < 0.80 (recomputed)", float(sum(w1["r2_per_joint"].values()) / 6 < 0.80), 1, 0)
check("B2: W1 anisotropy >= 0.20 (recomputed)", float(max(w1["r2_per_joint"].values()) - min(w1["r2_per_joint"].values()) >= 0.20), 1, 0)
hi_r2 = max(g["models"], key=lambda k: g["models"][k]["mean_r2"]); hi_g1b = max(g["models"], key=lambda k: g["models"][k]["g1b_top1"])
check("B3: higher R2 model has higher G1b top-1 (recomputed, n = 2)", float(hi_r2 == hi_g1b), 1, 0)
print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
