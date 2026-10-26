"""L0: recompute this post's numbers from expected/ (stdlib only). Exit 1 on any mismatch."""
import csv, json, statistics, sys
from pathlib import Path

E = Path(__file__).resolve().parent / "expected"
fails = 0


def check(name, got, want, tol):
    global fails
    ok = abs(got - want) <= tol
    fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name}: {got:.4f} (post: {want}, tol {tol})")


def J(name):
    return json.loads((E / name).read_text())

a = J("action_sensitivity.json")["by_k"]["30"]
check("K=30 open-loop error, true actions", a["true_mse"], 0.31, 0.005)
check("K=30 copy-last error", a["copy_last_mse"], 0.89, 0.005)
g = J("g1b_range_w1.json")
check("G1b best top-1 over all cells", max(c["accuracy"] for c in g["cells"].values()), 0.60, 0.006)
ao = J("g1_investigation.json")["pair"]["action_only"]
check("action-only top-1, min over d", min(v["top1"] for v in ao.values()), 0.68, 0.006)
check("action-only top-1, max over d", max(v["top1"] for v in ao.values()), 0.90, 0.006)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
