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

g = J("gate_path_a.json")
check("task swap median (deg)", g["task_swap_median_deg"], 0.020, 0.0005)
check("per-layout correct", g["per_layout_correct"], 0.389, 0.0005)
c = J("control.json")
check("LIBERO control (normalised units)", c["control_value_norm_units"], 0.105, 0.0005)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
