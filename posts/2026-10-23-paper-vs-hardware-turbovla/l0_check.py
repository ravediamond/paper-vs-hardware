"""L0: recompute this post's numbers from the row-level files in expected/ (stdlib, numpy for tick files).
Where a number is a plain count or a summary with no rows behind it, it is read from the pinned result file and
compared; each line says which. Exit 1 on any mismatch."""
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
# recomputed from the per-frame / per-layout rows, not read from the summary fields
check("task swap median (deg), over 8 frames", statistics.median(r["task_swap"] for r in g["rows"]), 0.020, 0.0005)
d = g["per_layout_diffs"]
check("per-layout correct (share of 36 pan diffs > 0)", sum(x > 0 for x in d) / len(d), 0.389, 0.0005)
c = J("control.json")
check("LIBERO control (normalised units), median over 40 frames", statistics.median(r["task_swap_median"] for r in c["rows"]), 0.105, 0.0005)
check("LIBERO control in deg-equivalent pan", c["control_value_deg_equiv_pan"], 5.5, 0.05)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
