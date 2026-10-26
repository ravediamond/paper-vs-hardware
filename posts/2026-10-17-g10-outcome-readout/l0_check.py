"""L0: recompute from the row-level files in expected/; where a number is a plain count or a summary with no rows
behind it, it is read from the pinned result file and compared; each line says which. Exit 1 on any mismatch."""
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

g = J("g10_results.json")
w1 = g["step1_readout"]["W1 (Top + Wrist fused, 192d)"]
check("Step 1 W1 AUROC vs judge (read from the pinned file)", w1["auroc_judge"], 1.0, 0)
check("Step 1 W1 AUROC vs ground truth (read)", w1["auroc_gt"], 0.97, 0.001)
d = g["step2_anticipation"]["paired_differences"]["(3) Imagined - (5) Policy Identity"]
check("Step 2 imagined - policy identity (read)", d["diff"], 0.1792, 0.0001)
check("test episodes valid (read)", g["test_set"]["valid_episodes"], 87, 0)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
