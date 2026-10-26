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

r = J("w1r.json")["record"]
check("W1-R regret (read from the pinned file)", r["W1R_regret"], 0.263, 0.0005)
check("W1 regret (read)", r["W1_regret"], 0.263, 0.0005)
check("action-only regret (read)", r["action_only_regret"], -0.0075, 0.0005)
check("verdict fail (1 = yes)", float(r["verdict"] == "fail"), 1, 0)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
