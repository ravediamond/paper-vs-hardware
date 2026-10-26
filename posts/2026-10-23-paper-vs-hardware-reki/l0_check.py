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

rows = list(csv.DictReader(open(E / "g14_episodes.csv")))
osc = [r for r in rows if r["oscillating"] == "True"]
check("episodes", len(rows), 69, 0)
check("oscillating, all (recomputed from episode rows)", len(osc), 14, 0)
check("share oscillating, all (recomputed)", len(osc) / len(rows), 0.203, 0.0005)
f = [r for r in rows if r["outcome"] != "success"]
check("oscillating among failures (recomputed)", sum(r["oscillating"] == "True" for r in f), 6, 0)
check("share among failures (recomputed)", sum(r["oscillating"] == "True" for r in f) / len(f), 0.24, 0.0005)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
