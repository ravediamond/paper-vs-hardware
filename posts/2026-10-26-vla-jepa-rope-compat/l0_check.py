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

rows = J("results.json")["rows"]
eps = sorted({r["episode"] for r in rows})
per = {a: [statistics.mean(r[a] for r in rows if r["episode"] == e) for e in eps] for a in ("legacy", "fixed", "copy_last")}
check("windows (count)", len(rows), 200, 0)
check("episodes (count)", len(eps), 25, 0)
L, F, C = (statistics.mean(per[a]) for a in ("legacy", "fixed", "copy_last"))
check("L1 legacy, mean over episodes (recomputed from window rows)", L, 1.175, 0.0005)
check("L1 fixed (recomputed)", F, 1.731, 0.0005)
check("L1 copy-last (recomputed)", C, 2.235, 0.0005)
check("R = fixed / legacy (recomputed)", F / L, 1.473, 0.0005)
check("episodes where fixed is worse (recomputed)", sum(f > l for f, l in zip(per["fixed"], per["legacy"])), 25, 0)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
