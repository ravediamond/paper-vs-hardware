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

def succ(f, pol):
    rows = [r for r in csv.DictReader(open(E / f)) if r["policy"] == pol]
    return sum(r["success"] == "True" for r in rows), len(rows)


for f, pol, want in (("trials_m1.csv", "act", 10), ("trials_m1.csv", "smolvla", 9),
                     ("trials_orin_nx.csv", "act", 9), ("trials_orin_nx.csv", "smolvla", 9)):
    s, n = succ(f, pol)
    check(f"{pol} {f[7:-4]} successes (of {n}, recomputed from trial rows)", s, want, 0)
import numpy as np  # noqa: E402
sp = []
for p in sorted((E / "ticks_2026-09-29_orin_nx").glob("smolvla_*.json")):
    s = np.array(json.loads(p.read_text())["select_action_ms"]); sp += s[s > 10 * np.median(s)].tolist()
check("SmolVLA chunk inference, Orin NX (ms, median of boundary calls)", float(np.median(sp)), 974, 1)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
