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

w2p = J("g1_final_w2prime_best.json")
w2 = J("g1_final_w2_best.json")
best = max(c["accuracy"] for c in w2["g1b"]["cells"].values()) if "g1b" in w2 and "cells" in w2["g1b"] else None
if best is not None:
    check("W2 best-val G1b best top-1 (recomputed over cells)", best, 0.61, 0.006)
check("W2′ open loop / copy-last at K = 30 (read from the pinned file)", w2p["openloop_over_copy_k30"], 0.633, 0.0005)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
