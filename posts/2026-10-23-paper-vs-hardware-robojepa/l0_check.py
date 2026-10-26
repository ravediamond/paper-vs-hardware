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

a = J("gateA_summary.json")
for k, want in (("100m_150k_settinga", 0.105), ("300m_300k_settinga", 0.118), ("1b_300k_settinga", 0.158)):
    m = a[k]["mean_1_9"]
    check(f"{k}: reduction vs copy-last (recomputed from mean L1)", 1 - m["model_true"] / m["copy_last"], want, 0.001)
for k in ("100m_150k_settinga", "300m_300k_settinga", "1b_300k_settinga"):
    m = a[k]["mean_1_9"]
    v = 1 - m["model_true"] / m["model_shuffled_cons"]
    check(f"{k}: reduction vs shuffled actions >= 0.089 (recomputed)", float(v >= 0.089), 1, 0)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
