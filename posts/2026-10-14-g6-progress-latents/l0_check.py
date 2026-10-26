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

b = J("g6b_results.json")
check("G6b W1 T1 Spearman rho (read from the pinned file)", b["level1_w1"]["t1_spearman_rho"], -0.72, 0.005)
check("G6b H-JEPA T1 Spearman rho (read from the pinned file)", b["level2_hjepa"]["t1_spearman_rho"], -0.7268, 0.0001)
check("G6b H-JEPA false stall rate (read)", b["level2_hjepa"]["false_stall_rate"], 0.52, 0)
check("G6b verdict is FAIL (1 = yes)", float(b["overall_verdict"] == "FAIL"), 1, 0)
a = J("g6a_results.json")
check("G6a stall rule, false stalls on val (read)", a["stall"]["val_false_stall_rate_indicative"], 0.32, 0)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
