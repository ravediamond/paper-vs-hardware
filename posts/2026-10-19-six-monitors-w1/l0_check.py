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

a = J("w1a_summary.json")["g16_set"]
check("W1-A L2: failures caught of 8 (read)", a["l2"]["caught_failed"], 2, 0)
check("W1-A L2 AUROC (read)", a["l2"]["auroc"], 0.86, 0.001)
check("W1-A cosine drift AUROC (read)", a["cos"]["auroc"], 0.30, 0.005)
check("W1-A live reflex: successes flagged of 42 (read)", a["reflex_live_rule"]["success_flagged"], 42, 0)
b = J("w1b_summary.json")["test"]
check("W1-B premature rate (recomputed from counts)", b["premature"] / b["events"], 0.2206, 0.0001)
check("W1-B missed rate (recomputed from counts)", b["missed"] / b["present"], 0.3111, 0.0001)
c = J("g11b_cal.json")["chosen"]
check("G11-B-cal failures flagged of 25 (read)", c["failure_flagged"], 3, 0)
g = J("g16b_summary.json")["g16_set"]
check("G16-B successes flagged of 42 (read)", g["success_flagged"], 41, 0)
rows = list(csv.DictReader(open(E / "g16b_episodes.csv")))
check("G16-B episodes in the per-episode file", len(rows), 69, 0)
t = J("judge_tau.json")
check("judge tau: failures caught of 8 (read)", t["caught_failed"], 0, 0)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
