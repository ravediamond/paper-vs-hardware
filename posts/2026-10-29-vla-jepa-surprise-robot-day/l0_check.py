"""L0: recount the robot day from expected/labels.csv and apply the frozen §29 branch for too few failures. Stdlib only."""
import csv, sys
from pathlib import Path

rows = list(csv.DictReader(open(Path(__file__).resolve().parent / "expected/labels.csv")))
fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name}: {got} (post: {want})")


check("rollouts", len(rows), 20)
check("successes", sum(r["outcome"] == "success" for r in rows), 20)
check("off-nominal rollouts", sum(r["stratum"] != "nominal" for r in rows), 9)
check("off-nominal successes", sum(r["stratum"] != "nominal" and r["outcome"] == "success" for r in rows), 9)
n_fail = sum(r["outcome"] == "failure" for r in rows)
check("verdict branch: failures < 6 -> not proven", "not proven" if n_fail < 6 else "scored", "not proven")
res = [float(r[k]) for r in rows for k in ("blue_residual_cm", "pink_residual_cm") if r[k]]
check("placement residual max <= 1 cm", max(res) <= 1.0, True)
print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
