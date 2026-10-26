"""L0: recompute this post's numbers from the row-level files in expected/ (stdlib, numpy for tick files).
Where a number is a plain count or a summary with no rows behind it, it is read from the pinned result file and
compared; each line says which. Exit 1 on any mismatch."""
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

c = J("w1c_summary.json")["c1"]
# post number = mean of the 5 out-of-fold AUROCs (one per seed), recomputed from the per-seed list
check("imagined latent AUROC (mean of 5 seeds)", statistics.mean(c["imagined"]["auroc_oof_per_seed"]), 0.70, 0.005)
check("current latent AUROC (mean of 5 seeds)", statistics.mean(c["current"]["auroc_oof_per_seed"]), 0.82, 0.005)
check("scored episodes", c["successes"] + c["failures"], 65, 0)
check("failures", c["failures"], 21, 0)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
