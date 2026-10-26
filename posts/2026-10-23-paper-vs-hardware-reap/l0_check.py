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

v1 = J("monitors.json")["primary_50"]["risk_logistic"]
check("v1 logistic AUROC", v1["auroc"], 0.93, 0.005)
check("v1 caught (of 8)", v1["caught"], 6, 0)
check("v1 false interventions (of 42)", v1["false_interventions"], 7, 0)
v2 = J("monitors_v2.json")["primary_50"]
# v2 post numbers = median over the 5 seeds of each metric (run_reap_v2.py), recomputed from per_seed
med = lambda m, k: statistics.median(s[k] for s in v2[m]["per_seed"])
check("v2 logistic AUROC, true actions (median of 5 seeds)", med("risk_logistic:true", "auroc"), 0.789, 0.0005)
check("v2 logistic AUROC, shuffled actions (median of 5 seeds)", med("risk_logistic:shuffled", "auroc"), 0.708, 0.0005)
check("v2 true > shuffled on every seed", sum(t["auroc"] > s["auroc"] for t, s in zip(v2["risk_logistic:true"]["per_seed"], v2["risk_logistic:shuffled"]["per_seed"])), 5, 0)
check("v2 caught (of 8, median of seeds)", med("risk_logistic:true", "caught"), 3, 0)
check("v2 false interventions (of 42, median of seeds)", med("risk_logistic:true", "fi"), 5, 0)
check("v2 common horizon (frames)", v2["T_common"]["Put the blue bowl in the pink bowl."], 374, 0)

sp = list(csv.DictReader(open(E / "onsets_spot_check.csv")))
off = sorted(int(r["gemini_onset"]) - int(r["claude_onset"]) for r in sp)
check("onset spot check, n (count)", len(sp), 5, 0)
check("onset offset median, frames (recomputed from Gemini vs by-eye onsets)", statistics.median(off), 6, 0)
check("largest onset offset, frames (recomputed)", max(off), 289, 0)
print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
