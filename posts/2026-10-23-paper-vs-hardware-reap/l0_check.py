"""L0: recompute this post's numbers from expected/ (stdlib only). Exit 1 on any mismatch."""
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
check("v2 logistic AUROC, true actions", v2["risk_logistic:true"]["auroc"], 0.789, 0.0005)
check("v2 logistic AUROC, shuffled actions", v2["risk_logistic:shuffled"]["auroc"], 0.708, 0.0005)
check("v2 caught (of 8)", v2["risk_logistic:true"]["caught"], 3, 0)
check("v2 false interventions (of 42)", v2["risk_logistic:true"]["fi"], 5, 0)
check("v2 common horizon (frames)", v2["T_common"]["Put the blue bowl in the pink bowl."], 374, 0)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
