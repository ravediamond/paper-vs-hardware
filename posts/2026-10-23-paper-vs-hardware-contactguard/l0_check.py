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

c = J("w1c_summary.json")["c1"]
check("imagined latent AUROC", c["imagined"]["auroc"], 0.70, 0.005)
check("current latent AUROC", c["current"]["auroc"], 0.82, 0.005)
check("scored episodes", c["scored_episodes"], 65, 0)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
