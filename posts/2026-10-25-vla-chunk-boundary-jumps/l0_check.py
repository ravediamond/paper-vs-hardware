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

o = J("vlacpp_overwrite_summary.json")
for d, want in (("8", 0.710), ("12", 0.733), ("20", 0.715)):
    check(f"jump cut vs naive, d = {d}", o["jump_cut_vs_naive"][d], want, 0.0005)
check("max prefix error (deg)", o["overwrite_max_prefix_err_deg"], 0.0, 0)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
