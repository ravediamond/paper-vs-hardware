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

o = J("vlacpp_overwrite_summary.json")
s = o["summary"]
# cut = 1 - overwrite jump / naive jump, recomputed per delay from the two summaries
for d, want in (("8", 0.710), ("12", 0.733), ("20", 0.715)):
    check(f"jump cut vs naive, d = {d}", 1 - s["overwrite"][d]["jump_ratio"] / s["naive"][d]["jump_ratio"], want, 0.0005)
    check(f"boundaries, d = {d}", s["naive"][d]["boundaries"], 254, 0)
check("overwrite cost per chunk (ms), max over d", max(s["overwrite"][d]["ms"] - s["naive"][d]["ms"] for d in ("8", "12", "20")), 0.4, 0.1)
check("max prefix error (deg)", max(s["overwrite"][d]["prefix_err"] for d in ("8", "12", "20")), 0.0, 0)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
