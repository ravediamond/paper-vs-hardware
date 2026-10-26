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

rows = list(csv.DictReader(open(E / "trials_vla_jepa.csv")))
check("VLA-JEPA arm successes (of %d, recomputed from trial rows)" % len(rows), sum(r["success"] == "True" for r in rows), 9, 0)
md = (E / "h0_latency_summary.md").read_text()
import re  # noqa: E402
tot = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"^\| (\d) \|(?:[^|]*\|){4} ([0-9.]+) \(", md, re.M)}
check("H0 latency N = 8 (ms, read from the pinned summary)", tot[8], 643, 1)
check("H0 latency N = 1 (ms, read from the pinned summary)", tot[1], 349, 1)
check("H0 ratio N = 8 / N = 1 (recomputed)", tot[8] / tot[1], 1.84, 0.01)
check("H0 oracle / mean error, overall (read from the pinned report)",
      float(re.search(r"\*\*0\.(\d+)\*\* overall", (E / "h0_report.md").read_text()).group(1)) / 100, 0.76, 0.005)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
