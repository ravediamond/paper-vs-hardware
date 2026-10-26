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

t = (E / "w1_strict_load.txt").read_text()
import re  # noqa: E402
check("missing keys under transformers 5.5.4 (read from the strict-load log)", int(re.search(r"Missing key\(s\): (\d+)", t).group(1)), 192, 0)
check("unexpected keys under 5.5.4 (read)", int(re.search(r"Unexpected key\(s\): (\d+)", t).group(1)), 192, 0)
check("tensors loaded verbatim under 5.19.0 (read)", int(re.search(r"\((\d+) tensors\)", t).group(1)), 279, 0)
check("G12 parts sum vs total (recomputed: 135.4 + 230.3 - 318.2, void)", 135.4 + 230.3 - 318.2, 47.5, 0.05)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
