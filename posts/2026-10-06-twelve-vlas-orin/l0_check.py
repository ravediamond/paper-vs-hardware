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

rows = {}
for line in open(E / "orin_nx.jsonl"):
    d = json.loads(line); rows[d["model"]] = d
want = {"Octo-Small": 27.4, "TurboVLA": 79.7, "VLA-JEPA": 176.7, "GR00T N1.7": 251.4, "BitVLA": 269.1, "SmolVLA": 411.2,
        "π0": 739.6, "Evo-1": 893.2}
for m, w in want.items():
    runs = rows[m]["defaults"]["runs"]
    check(f"{m}: min ms at defaults, best of {len(runs)} processes (recomputed from per-process runs)",
          min(r["min"] for r in runs if r.get("ok")), w, 0.05)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
