"""L0: recompute the AAC bars (A1, A3, A4) from expected/gate0.json and gate1_latency.json. Standard library only."""
import json, sys
from pathlib import Path

E = Path(__file__).resolve().parent / "expected"
g0, g1 = (json.loads((E / f).read_text()) for f in ("gate0.json", "gate1_latency.json"))
fails = 0


def check(name, got, want, tol=0.0005):
    global fails
    ok = abs(got - want) <= tol
    fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name}: {got:.4f} (post: {want}, tol {tol})")


a1 = g0["A1"]
check("A1 median h* transport (read)", a1["median_h_transport"], 2, 0)
check("A1 median h* grasp window (read)", a1["median_h_grasp"], 2, 0)
check("A1: transport >= grasp + 2 (recomputed; 0 = fail)", float(a1["median_h_transport"] >= a1["median_h_grasp"] + 2), 0, 0)
a3 = g0["A3"]
check("A3 AAC + overwrite, d = 8 (read)", a3["aac+overwrite_d8"], 0.973, 0.0005)
check("A3 fixed + overwrite, d = 8 (read)", a3["fixed+overwrite_d8"], 0.940, 0.0005)
check("A3: AAC <= fixed (recomputed; 0 = fail)", float(a3["aac+overwrite_d8"] <= a3["fixed+overwrite_d8"]), 0, 0)
n20 = g1["by_n"]["20"]["median_ms"]
check("A4 N = 20 decision, ms (read)", n20, 4751, 1)
check("A4: N = 20 <= 298 ms (recomputed; 0 = fail)", float(n20 <= 298.0), 0, 0)
check("timing at MAXN_SUPER (read; 1 = yes)", float("MAXN_SUPER" in g1["power"]), 1, 0)
print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
