"""L0: recompute the PACT bars (A, B-P1, B-P2) from expected/gate_a.json, gate_b.json, ver_ratio.json and the WS-057
A2 latency file they use. Standard library only."""
import json, math, sys
from pathlib import Path

E = Path(__file__).resolve().parent / "expected"
a, b, r, a2 = (json.loads((E / f).read_text()) for f in ("gate_a.json", "gate_b.json", "ver_ratio.json", "a2_latency.json"))
fails = 0
TICK = 1000 / 30


def check(name, got, want, tol=0.0005):
    global fails
    ok = abs(got - want) <= tol
    fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name}: {got:.4f} (post: {want}, tol {tol})")


k8 = b["k8"]
check("PACT mean error d = 8, deg (read)", k8["mean_error_deg"]["pact"], 2.274, 0.0005)
check("random mean error d = 8, deg (read)", k8["mean_error_deg"]["random"], 2.543, 0.0005)
check("B-P1: CI of PACT - random entirely < 0 (recomputed; 1 = pass)", float(k8["pact_minus_random"]["ci95"][1] < 0), 1, 0)
check("B-P2: CI of PACT - Old-FlowRes entirely < 0 (recomputed; 1 = pass)", float(k8["pact_minus_old"]["ci95"][1] < 0), 1, 0)
t1 = a2["by_n"]["10"]["median"]["rtt"]; sp = a2["split"]
t_pact = 4 * t1 + sp["vision_ms"] + sp["prefill_ms"] + r["ratio"] * sp["per_step_denoise_ms"]
check("T_PACT, ms (recomputed from A2 and the ratio)", t_pact, a["T_PACT_ms"], 0.01)
check("T_PACT, ms (post)", t_pact, 1200.1, 0.1)
dn, dp = math.ceil(t1 / TICK), math.ceil(t_pact / TICK)
check("d_naive, ticks (recomputed)", dn, 8, 0)
check("d_PACT, ticks (recomputed)", dp, 37, 0)
check("A: d_PACT >= d_naive + 2 (recomputed; 1 = fail)", float(dp >= dn + 2), 1, 0)
check("B-P3 window exists: d_PACT + 25 <= 50 (recomputed; 0 = not computable)", float(dp + 25 <= 50), 0, 0)
print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
