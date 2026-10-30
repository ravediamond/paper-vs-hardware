"""L0: recompute the horizon x depth bars (A1, A2) and the A4 reading from expected/. Standard library only."""
import csv, json, statistics, sys
from pathlib import Path

E = Path(__file__).resolve().parent / "expected"
a1 = json.loads((E / "a1_cloud.json").read_text())["summary"]["2"]
a2 = json.loads((E / "a2_latency.json").read_text())
fails = 0


def check(name, got, want, tol=0.0005):
    global fails
    ok = abs(got - want) <= tol
    fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name}: {got:.4f} (post: {want}, tol {tol})")


r, dmu, mx = a1["R_ratio"]["median"], a1["dmu"]["median"], a1["max_abs_centroid_deg"]["median"]
check("A1 median R(2)/R(10) (read)", r, 0.78, 0.005)
a1_pass = r <= 0.70 and dmu <= 0.05 and mx <= 3.0
a1_fail = r > 0.85 or dmu > 0.10
check("A1 not proven: neither pass nor fail (recomputed; 1 = not proven)", float(not a1_pass and not a1_fail), 1, 0)
n2, n10 = a2["by_n"]["2"]["median"]["rtt"], a2["by_n"]["10"]["median"]["rtt"]
check("A2 N = 10 median, ms (read)", n10, 237.5, 0.05)
check("A2 N = 2 / N = 10 (recomputed)", n2 / n10, 0.83, 0.005)
check("A2 pass: <= 0.85 and N = 10 within 5 % of 238 ms (recomputed; 1 = pass)",
      float(n2 / n10 <= 0.85 and abs(n10 / 238 - 1) <= 0.05), 1, 0)
check("A2 at MAXN_SUPER (read; 1 = yes)", float("MAXN_SUPER" in a2["power"]), 1, 0)


def throttled(clock):
    ref = statistics.median(clock[:60]); run = best = 0
    for v in clock:
        run = run + 1 if v < 0.95 * ref else 0; best = max(best, run)
    return best >= 10


READINGS = {"maxn": "not applicable here", "25w": "not applicable here"}
for tag, want in READINGS.items():
    rows = list(csv.DictReader(open(E / f"a4_thermal_{tag}.csv")))
    t = {n: throttled([float(x["gpu_mhz"]) for x in rows if x["n"] == n and x["gpu_mhz"]]) for n in ("10", "2")}
    reading = ("not applicable here" if not t["10"] else "replicates" if not t["2"] else "does not replicate")
    print(f"{'ok ' if reading == want else 'BAD'} A4 {tag}: {reading} (post: {want})"); fails += reading != want
print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
