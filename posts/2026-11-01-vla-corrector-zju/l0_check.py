"""L0: recount the flags and the AUROC of max E_t from expected/lvm_episodes.json (one record per episode of the
50-episode monitor set) and apply the bar (>= 5/8 failures at <= 2/42 successes). Standard library only."""
import json
from pathlib import Path

E = Path(__file__).resolve().parent / "expected"; S = json.loads((E / "lvm_summary.json").read_text())
ep = json.loads((E / "lvm_episodes.json").read_text()); fails = 0
F = [e for e in ep if e["failure"]]; G = [e for e in ep if not e["failure"]]
kf, ks = sum(e["flagged"] for e in F), sum(e["flagged"] for e in G)
pos = [e["max_E"] for e in F]; neg = [e["max_E"] for e in G]
auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
for name, got, want in (("failures flagged", kf, S["failures_flagged"]), ("successes flagged", ks, S["successes_flagged"]),
                        ("AUROC max E", round(auc, 4), round(S["auroc_maxE"], 4))):
    ok = got == want; fails += not ok; print(f"{'ok ' if ok else 'BAD'} {name}: {got} (expected {want})")
p = kf >= 5 and ks <= 2; ok = p == S["pass"]; fails += not ok
print(f"{'ok ' if ok else 'BAD'} bar: {'pass' if p else 'fail'} ({kf}/{len(F)} failures at {ks}/{len(G)} successes)")
print("L0 PASS" if not fails else "L0 FAIL")
