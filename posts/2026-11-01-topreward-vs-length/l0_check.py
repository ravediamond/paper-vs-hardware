"""L0: recompute TOPReward's success score and its within-policy AUROC, and episode length's, from the per-episode
log-probabilities in expected/ab.jsonl (ArmnetBench) and expected/own.jsonl; check them and the reading against
expected/topreward.json. Standard library only; the bootstrap CIs are read from topreward.json."""
import csv, json, statistics as st
from pathlib import Path

E = Path(__file__).resolve().parent / "expected"; R = json.loads((E / "topreward.json").read_text()); fails = 0


def check(name, got, want, tol=0.0005):
    global fails
    ok = abs(got - want) <= tol; fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name}: {got:.4f} (expected {want:.4f})")


def auroc(y, s):
    pos = [v for v, t in zip(s, y) if t]; neg = [v for v, t in zip(s, y) if not t]
    if not pos or not neg:
        return None
    order = sorted(range(len(s)), key=lambda i: s[i]); ranks = [0.0] * len(s); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(r for r, t in zip(ranks, y) if t)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def zs(x):
    m, sd = st.mean(x), st.pstdev(x); return [(v - m) / sd for v in x]


def scores(rows):
    top = [st.mean([a - b for a, b in zip(r["r_true"][-3:], r["r_false"][-3:])]) - st.mean([a - b for a, b in zip(r["r_true"][:3], r["r_false"][:3])]) for r in rows]
    yn = [r["yn_full"][0] - r["yn_full"][1] for r in rows]
    return [a + b for a, b in zip(zs(top), zs(yn))]


ab = [json.loads(l) for l in open(E / "ab.jsonl")]
ab = [r for r in ab if "error" not in r and r.get("success_class") in ("successful", "failure") and r.get("policy_type") != "teleoperated"]
check("ArmnetBench episodes", len(ab), R["armnetbench"]["n"], 0)
y = [r["success_class"] == "successful" for r in ab]; s = scores(ab); ln = [-r["length"] for r in ab]
check("TOPReward pooled AUROC", auroc(y, s), R["armnetbench"]["s_succ"]["pooled"])
check("length pooled AUROC", auroc(y, ln), R["armnetbench"]["neg_len"]["pooled"])


def within(score):
    num = den = 0.0
    for p in sorted({r["policy_type"] for r in ab}):
        idx = [i for i, r in enumerate(ab) if r["policy_type"] == p]
        yy = [y[i] for i in idx]; a = auroc(yy, [score[i] for i in idx]); w = sum(yy) * (len(yy) - sum(yy))
        if a is not None:
            num += a * w; den += w
    return num / den


check("TOPReward within-policy AUROC", within(s), R["armnetbench"]["s_succ"]["within"])
check("length within-policy AUROC", within(ln), R["armnetbench"]["neg_len"]["within"])
ci = R["armnetbench"]["delta_within_vs_length_ci95"]
reading = "beats" if ci[0] > 0 else ("below" if ci[1] < 0 else "matches")
print(f"{'ok ' if reading == R['armnetbench']['reading_full'] else 'BAD'} reading on full episodes: TOPReward {reading} elapsed time (paired CI {ci[0]:.3f} to {ci[1]:.3f})")
fails += reading != R["armnetbench"]["reading_full"]
lab = {int(r["episode_index"]): r["outcome"] for r in csv.DictReader(open(E / "own_rollout_labels.csv"))}
own = [r for r in map(json.loads, open(E / "own.jsonl")) if "error" not in r and lab.get(r["episode_index"]) in ("success", "failure")]
check("own TOPReward AUROC", auroc([lab[r["episode_index"]] == "success" for r in own], scores(own)), R["own"]["s_succ"]["auroc"])
print("L0 PASS" if not fails else f"L0 FAIL ({fails})")
