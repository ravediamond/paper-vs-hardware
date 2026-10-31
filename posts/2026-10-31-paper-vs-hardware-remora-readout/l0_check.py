"""L0: recompute every REMORA bar (P1, P2, P3) and the descriptive MAE from expected/readouts.jsonl (the four readouts
per query on the 34 test episodes), expected/target.jsonl (the error-aware target), the G15 transition labels and the
G14 windows. Standard library only. The bootstrap CIs (numpy's generator) are read from summary.json, not recomputed."""
import csv, json, sys
from pathlib import Path

E = Path(__file__).resolve().parent / "expected"
S = json.loads((E / "summary.json").read_text())
split = json.loads((E / "g15_split.json").read_text())
lab = {int(r["episode_index"]): (int(r["grasp_frame"]), int(r["place_frame"])) for r in csv.DictReader(open(E / "g15_transition_labels.csv"))}
win = {int(r["episode_index"]): (int(r["run_start_frame"]), int(r["run_end_frame"]))
       for r in csv.DictReader(open(E / "g14_episodes.csv")) if r["oscillating"] == "True"}
ro = {d["episode"]: d for d in map(json.loads, open(E / "readouts.jsonl"))}
tg = {d["episode"]: d for d in map(json.loads, open(E / "target.jsonl"))}
SLACK, LATE = 5, 30                                   # G15's score_event: premature if > 5 frames early, missed if > 30 late
fails = 0


def check(name, got, want, tol=0.0005):
    global fails
    ok = abs(got - want) <= tol
    fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name}: {got:.4f} (post: {want}, tol {tol})")


def score(fire, truth):
    if truth < 0:
        return "premature" if fire is not None else "correct"
    if fire is None or fire > truth + LATE:
        return "missed"
    return "premature" if fire < truth - SLACK else "on_time"


def first(y, fr, v):
    return next((f for f, x in zip(fr, y) if x >= v), None)


eps = split["test_episodes"]
check("test episodes", len(eps), 34, 0)
res = {}
for k in ("R0", "R1", "R2", "R3"):
    prem = miss = pres = 0; flips = []; rec = []; false = []; mae = []
    for e in eps:
        d, (g, p) = ro[e], lab[e]; fr, y = d["frames"], d[k]; t = tg[e]["target"]
        ev = [score(first(y, fr, 0.5), g), score(first(y, fr, 1.0), p)]
        prem += ev.count("premature"); miss += ev.count("missed"); pres += (g >= 0) + (p >= 0)
        flips.append(sum(a != b for a, b in zip(y, y[1:])))
        dec = [b < a for a, b in zip(y, y[1:])]
        if e in win:
            a_, c_ = win[e]; rec.append(any(dc and a_ <= f < c_ for dc, f in zip(dec, fr[1:])))
        else:
            false.append(any(dec))
        end = max([x for x in (g, p) if x >= 0], default=len(t) - 1)
        span = [(yh, t[min(f, len(t) - 1)]) for f, yh in zip(fr, y) if f <= end]
        if span:
            mae.append(sum(abs(a - b) for a, b in span) / len(span))
    res[k] = {"premature": prem / (2 * len(eps)), "missed": miss / max(pres, 1), "flips": flips,
              "recall": sum(rec) / len(rec), "windows": len(rec), "false": sum(false) / len(false), "mae": sum(mae) / len(mae)}
r1, r3 = res["R1"], res["R3"]
for k in ("R1", "R3"):
    check(f"{k} premature rate (recomputed)", res[k]["premature"], S[k]["premature_rate"])
    check(f"{k} missed rate (recomputed)", res[k]["missed"], S[k]["missed_rate"])
    check(f"{k} flips per episode (recomputed)", sum(res[k]["flips"]) / len(eps), S[k]["flips_mean"])
check("R0 premature (read: G15's stage machine fires, = G15 test 0.4853)", S["R0"]["premature_rate"], split["R0_test"]["premature_rate"])
check("R0 missed (read, = G15 test 0.4667)", S["R0"]["missed_rate"], split["R0_test"]["missed_rate"])
p1 = r3["premature"] <= r1["premature"] and r3["missed"] <= r1["missed"] + 0.02
check("P1: R3 premature <= R1 and missed <= R1 + 2 pt (recomputed; 1 = pass)", float(p1), 1, 0)
ratio = sum(r3["flips"]) / sum(r1["flips"])
check("P2: flips R3 / R1 (recomputed)", ratio, S["paired"]["flips_R3_over_R1"])
check("P2: ratio <= 0.5 and CI upper <= 0.7 (CI read; 0 = fail)", float(ratio <= 0.5 and S["paired"]["flips_ratio_ci95"][1] <= 0.7), 0, 0)
check("G14 windows on the test half", r3["windows"], 7, 0)
check("R3 regression recall (recomputed)", r3["recall"], S["R3"]["regression_recall"])
check("R3 false regressions (recomputed)", r3["false"], S["R3"]["false_regressions"])
check("P3: recall >= 0.5 and false <= 0.2 (recomputed; 0 = fail)", float(r3["recall"] >= 0.5 and r3["false"] <= 0.2), 0, 0)
check("MAE R3 vs target (recomputed)", r3["mae"], S["R3"]["MAE"])
check("MAE R1 vs target (recomputed)", r1["mae"], S["R1"]["MAE"])
print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
