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

def flags(rows, k):
    s = [r for r in rows if r["outcome"] == "success"]; f = [r for r in rows if r["outcome"] != "success"]
    return sum(int(r["score"]) <= k for r in s), sum(int(r["score"]) <= k for r in f), len(s), len(f)


def auroc(rows):
    pos = [-int(r["score"]) for r in rows if r["outcome"] != "success"]; neg = [-int(r["score"]) for r in rows if r["outcome"] == "success"]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


fl = list(csv.DictReader(open(E / "g_flora_scores.csv")))
s, f, ns, nf = flags(fl, 4)
check(f"Flora: successes flagged at k = 4 (of {ns}, recomputed)", s, 2, 0)
check(f"Flora: failures flagged at k = 4 (of {nf}, recomputed)", f, 8, 0)
check("Flora AUROC (recomputed from scores)", auroc(fl), 1.0, 0.0005)
ab = list(csv.DictReader(open(E / "g_flora_abl_scores.csv")))
q27 = [r for r in ab if r["arm"] == "qwen27b"]
s, f, _, _ = flags(q27, 4)
check("Qwen3.5-27B: successes flagged at k = 4 (recomputed)", s, 2, 0)
check("Qwen3.5-27B: failures flagged at k = 4 (recomputed)", f, 7, 0)
check("Qwen3.5-27B AUROC (recomputed)", auroc(q27), 0.9137, 0.0005)
check("Gemini 3.8 Flash AUROC (recomputed)", auroc([r for r in ab if r["arm"] == "gemini38flash"]), 0.9643, 0.0005)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
