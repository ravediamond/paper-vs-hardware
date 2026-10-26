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

def succ(f, pol):
    rows = [r for r in csv.DictReader(open(E / f)) if r["policy"] == pol]
    return sum(r["success"] == "True" for r in rows), len(rows)


for f, pol, want in (("trials_m1.csv", "act", 10), ("trials_m1.csv", "smolvla", 9),
                     ("trials_orin_nx.csv", "act", 9), ("trials_orin_nx.csv", "smolvla", 9)):
    s, n = succ(f, pol)
    check(f"{pol} {f[7:-4]} successes (of {n}, recomputed from trial rows)", s, want, 0)
import numpy as np  # noqa: E402
sp = []
for p in sorted((E / "ticks_2026-09-29_orin_nx").glob("smolvla_*.json")):
    s = np.array(json.loads(p.read_text())["select_action_ms"]); sp += s[s > 10 * np.median(s)].tolist()
check("SmolVLA chunk inference, Orin NX (ms, median of boundary calls)", float(np.median(sp)), 974, 1)


def summarize(folder, prefix, chunk):  # as shared/gate/chunk_latency.py
    spikes, rest, on_b, blocked, wall = [], [], 0, 0.0, 0.0
    for f in sorted((E / folder).glob(prefix + "*.json")):
        t = json.loads(f.read_text()); s = np.array(t["select_action_ms"]); idx = np.where(s > 10 * np.median(s))[0]
        on_b += int(np.sum(idx % chunk == 0)); spikes += s[idx].tolist(); rest += np.delete(s, idx).tolist()
        ts = np.array(t["tick_ts"]); wall += ts[-1] - ts[0]; blocked += s[idx].sum() / 1e3
    return 100 * on_b / len(spikes), float(np.median(spikes)), 100 * blocked / wall


for folder, pol, chunk, w_on, w_ms, w_bl in (("ticks_2026-09-29_orin_nx", "smolvla_", 50, 100, 974, 39),
                                             ("ticks_2026-09-29_m1", "smolvla_", 50, 98, 954, 39),
                                             ("ticks_2026-09-29_orin_nx", "act_", 100, 100, 59, 2),
                                             ("ticks_2026-09-29_m1", "act_", 100, 100, 122, 4)):
    on, ms, bl = summarize(folder, pol, chunk)
    tag = f"{pol[:-1]} {folder[17:]}"
    check(f"{tag}: spikes on a chunk boundary, % (recomputed from ticks)", on, w_on, 0.5)
    check(f"{tag}: chunk inference ms (recomputed)", ms, w_ms, 0.5)
    check(f"{tag}: sync loop blocked, % (recomputed)", bl, w_bl, 0.5)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
