"""L0: recompute this post's numbers from the row-level files in expected/ (stdlib, numpy for tick files).
Where a number is a plain count or a summary with no rows behind it, it is read from the pinned result file and
compared; each line says which. Exit 1 on any mismatch."""
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

import numpy as np  # noqa: E402  (numpy for the tick summary; see env/requirements.txt)


def chunk_ms(folder, prefix, chunk):
    spikes = []
    for f in sorted((E / folder).glob(prefix + "*.json")):
        s = np.array(json.loads(f.read_text())["select_action_ms"])
        spikes += s[s > 10 * np.median(s)].tolist()
    return float(np.median(spikes))


check("PyTorch SmolVLA chunk inference (ms)", chunk_ms("ticks_2026-09-29_orin_nx", "smolvla_", 50), 974, 1)
check("vla.cpp _fa chunk inference (ms)", chunk_ms("ticks_2026-10-01_orin_nx_vlacpp", "smolvla_vlacpp_fa_", 50), 238, 1)

print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
