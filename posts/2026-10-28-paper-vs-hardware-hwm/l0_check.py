"""L0: recompute the three bars (H1-H3) from the per-horizon means in expected/gate0.json, and read the post-hoc
baselines from expected/posthoc_baselines.json; each line says which. Exit 1 on any mismatch. Standard library only."""
import json, sys
from pathlib import Path

E = Path(__file__).resolve().parent / "expected"
fails = 0


def check(name, got, want, tol=0.0005):
    global fails
    ok = abs(got - want) <= tol
    fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name}: {got:.4f} (post: {want}, tol {tol})")


g = json.loads((E / "gate0.json").read_text())
p = json.loads((E / "posthoc_baselines.json").read_text())
low, high, H = g["low"]["mean"], g["learned"]["err_mean"], g["H_frames"]
below = [h < lo for h, lo in zip(high, low)]
hstar = next((H[k] for k in range(len(H)) if all(below[k:]) and not any(below[:k])), None)
check("H1: crossover frame in [30, 60] (recomputed; none = 0)", float(hstar is not None and 30 <= hstar <= 60), 0, 0)
check("H2: high / low at 60 frames (recomputed)", high[-1] / low[-1], 1.35, 0.005)
check("H2: ratio <= 0.95 (recomputed)", float(high[-1] / low[-1] <= 0.95), 0, 0)
check("H3: learned cosine (read)", g["learned"]["cos_mean"], 0.608)
check("H3: displacement cosine (read)", g["disp"]["cos_mean"], 0.551)
check("H3: learned >= displacement (recomputed)", float(g["learned"]["cos_mean"] >= g["disp"]["cos_mean"]), 1, 0)
check("W1 error at 0.73 s (read)", low[0], 0.382)
check("W1 error at 2.0 s (read)", low[-1], 0.563)
check("high-level error at 2.0 s (read)", high[-1], 0.759)
check("post-hoc: copy z_t at 2.0 s (read)", p["copy_z_t"]["mean"][-1], 0.954)
check("post-hoc: train-mean latent at 2.0 s (read)", p["train_mean_latent"]["mean"][-1], 0.806)
print("L0", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
