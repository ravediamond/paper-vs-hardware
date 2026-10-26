import sys
sys.path.insert(0, "/work/scripts")
import numpy as np, pandas as pd
from run_g14_oscillation_audit import download, load_frames, episode_states, chunk_velocities, reversal_runs, ARM_JOINTS
root = download("ravediamond/so101-bowls-policy-rollouts")
frames = load_frames(root)
lab = pd.read_csv("/work/data/rollouts/so101-bowls-policy-rollouts_labels.csv").set_index("episode_index")
res = pd.read_csv("/work/shared/gate/data/2026-10-18_g14/g14_episodes.csv").set_index("episode_index")
np.set_printoptions(precision=1, suppress=True, linewidth=160)
for ep in [int(x) for x in sys.argv[1:]]:
    r = res.loc[ep]; L = lab.loc[ep]
    q = episode_states(frames, ep)
    rs = int(L.reset_start)
    print(f"\n=== ep {ep} {r.policy} {r.outcome} {r.failure_mode if isinstance(r.failure_mode,str) else ''} len={len(q)} reset_start={rs} steps={r.chunk_steps} flagged={r.oscillating}")
    if rs >= 0: q = q[:rs]
    vel, disp, starts = chunk_velocities(q[:, ARM_JOINTS], int(r.chunk_steps))
    rev = reversal_runs(vel, disp)
    for k in range(len(vel)):
        d = q[starts[k] + int(r.chunk_steps) - 1, ARM_JOINTS] - q[starts[k], ARM_JOINTS]
        cos = "" if k == 0 else f"{float(vel[k-1]@vel[k]/(np.linalg.norm(vel[k-1])*np.linalg.norm(vel[k])+1e-9)):+.2f}"
        print(f"  k={k:2d} f{starts[k]:4d} |d|={disp[k]:6.1f}  d={d}  cos={cos:>5}  {'REV' if rev[k] else ''}  grip {q[starts[k],5]:.0f}->{q[starts[k]+int(r.chunk_steps)-1,5]:.0f}")
