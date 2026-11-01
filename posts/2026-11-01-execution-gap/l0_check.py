"""L0: apply the frozen readings of C1 (dead zone) and C2 (tracking gap) to expected/exec_gap.json, and print the
normalised command-execution residual. The per-frame values come from the public rollouts and demos (REPRODUCE, L1)."""
import json
from pathlib import Path

R = json.loads((Path(__file__).resolve().parent / "expected/exec_gap.json").read_text()); fails = 0
dz = R["c1_dead_zone_rollouts"]; big = [j for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll") if dz[j]["ci95"][0] >= 0.3]
c1 = "sub-servo" if len(big) >= 3 else "not sub-servo"; ok = c1 == R["c1_reading"]; fails += not ok
print(f"{'ok ' if ok else 'BAD'} C1: {c1} (settle band CI lower >= 0.3 deg on {len(big)} of 5 joints: {big})")
c = R["c2"]; r, ci = c["gripper_ratio"], c["gripper_ratio_ci95"]
c2 = "execution matters" if r >= 2 and ci[0] > 1 else ("not seen" if ci[0] <= 1 <= ci[1] else "partial")
ok = c2 == R["c2_reading"]; fails += not ok
print(f"{'ok ' if ok else 'BAD'} C2: {c2} (gripper gap failures / successes {r:.2f}, CI {ci[0]:.2f}-{ci[1]:.2f})")
print(f"ok  normalised residual, rollouts {R['normalised_residual_rollouts']['mean']:.3f}; demos leader vs follower {R['normalised_residual_demos_leader_follower']:.3f}")
print("L0 PASS" if not fails else "L0 FAIL")
