# A correctness fix upstream made my checkpoint 47 % worse (VLA-JEPA rotary, LeRobot #4835)

Posts:
- https://shissorobotics.com/blog/2026/10/26/upstream-fix-checkpoint-47-percent-worse/

## Claim

On its own 25 validation episodes (200 windows), the legacy-trained world predictor's L1 is 1.175 under the old rotary, 1.731 under the fixed one, 2.235 for copy-last: R = 1.473, worse on 25/25 episodes (DEGRADES).

## Compute

**Compute.** External GPU: no (Orin NX, 25W, ~10 min). Cost: $0.

## What is not proven

One checkpoint, in distribution; the LIBERO checkpoint not run (disk).

See `REPRODUCE.md`.
