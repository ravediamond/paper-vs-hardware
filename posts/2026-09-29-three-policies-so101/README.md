# Three policies on a $300 arm, and the gate harness

Posts:
- https://shissorobotics.com/blog/2026/09/29/three-policies-on-a-300-dollar-arm/

## Claim

10-trial gates on the bowls task: ACT 10/10 (M1) and 9/10 (Orin NX), SmolVLA 9/10 on both; SmolVLA's chunk inference 974 ms on the Orin NX.

## Compute

**Compute.** External GPU: none for this post (Orin NX and an M1 Mac; 2 x 2 x 10 trials of 25 s on the arm). Training of the ACT and SmolVLA checkpoints: cost not recorded in the research repo.

## What is not proven

n = 10 trials per policy and machine; one task; success judged by the operator.

See `REPRODUCE.md`.
