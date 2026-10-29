# Paper vs hardware 18: SURVEYOR (the accept threshold has no operating point on real-robot latents: fail)

Posts:
- https://shissorobotics.com/blog/2026/10/29/paper-vs-hardware-surveyor/

## Claim

SURVEYOR (NeurIPS 2026 workshop; anonymous submission, simulation only: PushT, Reacher, OGBench-Cube, Two-Room)
accepts a drafted plan when the relative distance between imagined and reached latents stays below τ (0.20 in the
paper). On W1's latents of 69 real SO-101 rollouts, τ calibrated the paper's way is **1.249**: two at-rest success
frames from different episodes are as far apart as a success and a mid-episode frame (S1, S2 fail). Used as a failure
monitor, the accept rule catches 1/8 (S = 10 frames) and 3/8 (S = 30) on the 50-rollout set at 1 and 6 false
interventions out of 42 (S3 fail at both); at S = 30 it carries signal beyond a shuffled drafter (AUROC 0.756 vs 0.560,
S4 met), but warns after the failure onset.

## Compute

**Compute.** External GPU: no (Orin NX). Cost: $0. Retries: none.

## What is not proven

FF-JEPA's diffusion subgoal drafter (GDM) is not available, so W1's own imagination stands in for it; no closed-loop
planning; nothing in simulation. 8 failures in the primary set.

See `REPRODUCE.md`.
