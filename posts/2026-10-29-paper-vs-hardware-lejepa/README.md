# Paper vs hardware 17: LeJEPA identifiability (partial, anisotropic recovery of the arm's joints: reproduces, gate 0)

Posts:
- https://shissorobotics.com/blog/2026/10/29/paper-vs-hardware-lejepa/

## Claim

Klindt, LeCun, Balestriero (arXiv 2605.26379): a SIGReg JEPA recovers the true latent state, and recovers it unevenly
when trained on goal-directed rather than random data. On the SO-101 the true latent is logged (joint positions), so
recovery is measurable directly: W1 (a SIGReg LeWM trained on teleop) recovers the joints with mean R² 0.565 and an
anisotropy of 0.639 (shoulder lift 0.91, wrist flex 0.27); a frozen DINOv2-S encoder (W2′) reaches 0.867 / 0.301. B1, B2
and B3 met (B3 with n = 2 models).

## Compute

**Compute.** External GPU: no (Orin NX). Cost: $0. Retries: one extraction redone after a red-channel-only bug
(`expected/gate0_redchannel_superseded.json`, first reported 0.624 / 0.582; same verdict).

## What is not proven

Gate 1 (the random-walk side: collect isotropic random-walk data on the arm, retrain W1's recipe, re-measure) has not
run. B3 compares only two models. The theorem's premises (Gaussian, stationary latents) do not hold for teleop.

See `REPRODUCE.md`.
