# ContactGuard: an event-anchored probe on imagined latents did not transfer

Posts:
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-contactguard/
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-fail-detect-and-safe/

On the site: shissorobotics.com result 6.

## Claim

On the 69 published rollouts, a logistic probe on W1's imagined latent 30 steps after the grasp anchor scores AUROC 0.70 vs 0.82 for the current latent (bar: >= 0.85 and >= +0.05): the paper's claim reversed.

## Compute

**Compute.** External GPU: no (Orin NX, offline). Cost: $0. The same run holds W1-C1 (ContactGuard) and W1-C2 (FAIL-Detect / SAFE band, `w1c_summary.json` block `c2`).

## What is not proven

65 scored episodes, 21 failures; one world model (W1); grasp anchor from the gripper command.

See `REPRODUCE.md`.
