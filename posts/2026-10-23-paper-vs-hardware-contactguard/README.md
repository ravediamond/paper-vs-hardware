# ContactGuard: an event-anchored probe on imagined latents did not transfer

Post: https://ravediamond.github.io/blog/2026/10/23/paper-vs-hardware-contactguard/ (shissorobotics.com result 6).

## Claim

On the 69 published rollouts, a logistic probe on W1's imagined latent 30 steps after the grasp anchor scores AUROC 0.70 vs 0.82 for the current latent (bar: >= 0.85 and >= +0.05): the paper's claim reversed.

## What is not proven

65 scored episodes, 21 failures; one world model (W1); grasp anchor from the gripper command.

See `REPRODUCE.md`.
