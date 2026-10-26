# Latent distance is not progress (G6a, G6b / H-JEPA)

Posts:
- https://shissorobotics.com/blog/2026/10/14/latent-distance-is-not-progress/
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-h-jepa/

## Claim

Distance to a goal latent does not track task progress: T1 monotonicity fails for every variant (G6a -0.32 to -0.50; G6b W1 -0.72, H-JEPA level 2 -0.73 vs -0.80 bar), false stalls 32-52 %.

## Compute

**Compute.** External GPU: no (Mac MPS). Cost: $0.

## What is not proven

Teleop episodes only (no failures) for T1; one W1 / W2 training each.

See `REPRODUCE.md`.
