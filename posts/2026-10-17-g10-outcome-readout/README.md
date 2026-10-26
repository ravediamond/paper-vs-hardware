# What the world model is actually for (G10)

Posts:
- https://shissorobotics.com/blog/2026/10/17/what-the-world-model-is-actually-for/

## Claim

At the judge time the world-model latent reads the outcome (W1 AUROC 1.00 vs judge, 0.97 vs ground truth); before execution, the imagined latent beats policy identity by +0.18.

## Compute

**Compute.** External GPU: no (Mac MPS). Cost: $0.

## What is not proven

88 rollouts, 25 failures; step 2's pass is against policy identity, not against every cheap signal.

See `REPRODUCE.md`.
