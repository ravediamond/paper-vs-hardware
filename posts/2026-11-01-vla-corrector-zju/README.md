# VLA-Corrector's latent-vision monitor (arXiv 2607.01804) on our rollouts (2026-11-01)

A residual MLP trained on demonstrations only to predict the change of frozen SmolVLA vision latents from the action,
read with the paper's robust trigger on the 50-episode monitor set (42 successes, 8 failures). Bar ≥ 5/8 failures at
≤ 2/42 successes. Result: 8/8 at 40/42 — **fail**; AUROC of max E 0.72 vs 0.67 with shuffled actions. Not the
register's VLA-Corrector (2609.06508).
