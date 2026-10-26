# Robometer-4B as a failure judge vs episode length; a deterministic judge at rest

Post: https://ravediamond.github.io/blog/2026/10/23/paper-vs-hardware-robometer/ (shissorobotics.com result 5).

## Claim

Zero-shot Robometer-4B on ArmnetBench (2099 rollouts): pooled AUROC 0.818, under the 0.85 bar; on our 88 rollouts a deterministic top-camera judge at rest (G9c-3) reaches 94.3 % with 0 false positives.

## What is not proven

Robometer zero-shot only (the LoRA run never trained); judge accuracy 94.3 % is a narrow fail of its 95 % bar; reset marks by eye on 11 episodes.

See `REPRODUCE.md`.
