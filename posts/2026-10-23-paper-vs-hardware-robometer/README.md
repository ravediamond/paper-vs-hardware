# Robometer-4B as a failure judge vs episode length; a deterministic judge at rest

Posts:
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-robometer/
- https://shissorobotics.com/blog/2026/10/16/judging-a-rollout-the-zero-false-positive-rule/

On the site: shissorobotics.com result 7.

## Claim

Zero-shot Robometer-4B on ArmnetBench (2099 rollouts): pooled AUROC 0.818, under the 0.85 bar; on our 88 rollouts a deterministic top-camera judge at rest (G9c-3) reaches 94.3 % with 0 false positives.

## Compute

**Compute.** External GPU: yes (vast.ai). Robometer-4B zero-shot on ArmnetBench ~$5, re-score of the 462 OOM episodes ~$1, own rollouts ~$0.25, and the LoRA run (G9b) ~$2.3-3.3 that stopped at preprocessing and never trained (5.0 h, invoice not found). Total ~$8.55-9.55, all estimates. Retries counted: the 462-episode OOM re-score and the failed G9b run. The deterministic judge (G9c family): Mac / Orin CPU, $0.

## What is not proven

Robometer zero-shot only (the LoRA run never trained); judge accuracy 94.3 % is a narrow fail of its 95 % bar; reset marks by eye on 11 episodes.

See `REPRODUCE.md`.
