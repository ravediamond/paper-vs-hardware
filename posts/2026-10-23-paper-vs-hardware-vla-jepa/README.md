# VLA-JEPA on the SO-101 for $9.7, and why best-of-N reranking is a no-go

Posts:
- https://shissorobotics.com/blog/2026/10/08/vla-jepa-for-10-dollars-and-why-reranking-is-a-no-go/
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-vla-jepa/

## Claim

VLA-JEPA fine-tuned for ~$9.7 scores 9/10 on the arm (355 ms per chunk). H0: picking the best of 8 action samples would reach 0.76 of a random pick's error (null 0.92) and costs 1.84x (643 vs 349 ms): no-go.

## Compute

**Compute.** External GPU: yes (vast.ai, training). Cost: ~$9.70 for the fine-tune (retries not itemised in the repo); H0 and the arm gate on the Orin NX, $0.

## What is not proven

n = 10 arm trials; H0 on the teleop val split; one checkpoint.

See `REPRODUCE.md`.
