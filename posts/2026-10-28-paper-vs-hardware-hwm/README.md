# Paper vs hardware 16: HWM, hierarchical latent world models (no crossover on 0.6 h of data: not proven)

Posts:
- https://shissorobotics.com/blog/2026/10/28/paper-vs-hardware-hwm/

## Claim

Zhang et al. (CoRL 2026, arXiv 2604.03208), Fig. 6: on held-out data the low-level world model has the lower latent
error up to ~1 s, a high-level model over learned macro-actions the lower error from ~1.5 s, crossing near 1.25 s.
On SO-101 bowls data (W1 as the low level, the paper's high level trained on W1's frozen latents) there is no
crossover: W1 0.382 ± 0.005 → 0.563 ± 0.007 from 0.73 to 2 s, the high level 0.769 ± 0.009 → 0.759 ± 0.009.
H1 fail, H2 fail (ratio 1.35 at 2 s), H3 met (learned macro-actions 0.608 ± 0.007 vs net displacement 0.551 ± 0.006).
**Verdict: not proven**, because of an open confound: ~0.6 h of one task here against ~130 h of DROID + RoboSet.

## Compute

**Compute.** External GPU: yes (vast.ai RTX 3090, high-level training, 3 × 20 000 steps): ≈ $0.4, including one
invalid run (red-only frames, `DEVIATION_2026-10-28.md`) and one host that never started. Latent extraction on the
Orin NX: $0.

## What is not proven

The data-scale confound: the high level may simply need far more data than the low level; this run cannot separate
that from "no crossover on this arm". Post-hoc only (not in `PROTOCOL.md`): the copy-z_t and train-mean baselines and
the overfitting reading (last-batch train loss 0.15 vs val 0.76, different metrics). SEs are over 700 anchors from 25
episodes, which are correlated, so they understate episode-level uncertainty. No planning, no arm.

See `REPRODUCE.md`.
