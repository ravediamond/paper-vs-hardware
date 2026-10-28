# Paper vs hardware 14: REKI (oscillation metric on our rollouts)

Posts:
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-reki/

## Claim

REKI's oscillation metric on the 69 published rollouts: 14/69 = 20.3 % oscillate, 6/25 failures (24 %) and 8/44 successes (18.2 %). The G12 latency gate is void (see the voided-gates folder).

## Compute

**Compute.** Oscillation audit on the Orin NX: $0. REKI on vast.ai: training $11.25 (16.07 GPU h, RTX PRO 5000), smoke $0.60 over 3 attempts (1 host never reachable, 1 failed on git ownership), serving for the 40 arm trials $1.20 (RTX 3090, two sessions). Total $13.05.

## What is not proven

Metric on recorded actions only; the 10+10 video check was not done; REKI itself was run on the arm afterwards (below).

## Follow-up on the arm (2026-10-27)

REKI trained on the bowls data (VLM frozen, 3,500 steps, one seed), served from a rented RTX 3090, 20 trials per instruction, 25 s each, judged at rest: **19/20** blue-in-pink, **19/20** pink-in-blue (SmolVLA 9/10 under the same protocol). Oscillation by the metric above: **1 of 40** trials, against 20 % on the 69 earlier rollouts. Four trials hit a ≥ 5 s network stall and were counted invalid and replaced (rule fixed before the last one's outcome). Not done: the K = 1 (memory off) ablation on the same batch. Video (two successes, one failure; network pauses cut): [`media/reki_so101_highlights_2026-10-27.mp4`](media/reki_so101_highlights_2026-10-27.mp4).

See `REPRODUCE.md`.
