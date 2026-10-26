# Paper vs hardware 14: REKI (oscillation metric on our rollouts)

Posts:
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-reki/

## Claim

REKI's oscillation metric on the 69 published rollouts: 14/69 = 20.3 % oscillate, 6/25 failures (24 %) and 8/44 successes (18.2 %). The G12 latency gate is void (see the voided-gates folder).

## Compute

**Compute.** External GPU: no (Orin NX). Cost: $0 for the oscillation audit. REKI SO-101 training (WS-025): smoke $0.60 over 3 attempts (1 host never reachable, 1 failed on git ownership) + full run in progress (~$11 projected); added when the run ends.

## What is not proven

Metric on recorded actions only; the 10+10 video check was not done; REKI itself is not run here yet (WS-025 / 026 results will be added).

See `REPRODUCE.md`.
