# Paper vs hardware 5: D-JEPA (a decision-aligned operator that never fired)

Posts:
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-d-jepa/

## Claim

D-JEPA's relational ranking operator on frozen W1 leaves G1b regret at 0.263 (W1 alone 0.263) against an action-only baseline at -0.007: fail on both bars.

## Compute

**Compute.** External GPU: no (Orin NX, ~17 min). Cost: $0.

## What is not proven

Proxy labels (executed chunk = success); top-2 ceiling labelled exploratory in the post.

See `REPRODUCE.md`.
