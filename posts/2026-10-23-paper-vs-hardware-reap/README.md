# ReAP: a failure-risk head that looked strong was a shortcut

Posts:
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-reap/
- https://shissorobotics.com/blog/2026/10/26/audit-your-labels-where-they-matter/

On the site: shissorobotics.com result 2.

## Claim

v1 risk head AUROC 0.93 (6/8 at 7/42); with a ramp target, no time feature, common-horizon truncation and shuffled / zero action controls (v2), AUROC 0.789 vs 0.708 shuffled, 3/8 at 5/42: fail at the bar (>= 5/8 at <= 1/20 false interventions).

## Compute

**Compute.** External GPU: no (Orin NX). External API: Gemini 3.8 Flash for the v1 onset labels, 52 calls, cents (exact amount not reported). v2 uses no API. Cost: < $0.10.

## What is not proven

8 failures in the primary set; v1 onsets are VLM-marked (spot check n = 5, two retries marked late).

See `REPRODUCE.md`.
