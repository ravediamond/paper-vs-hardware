# Verify at rest: six mid-episode monitors fail in one day

Posts:
- https://shissorobotics.com/blog/2026/10/19/verify-at-rest-six-monitors-fail-in-one-day/
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-vla-corrector/

## Claim

W1-A (L2 surprise 2/8 at 2/42, AUROC 0.86; cosine drift 0/8, AUROC 0.30), W1-B, G11-B (3/25), G15 (48.5 % premature), G16-B (41/42 successes flagged), judge tau (0/8 caught): none reaches its bar.

## Compute

**Compute.** External GPU: no (Orin NX). External API: yes, Gemini 3.8 Flash: G15 ~$0.45 (168 calls) + ~$0.42 of Claude Code checks, G16-B ~$0.26, judge tau ~$1-2 (339 calls, cost not reported by the API; estimate). Total ~$2.1-3.1 of API. Retries: none recorded.

## What is not proven

8 failures in the 50-episode set; one world model; VLM monitors with one prompt each.

See `REPRODUCE.md`.
