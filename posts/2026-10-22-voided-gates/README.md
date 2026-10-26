# Two gates I voided, and the two-line check that would have caught them

Posts:
- https://shissorobotics.com/blog/2026/10/22/two-gates-i-voided/

## Claim

G11-A timed W1 with a random encoder (192 missing / 192 unexpected keys under transformers 5.5.4); G12 timed a stand-in model whose parts do not sum (135.4 + 230.3 != 318.2 ms). Both are void.

## Compute

**Compute.** External GPU: no (Orin NX). Cost: $0.

## What is not proven

The void numbers stay in `expected/` labelled void (`G11_G12_REPORT_void.md`); nothing is claimed from them.

See `REPRODUCE.md`.
