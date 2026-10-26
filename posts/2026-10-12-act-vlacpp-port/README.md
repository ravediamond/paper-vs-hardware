# Porting ACT to vla.cpp

Posts:
- https://shissorobotics.com/blog/2026/10/12/porting-act-to-vlacpp/

## Claim

ACT runs in vla.cpp (merged upstream as #38); the PyTorch ACT checkpoint scores 9/10 on the Orin NX gate.

## Compute

**Compute.** External GPU: no (Orin NX + Mac). Cost: $0; ACT training cost not recorded.

## What is not proven

The vla.cpp ACT path was not run through the 10-trial gate on the arm.

See `REPRODUCE.md`.
