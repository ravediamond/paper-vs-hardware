# Benchmarking twelve VLAs on an Orin NX for vla.cpp

Posts:
- https://shissorobotics.com/blog/2026/10/06/benchmarking-twelve-vlas-on-an-orin-nx/

## Claim

Minimum ms per inference at vla.cpp defaults on an Orin NX 16 GB: Octo 27.4, TurboVLA 79.7, VLA-JEPA 176.7, GR00T N1.7 251.4, BitVLA 269.1, SmolVLA 411.2, pi0 739.6, Evo-1 893.2 (upstream report #36).

## Compute

**Compute.** External GPU: no (Orin NX only, about 3 h of benchmarking). Cost: $0.

## What is not proven

Engine latency on fixed inputs, not on the robot loop; one board.

See `REPRODUCE.md`.
