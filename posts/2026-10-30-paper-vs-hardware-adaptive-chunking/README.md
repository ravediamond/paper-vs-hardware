# Paper vs hardware 19: Adaptive Action Chunking (AAC) (the entropy rule returns its floor and N = 20 costs 4.75 s on the Orin: fail)

Posts:
- https://shissorobotics.com/blog/2026/10/30/paper-vs-hardware-adaptive-chunking/

**Claim tested.** Adaptive Action Chunking (Liang et al., CVPR 2026, arXiv 2604.04161) picks each chunk's length from
the entropy of N sampled chunks: chunks get longer where the policy is confident (transport) and shorter where it is
not (grasp), which cuts the jump at chunk boundaries, for a cost that a batched sampler keeps small.

**Bars (frozen before any number, `PROTOCOL.md`; implementation choices `IMPLEMENTATION_NOTE.md`).**
A1 median h* in transport ≥ median h* in the grasp window + 2 · A3 median jump with AAC + prefix overwrite ≤ fixed
chunk + overwrite at d = 8 · A4 an N = 20 decision ≤ 298 ms on the Orin (25 % over one 238 ms chunk).

**Result.**

| bar | value | verdict |
|---|---|---|
| A1 | h* transport **2** (n = 72) vs grasp window **2** (n = 24) | fail |
| A3 | **0.973** vs 0.940 (median jump ratio, 254 decisions) | fail |
| A4 | **4,751 ms** for N = 20 (sequential; vla.cpp has no batched sampling) | fail |

Post-hoc (not in the protocol): h* equals the floor ξ = 2 in 219 of 254 decisions (86 %); SmolVLA's sample entropy
grows fastest at the first step, so the rule returns the floor. AAC's earlier switch alone lowers the naive jump
(3.42 → 2.28) but adds nothing to the prefix overwrite (0.94).

**Verdict: fail on this rig.**

**Compute.** External GPU: no. Gate 0 on an Orin NX at 25 W (PyTorch, ~3 h); gate 1 on the Orin at MAXN_SUPER +
`jetson_clocks` (vla.cpp, ~5 min). Cost: $0.

**What is not proven.** Gate 2 (arm) is not run: SmolVLA is at ceiling on the bowls task, so a +15-point effect could
not show. A batched sampler (not in vla.cpp) would change A4, not A1. One policy, one task family.
