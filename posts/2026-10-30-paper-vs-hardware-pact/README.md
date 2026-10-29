# Paper vs hardware: PACT-VLA on an SO-101 + Jetson

## Claim

PACT-VLA (Zhu et al., arXiv 2610.28019) samples several action chunks and picks the one whose flow-matching residual
(FlowRes) is lowest under the **pre-commit** observation, the latest frame before the chunk is committed, for **no
added control-step delay**. On SmolVLA and the SO-101 bowls episodes, the selection works offline (B-P1, B-P2 pass:
−0.27° against a random pick and against the stale-observation residual), but on the Orin the four candidates and the
extra pass cost 1,200 ms per cycle against 237 ms, 37 control ticks against 8 (A fails), and at that delay a 50-step
chunk cannot fill the executed window (B-P3 not computable).

**Verdict: replicates offline, does not transfer to the Orin.**

Protocol `PROTOCOL.md` (frozen 2026-10-29, before any number). Scripts `scripts/pact_flowres.py` (gate B) and
`scripts/pact_latency.py` (gate A), committed before their numbers (`15bcb89`). Policy `smolvla_2026-09-24_14-34-40`
@ `c1d76f6`; the 20 G13b episodes of `so101_bowls_all`, 254 boundaries per delay, s = 25.

## Measurement checks (2 episodes, 31 boundaries, before gate B)

Candidate 0 bit-identical to a fresh single sample ✔ · the recorded future scores below the median candidate on 68 %
of boundaries (≥ 60 %) ✔ · FlowRes with a shuffled observation does not pass B-P1 (−0.04° vs random) ✔.
Implementation detail fixed before the numbers: predict returns 6-dim chunks, padded with zeros to the model's action
dimension for the velocity query; the residual reads the 6 real dimensions only.

## Gate B — selection quality (offline, PyTorch, Orin at 25 W; `results/gate_b.json`)

Mean |Δ| in degrees to the recorded actions over the executed window (5 arm joints), 20 episodes:

| selector | d = 8 | d = 12 |
|---|---|---|
| single (naive async) | 2.47 | 2.66 |
| random of 4 | 2.54 | 2.76 |
| FlowRes, stale observation (Old-FlowRes) | 2.54 | 2.72 |
| **PACT** (FlowRes, pre-commit observation) | **2.27** | **2.47** |
| mean of 4 | 2.32 | 2.51 |
| oracle (ceiling) | 1.79 | 1.89 |

| bar | value (episode bootstrap, 95 % CI) | verdict |
|---|---|---|
| **B-P1** PACT < random at d = 8 | −0.27° (−0.44 to −0.09) | **pass** |
| **B-P2** PACT < Old-FlowRes at d = 8 | −0.27° (−0.48 to −0.09) | **pass** |

Old-FlowRes equals random (+0.00°): the fresher observation does the work, not the residual itself. Same picture at
d = 12 (PACT − random −0.29°).

## Gate A — the cost on the Orin (`results/gate_a.json`, `results/ver_ratio.json`)

Orin NX at **MAXN_SUPER + jetson_clocks**, nothing else on the GPU. T_gen(1) and the server split come from WS-057 A2
(same server flags as the 238 ms reference). vla.cpp has no batched sampling and no velocity query on a given x_τ, so
T_gen(4) is 4 sequential calls and T_ver is **projected**: the PyTorch ratio T(16 velocity rows) / T(one denoise
step) on the Orin (1,018.8 / 63.8 ms = 16.0) times vla.cpp's 5.3 ms per step.

| | ms |
|---|---|
| T_gen(1) | 237.5 |
| T_gen(4), sequential | 949.9 |
| T_prefix (vision + prefill) | 165.7 |
| T_ver, projected | 84.5 |
| **T_PACT** | **1,200.1** (overlap-optimistic: 1,034.5) |

| bar | value | verdict |
|---|---|---|
| **A** d_PACT ≤ d_naive (fail if ≥ d_naive + 2), ticks of 33.3 ms | d_PACT **37** (optimistic 32) vs d_naive **8** | **fail** |

## B-P3 (transfer): not computable as written

B-P3 compares PACT committed at k = d_PACT with single at k = 8. With d_PACT = 37 (projected), a 50-step chunk holds
only 13 of the 25 steps PACT would have to execute, so the protocol's executed window does not exist at that delay.
No other window was chosen after seeing this: B-P3 is reported as not computable, which in practice means PACT does
not transfer to this Orin setup without batched sampling.

## Verdict (the project's wording)

**Replicates offline, does not transfer to the Orin.** Selecting among four candidates with the pre-commit
observation beats a random pick and the stale-observation residual (B-P1, B-P2 pass); the paper's "no added
control-step delay" (A) fails by 29 ticks because vla.cpp samples the four candidates one after another. By the
decision rule this is a register entry, with no executor change. Gate C (arm) is not run.

## Compute

External GPU: no. Gate B on an Orin NX at 25 W (PyTorch); gate A on the Orin at MAXN_SUPER + `jetson_clocks`
(vla.cpp + a short PyTorch ratio). Cost: $0.

## What is not proven

T_ver is projected (vla.cpp has no velocity query on a given x_τ). A batched sampler would shrink T_gen(4) and could
change A and make B-P3 computable. Gate C (arm) is not run. One policy, one task family, offline episodes.
