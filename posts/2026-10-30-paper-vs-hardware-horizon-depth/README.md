# Paper vs hardware 21: Horizon × depth (two Euler steps contract SmolVLA's actions by 22 %, not 49 %; vision caps the saving: not proven)

Posts:
- https://shissorobotics.com/blog/2026/10/30/paper-vs-hardware-horizon-depth/

## Claim

Yang et al. (arXiv 2610.32102): with long open-loop commitment, cutting a flow-matching VLA's Euler steps from 10 to 2
contracts the action cloud around an unchanged centroid (R(2)/R(10) 0.51 on π0), saves most of the inference time and
removes edge-GPU thermal throttling. On SmolVLA and the Orin NX: R(2)/R(10) **0.78** (between the bar 0.70 and the
fail line 0.85), N = 2 saves **17 %** of a chunk (196.1 vs 237.5 ms; vision is 151 ms of it), and the Orin does not
throttle at N = 10 in either power mode.

**Verdict: not proven.**

Protocol `PROTOCOL.md` (frozen 2026-10-29, before any number). Scripts `scripts/hd_cloud.py` (A1), `hd_latency.py`
(A2), `hd_thermal.py` (A4), committed before their numbers (`f010075`). Policy `smolvla_2026-09-24_14-34-40` @
`c1d76f6`; the 20 G13b episodes of `so101_bowls_all`, one frame each at 40 % of the episode.

## Gate A, by the frozen bars

| bar | value | verdict |
|---|---|---|
| **A1** median R(2)/R(10) ≤ 0.70 and δμ(2) ≤ 0.05 and max centroid \|Δ\| ≤ 3.0° (fail if R > 0.85) | R(2)/R(10) **0.78** (CI 0.73–0.81); δμ(2) 0.018 ✔; max \|Δ\| 2.35° ✔ | **not proven** |
| **A2** median latency N = 2 ≤ 0.85 × N = 10, and N = 10 within ±5 % of 238 ms (fail if saving < 8 %) | **0.83** (196.1 vs 237.5 ms); N = 10 −0.2 % vs 238 ms | **pass** |

A1 (PyTorch, Orin at 25 W, 20 frames × 32 draws × N ∈ {1, 2, 3, 5, 10}; `results/a1_cloud.json`): two Euler steps
do contract the action cloud around a nearly unchanged centroid, as the paper's Proposition 1 says, but by 22 % here
against the paper's 49 %; the CI lies between the bar and the fail line. Measurement check: the checkpoint runs 10
steps, and draw 0 at N = 10 equals the normal path (max |Δ| 0.0°).

A2 (vla.cpp, Orin NX at **MAXN_SUPER + jetson_clocks**, nothing else on the GPU, 50 timed calls after 5 warm-ups per N;
`results/a2_latency.json`):

| N Euler steps | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| median client round trip | 190.2 ms | 196.1 ms | 200.6 ms | 211.1 ms | 237.5 ms |

Split at N = 10: vision 150.7 ms, prefill 14.9 ms, **5.3 ms per denoise step**. Vision dominates the chunk on the
Orin, so cutting 8 of 10 denoise steps saves 41 ms (17 %), not 80 %. (A first run with the same medians, ±0.6 ms,
could not write its file to a results folder an earlier container run had left root-owned; it was re-run after the
ownership was fixed. Its log is kept.)

A3 (descriptive, training episodes; `results/a3_openloop.json`): mean |Δ| to the demonstration, N = 2 vs N = 10:
s = 10 1.01° vs 1.03° (668 boundaries), s = 25 1.34° vs 1.39° (274), s = 50 1.53° vs 1.67° (142). N = 2 is slightly
closer at every s.

A4 (thermal, descriptive with a reading rule: 10 min at N = 10, 10 min idle, 10 min at N = 2, one chunk every 1.67 s,
tegrastats at 1 Hz; throttling = GPU clock below 95 % of its first-minute median for ≥ 10 s):

| power mode | N = 10 | N = 2 | GPU temp max | reading |
|---|---|---|---|---|
| MAXN_SUPER + jetson_clocks (`results/a4_thermal_maxn.csv`) | no throttling (clock 1,173 MHz throughout) | no throttling | 68.6 °C / 67.9 °C | **not applicable here** |
| 25 W, quiet fan profile (`results/a4_thermal_25w.csv`) | no throttling | no throttling | 63.2 °C / 62.7 °C | **not applicable here** |

At this rig's s = 50 cadence the Orin does not throttle at N = 10, so the paper's "N = 2 avoids edge thermal
throttling" has nothing to act on here. Measurement note: this JetPack's tegrastats prints no GPU clock, so the
first MAXN run recorded none and could not be read; the script now reads the clock from devfreq (`2d2f2f9`) and the
run was repeated. The first run is kept in `archive/` (its temperatures: peak 67.8 °C).

Post-hoc note (not in the protocol): at 25 W the GPU governor keeps the clock at its floor (306 MHz median, 408 MHz
max), so the first-minute median is the floor and the rule cannot see a drop; the reading there rests on the rule as
written, and the temperature (63.2 °C max) agrees with it.

## Decision

The arm gate (B) runs only after A1 and A2 pass. A1 is not proven, so B is not run and SmolVLA stays at 10 Euler
steps on the Orin. Verdict: **not proven** — the contraction is real but half the paper's size, the latency gain on the Orin is capped
by the vision encoder (151 of 237 ms), and the thermal claim has nothing to act on here (no throttling at N = 10 in
either power mode).

## Compute

External GPU: no. A1 and A3 on the Orin NX at 25 W (PyTorch); A2 and the MAXN half of A4 at MAXN_SUPER +
`jetson_clocks` (vla.cpp); the 25 W half of A4 at 25 W with the quiet fan profile. Cost: $0. Retries: A2 once (its
JSON could not be written to a root-owned folder), A4 at MAXN once (tegrastats prints no GPU clock on this JetPack).

## What is not proven

The arm gate (success at N = 2 vs 10 over 20 sealed layouts) is not run because A1 did not pass. One policy, one
task family; A3 uses training episodes.
