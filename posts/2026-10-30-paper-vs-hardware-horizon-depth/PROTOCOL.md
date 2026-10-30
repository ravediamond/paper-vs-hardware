# Horizon × depth (2610.32102) on SmolVLA + Orin: protocol (frozen 2026-10-29, before any number)

Paper: Yang et al., *Coupling Execution Horizon and Sampling Depth in Flow-Matching VLA Policies*, arXiv 2610.32102v1
(23 Oct 2026). UCAS / CASIA NLPR, NUS, SJTU, FiveAges. Project page https://jiabingyang01.github.io/Horizon-Depth-VLA/,
no code. Analysis: `journal/queue/2026-10-29_three_papers.md` §1. Draft: `journal/drafts/2610.32102.md`.

## Question

On this rig (SmolVLA, H = 50, sync s = 50, vla.cpp on a Jetson Orin NX), does cutting the Euler steps from 10 to 2
(a) contract the action cloud around an unchanged centroid as the paper's Proposition 1 says, (b) save enough latency
to matter, and (c) keep task success, so that N = 2 can become the Orin default?

## Decision rule

- A1 pass and A2 pass and B1 pass → `num_steps` = 2 becomes the SmolVLA default on the Orin (vla.cpp and the
  PyTorch fallback), written into the deploy config and DECISIONS.md.
- A1 or A2 fail → N stays 10; no arm time is spent (B is not run).
- B1 fail → N stays 10, the result is published as "does not transfer".

## Fixed setup

- Policy `ravediamond/smolvla_2026-09-24_14-34-40` @ `c1d76f6` (step 020000, H = 50, the checkpoint behind every
  vla.cpp number). Its GGUF for vla.cpp: the one the 238 ms figure was measured with.
- Frames for A1 / A3: `ravediamond/so101_bowls_all`, the 20 episodes of G13b (seed 0 list,
  `scripts/run_g13b_rtc_offline.py`), one frame per episode at the episode's 40 % time point (top + wrist + state,
  the task string of the episode). 20 frames.
- Euler schedule: the policy's own (uniform, t from 1 to 0, Δt = −1/N). No other solver.
- Device for timing: Orin NX 16 GB, `MAXN_SUPER` + `jetson_clocks`, vla-server `--flash-attn 1 --mm-prec default`
  on the branch the 238 ms was measured with (`805570d` or its successor; record the commit). Nothing else on the GPU
  (check before each timed block; WS-049 must not be running).

## Gate A — offline, $0

### A1 — mechanism (PyTorch LeRobot SmolVLA, Orin or CPU)

For each of the 20 frames, draw K = 32 noise tensors (seeds 0–31, shared across N), integrate with
N ∈ {1, 2, 3, 5, 10}, unnormalise to degrees (5 arm joints + gripper). Per frame and N:
- centroid ā_N = mean over K; radius R(N) = mean over K of ‖a_k − ā_N‖₂ (over the 50 × 6 chunk);
- centroid shift δμ(N) = ‖ā_N − ā_10‖₂ / mean_k ‖a_k(10)‖₂ (relative to the nominal action magnitude, as the paper);
- linear-fit residual: fraction of the variance of {a_k(N)} unexplained by the best scalar map from the paired
  {a_k(10)} (paper §III-D).
Report medians over the 20 frames, with episode-bootstrap 95 % CIs (2,000 resamples, seed 0).

**Bar A1 (mechanism reproduces):** median R(2)/R(10) ≤ 0.70 AND median δμ(2) ≤ 0.05 AND median max-|Δ| between ā_2
and ā_10 over the chunk ≤ 3.0° (five arm joints). Paper: 0.51, 3.2 %. FAIL if R(2)/R(10) > 0.85 or δμ(2) > 0.10.
Between: not proven.

### A2 — latency on the Orin (vla.cpp)

For N ∈ {1, 2, 3, 5, 10}: 50 timed chunk calls after 5 warm-ups, the real client, fixed frames (the 20 above,
cycled). Record client wall-clock per call and the server split (vision / prefill / denoise, the fields G-lat-2
logged; split prefill from denoise if the server can time them separately, else derive it: per-step denoise =
(server total at N = 10 − at N = 1) / 9, prefill = server total at N = 1 − vision − one step). Peak memory. If vla.cpp exposes no step-count setting, add one (a server flag or request field that sets
the Euler loop count; default 10, path bit-identical at 10) on a branch of the vla.cpp fork and record the commit.

**Bar A2:** median client latency at N = 2 ≤ 0.85 × median at N = 10 measured in the same session (≥ 15 % saving)
AND the N = 10 median within ±5 % of 238 ms (else the setup differs from the reference and the gate is void).
FAIL if the saving is < 8 %. Between: not proven.

### A3 — open-loop error vs the demonstration (descriptive, no bar)

On the same 20 episodes at every s-step boundary from t = 0 (s ∈ {10, 25, 50}), one noise seed per boundary shared
across N, N ∈ {2, 10}: median over boundaries of the mean |Δ| (degrees, 5 joints) between the first s predicted
actions and the recorded actions. Reported as the offline analogue of Fig. 1; it decides nothing (the episodes are
in the policy's training data).

### A4 — thermal (descriptive with a reading rule)

10 min continuous chunk loop (one call every 1.67 s, the s = 50 cadence) at N = 10, then N = 2 after 10 min idle;
in `MAXN_SUPER` and in 25 W. `tegrastats` at 1 Hz: GPU clock, GPU temperature, power. Throttling = GPU clock below
95 % of its first-minute median for ≥ 10 consecutive seconds. Reading rule: no throttling at N = 10 → the paper's
thermal claim is "not applicable here"; throttling at N = 10 and none at N = 2 → "replicates"; throttling in both
→ "does not replicate".

## Gate B — on the arm (needs `go arm`; only after A1 and A2 pass)

- **Task** "Put the blue bowl in the pink bowl.", SmolVLA via vla.cpp, sync, s = 50, 30 Hz, 25 s per trial, the
  live runner `scripts/run_g13a_live.py` (mode `sync`, `--exec 50`), `--record-dir` on.
- **Layouts**: the §29 sealed list `shared/gate/data/2026-10-26_s29_robot_day/LAYOUTS.csv` (sha256 `49e4d364…`),
  first 20 rows in file order; each layout is run once per arm, arm order per layout alternating
  (layout 1: N = 10 then N = 2; layout 2: N = 2 then N = 10; …). Placement guidance with
  `scripts/run_s29_session.py`'s placement tool, as on the §29 robot day.
- **Arms**: N = 10 (reference) and N = 2. 20 trials each, 40 total.
- **Outcome**: success = blue bowl inside the pink bowl at rest at the end of the trial, judged by the owner at rest
  (AGENTS.md §1), blind to the arm (the runner prints the arm only after the label is entered). Secondary, recorded,
  no bar: time to success, G14 oscillation flag (REKI criterion, `scripts/run_g14_oscillation_audit.py` rule), median
  boundary jump.

**Bar B1 (non-inferiority, the decision):** successes(N = 2) ≥ successes(N = 10) − 1 (out of 20). FAIL if
successes(N = 2) ≤ successes(N = 10) − 3. Between: not proven.
**Bar B2 (the paper's "better" claim):** successes(N = 2) ≥ successes(N = 10) + 3 → "replicates"; if N = 10 scores
≥ 18/20 the B2 verdict is "not proven: ceiling", stated as such.

## Statistics and honesty notes

n = 20 per arm: B1 is a screen, a 1-trial margin is not a power calculation. A3 uses training episodes.
Anything decided after this commit is exploratory. The paper's 5.6× is per-step amortised latency and mostly comes
from s; this rig already runs s = 50, so only the N effect is tested.

## Cost

$0. Gate A ≈ 2 h Orin (A1 ≈ 40 min PyTorch, A2 ≈ 15 min, A4 ≈ 45 min). Gate B: one robot session (~40 trials, ~1 h
plus resets), owner present.

## Outputs

`projects/horizon-depth/README.md` (results), `projects/horizon-depth/results/` (`a1_cloud.json`, `a2_latency.json`,
`a3_openloop.json`, `a4_thermal.csv`, `b_trials.csv`), EXPERIMENTS.md §2 rows (one per gate), draft sections 5–8.
