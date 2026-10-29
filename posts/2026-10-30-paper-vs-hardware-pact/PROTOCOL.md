# PACT-VLA (2610.28019) on SmolVLA + Orin: protocol (frozen 2026-10-29, before any number)

Paper: Zhu et al., *PACT-VLA: Speculative Action-Chunk Verification with Pre-Commit Observations*, arXiv
2610.28019v1 (20 Oct 2026). HKUST (Guangzhou), One Robotics. No code. Analysis:
`journal/queue/2026-10-29_three_papers.md` §3. Draft: `journal/drafts/2610.28019.md`.

## Question

Under async execution with this rig's delay, does choosing among N = 4 SmolVLA candidate chunks with FlowRes under the
pre-commit observation pick chunks that fit what happened next better than a random pick and than FlowRes under the
stale observation, and does that survive once the Orin pays for the extra observation pass?

## Decision rule

- B-P1, B-P2 and B-P3 pass, and C passes → PACT becomes an arm of the async executor (`run_g13a_live.py`), default off.
- B-P1 or B-P2 fail → the selector does not work on this policy; register entry, no executor change.
- B-P1 and B-P2 pass, B-P3 fails → "works, does not transfer to an edge GPU": register entry, no executor change.

## Method under test (paper §3.2, Eq. 5–6, Table 6, App. E)

- Candidates: N = 4 chunks from o_t by plain policy sampling (independent noise, seeds 4b … 4b + 3 for boundary b),
  10 Euler steps, H = 50.
- FlowRes: one verification noise ε_ver per boundary (seed 10 000 + b), shared across candidates and flow times.
  For each candidate A (model-normalised action space) and τ ∈ {0.2, 0.4, 0.6, 0.8}:
  x_τ = (1 − τ)A + τ ε_ver, u = ε_ver − A, v̂ = SmolVLA velocity at (x_τ, τ) conditioned on the observation used for
  scoring; r_τ = Σ_j w_j ‖m ⊙ (v̂_j − u_j)‖². E = Σ_τ r_τ. Choose argmin E.
  w_j = 1 for the executed positions j ∈ [d, d + s), 0 elsewhere; m keeps all 6 dimensions (5 joints + gripper),
  as their real robot kept all 14. SmolVLA's flow convention (`x_t = t·noise + (1 − t)·actions`, `v = noise − actions`,
  t from 1 to 0) equals Eq. 2 with τ = t.
- Scoring observation: PACT = o_{t+d−1} (pre-commit); Old-FlowRes = o_t.

## Data and geometry (the G13b harness)

`ravediamond/so101_bowls_all`, the 20 G13b episodes (seed 0 list), policy `ravediamond/smolvla_2026-09-24_14-34-40`
@ `c1d76f6`, PyTorch LeRobot in `lerobot-jetson:rc` (`LEROBOT_SRC=image`), s = 25, boundaries every s frames as in G13b
(254 per delay). A boundary is used only if t + d + s ≤ episode length.

## Gate A — the cost on the Orin (offline, $0)

Measure, MAXN_SUPER, nothing else on the GPU:
1. T_gen(N): N-candidate generation in vla.cpp for N = 1 and N = 4. Take it from WS-049 gate 1 (AAC) if that gate
   measured N = 1 and N = 4 or a superset under the same server flags; else measure (50 calls, 5 warm-ups).
2. T_prefix: vision + prefill for one observation in vla.cpp (server split, as G-lat-2).
3. T_ver: one batched velocity query of 4 candidates × 4 flow times. If vla.cpp cannot run a velocity query on given
   x_τ, measure the ratio T_ver / T_step(one denoise step, batch 1) in PyTorch on the Orin and multiply by the vla.cpp
   per-step denoise time; label the result "projected".
Cycle (no overlap: the Orin GPU runs at 94–98 % during a chunk, so overlap is not assumed):
T_PACT = T_gen(4) + T_prefix + T_ver. d_naive = ceil(T_gen(1) / 33.33 ms), d_PACT = ceil(T_PACT / 33.33 ms).
Also report the overlap-optimistic figure max(T_gen(4), 33.33 + T_prefix) + T_ver, labelled optimistic.

**Bar A ("no added control-step delay", the paper's claim):** d_PACT ≤ d_naive. FAIL if d_PACT ≥ d_naive + 2.
Between: not proven. A is a measurement of the claim; it does not stop gate B.

## Gate B — selection quality on recorded episodes (offline, PyTorch, $0)

Per boundary b at frame t, candidates from o_t; each selector's chunk is compared with the recorded actions over its
own executed window. Error = mean |Δ| in degrees over the 5 arm joints between chunk[k : k + s] and recorded
a[t + k : t + k + s], where k is the selector's commit offset.

Selectors at the nominal delay d = 8 (k = 8; scoring frame t + 7 for PACT):
1. single = candidate 0 (naive async);
2. random = mean of the 4 candidates' errors (the exact expectation of a uniform pick);
3. Old-FlowRes = argmin E under o_t;
4. PACT = argmin E under o_{t+7};
5. mean-of-N = element-wise mean chunk;
6. oracle = min of the 4 errors (ceiling, not a method).
Same at d = 12 (k = 12, scoring frame t + 11), for the delay trend.
Orin-realistic PACT: k = d_PACT from gate A (scoring frame t + d_PACT − 1), compared with single at k = 8.
If gate A ends "projected", use the projected d_PACT and say so.

Per-episode mean error (20 episodes); paired differences; episode bootstrap 2,000 resamples, seed 0, 95 % CI.

**B-P1:** PACT < random at d = 8, CI of (PACT − random) entirely < 0.
**B-P2:** PACT < Old-FlowRes at d = 8, CI of (PACT − Old) entirely < 0.
**B-P3 (transfer):** Orin-realistic PACT < single at k = 8, CI entirely < 0.
FAIL for a clause: its CI entirely ≥ 0. Otherwise that clause is not proven. Verdict: replicates (P1 and P2),
transfers (P3). Mean-of-N and oracle are reported, no bar.

## Measurement checks

- Candidate 0 of PACT's pool must equal a fresh single sample with the same seed (bit-identical) — else stop.
- FlowRes sanity: scoring the recorded future actions (normalised) as a fifth "candidate" under o_{t+d−1} must get a
  lower E than the median candidate on ≥ 60 % of boundaries; if not, the velocity query is wired wrong and no
  selection number is read.
- Smoke: FlowRes with a shuffled scoring observation (o from another episode) must not pass B-P1.

## Gate C — on the arm (needs `go arm`; only after B-P1, B-P2 and B-P3 pass)

Async, s = 25, naive vs PACT, 20 + 20 trials, the first 20 rows of the §29 sealed `LAYOUTS.csv`, arm order alternating
per layout, owner labels at rest blind to arm. **Bar C:** successes(PACT) ≥ successes(naive) + 3; if naive ≥ 18/20 the
verdict is "not proven: ceiling". Gate C is not run otherwise; the README says why.

## Notes

The recorded future is one teleoperated continuation; fitting it is a proxy for "fits the scene". The episodes are in
the policy's training data. The paper's own single-decision fixed-pool gain over random was +0.44 points (App. D.2),
so a null B-P1 at n = 20 episodes is a likely outcome and is reported as such. Anything decided after this commit is
exploratory.

## Cost

$0. Gate A ≈ 30 min Orin (less if WS-049's numbers cover T_gen). Gate B: 254 boundaries × 2 delays × (4 candidates +
3 observation prefixes + 2 × 16 velocity queries) in PyTorch at ~1 s per chunk ≈ 3–4 h Orin, resumable per episode.

## Outputs

`projects/pact-vla/README.md`, `projects/pact-vla/results/` (`a_latency.json`, `b_rows.jsonl` per boundary,
`b_summary.json`), EXPERIMENTS.md §2 rows, draft sections 5–8.
