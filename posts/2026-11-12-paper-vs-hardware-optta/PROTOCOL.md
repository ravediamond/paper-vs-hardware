# WS-196 protocol: OpTTA on the Orin, then prediction-only TTA of W1 (frozen before any number)

Status: bars H1–H4 confirmed by the owner 2026-11-07 (GO.txt `go WS-196`); deviations of §7 added 2026-11-07 before any H1/H2/H3 number. Bars are
copied from `journal/prompts/2026-11-07_paper_scan_part2_jepa.md` Part H; none was added or changed. Paper: OpTTA,
arXiv 2610.35636 (`journal/papers/scan-2026-11-06/MEMO_part2.md` §8). Cost: $0, no vast, no arm.

**What this tests and what it does not.** OpTTA's optimistic term needs a planner, and W1 has none that works (G1b
failed). Parts H1 and H2 re-run the paper's own LeWM PushT setting on the Orin. Part H3 transfers only the
prediction-only half (one Adam step on the last predictor block) to W1 and judges it on the surprise monitor. H3 and H4
**do not test optimism** and no result of them is worded as if they did.

## 1. Question

(H1, H2) What does a replan cost on an Orin NX for OpTTA's LeWM PushT setting, and does OpTTA beat prediction-only
AdaJEPA by 5 points under one darkening severity? (H3, H4) Does adapting W1's last predictor block along an episode cut
the surprise monitor's false alarms under a synthetic lighting shift without losing its recall, and does one step fit in
the idle time of a 238 ms chunk?

## 2. Data

- H1, H2: the paper's public LeWM PushT checkpoint and code (builds on the AdaJEPA and LeWM codebases, paper App. A.1),
  set up in a venv `~/venvs/optta/`. The repository URL, commit and licence are not in our memo ("Code" badge, p. 1):
  they are recorded in the first log line of the run, before any number. Seeds 0, 1, 2; 50 start–goal episodes per
  seed; K_max 30; darkening severity 0.85 (their Fig. 4 setting); CEM 200 candidates × 10 iterations, horizon 5, FP32.
- H3, H4: W1 `ravediamond/lewm-w2-so101-bowls`, `w1/weights_final.pt`, revision
  `d0336db6e13d7477d72c33261e0892969fc035c0` (`data/FACTS.json` `/w1_checkpoint`), strict load under transformers 5.19.0.
  Held-out set: the 25 held-out episodes of the G6a `split.json` (`ravediamond/so101_bowls_all` @
  `a400d2e748e8b1ba557d0de03f7d4ec03255a9c6`, revision `4e5c6fda8dc5` split). Rollouts for recall: 69 labelled episodes
  of `ravediamond/so101-bowls-policy-rollouts` @ `bad1c8ea363789f5f55f9098e775f7361a0f019f` (44 success, 25 failure;
  `data/FACTS.json` `/rollouts/labels`), failure onsets `shared/gate/data/2026-10-23_onsets_vlm/onsets.csv`.
- Shift: synthetic darkening, gamma 1.5 and brightness × 0.7 (the paper's two corruption families), applied to the camera
  frames before encoding; no curtains-open recordings exist.
- Monitor: the W1-A L2 surprise scorer (`shared/gate/data/2026-10-19_w1_screening/`, README row W1-A: anchor every 25
  frames, context of anchor and two previous frames with recorded actions, imagined to k = 25). Its threshold τ is the
  one fitted on the unshifted data and is not refitted under the shift.

## 3. Metric

- H1: milliseconds per replan, median of 100 replans, for Frozen / AdaJEPA / OpTTA (all 200 candidates) / OpTTA-30 (30
  elites); ratio to the paper's 118 / 141 / 163 / 165 ms (Table 2).
- H2: final success rate (%), per seed and mean; the OpTTA − AdaJEPA difference with a seed-paired bootstrap 95 % CI
  (2,000 resamples, seed 0).
- H3: the monitor's false-alarm rate on the corrupted successes, and recall on the failures' onsets under the same
  corruption, each for frozen W1 and adapted W1 (adaptation: Adam, lr 5e-5, buffer 5, one step per chunk, along each
  episode, last predictor block only); episode-level bootstrap CIs, 2,000 resamples, seed 0.
- H4: wall time of one adaptation step on W1's last block, FP16 or FP32 as run, median of 100 steps, on the Orin.

## 4. Baselines and nulls

- Frozen model (no adaptation) and AdaJEPA are the baselines of H2; frozen W1 is the baseline of H3.
- Pessimism control (H2): ρ = −0.1, one seed; must sit below AdaJEPA or the sign of the effect is not the paper's.
- Shuffled null (H3): adaptation on **shuffled transitions** must not reduce false alarms by more than 10 %; if it does, the
  gain is forgetting, not adaptation.
- Same episodes and same corruption for every row of a comparison.

## 5. Bars (verbatim from the prompt; each ends with pass / fail / not proven)

- **H1 Orin cost (O2).** Measurement, no bar: ms per replan for the four methods, ratio to the paper's numbers. If the
  OpTTA update alone exceeds 1 s on the Orin, the result says "not real-time on this device". (prompt Part H)
- **H2 one gain point (O1).** OpTTA − AdaJEPA **≥ +5 pp** with the seed-paired CI excluding 0 (paper: about 2.5 × the
  AdaJEPA gain on PushT); the pessimism control must be below AdaJEPA. (prompt Part H)
- **H3 W1 prediction-only TTA under lighting shift.** The surprise monitor's false-alarm rate on the corrupted successes
  drops from the frozen value by **≥ 50 %**, while recall on the 25 failures' onsets (69 rollouts, same corruption) stays
  **≥ 0.8 of the frozen recall**; the shuffled null must not reduce false alarms by more than **10 %**. (prompt Part H)
- **H4 Orin cost of H3.** One adaptation step on W1's last block **≤ 20 ms** (fits the idle time of a 238 ms chunk).
  (prompt Part H)

Result wording (prompt): "OpTTA on Orin NX: <ms> per replan (×<r> vs RTX PRO 6000); PushT darkening 0.85: Frozen <a> /
AdaJEPA <b> / OpTTA <c> %; W1 prediction-only TTA: false alarms <f0>→<f1>, recall <r0>→<r1>".

## 6. Checks before numbers

1. H1/H2: their code reproduces the frozen PushT success they report at severity 0 on 5 episodes (sanity, not a bar), and
   a replan runs end to end for all four methods on one episode.
2. H3: frozen W1 with the unshifted data reproduces the W1-A numbers (L2 surprise AUROC 0.86 on its 50 episodes) from the
   same scorer before any corruption is applied.
3. A hand-computed false-alarm rate on one corrupted episode (anchors listed by hand) equals the code's value.
4. Everything runs under `scripts/memcap.sh` (11 GB cap, no swap); a CPU fallback for PushT is allowed if the GPU build
   fails, and H1 then says "CPU".

## 7. Deviations (no bar changed)

1. **Our own implementation (owner 2026-11-07, "use our own").** No OpTTA repository or checkpoint exists. H1/H2 test
   **our** AdaJEPA and OpTTA, written from the paper's Alg. 1, Eq. 6–8 and App. A.1 Table 3 (`projects/optta/optta_pusht.py`),
   not the authors' code. Public parts used: the LeWM model code (`lucas-maes/le-wm` @ 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac,
   MIT), the PushT environment (`galilai-group/stable-worldmodel` @ 21446f1ede6d5284e981bd7b47f432b994e6d812, MIT) and the
   official checkpoint `quentinll/lewm-pusht` @ 22b330c28c27ead4bfd1888615af1340e3fe9052. The CEM planner is our re-write of
   the stable-worldmodel CEM solver (same elites update, first candidate = mean, warm start from the unexecuted plan).
2. **Start–goal pairs.** The dataset `pusht_expert_train.h5.zst` is 13.1 GB compressed and 46.3 GB raw; the Orin has under
   10 GB free. Only the first 15,987 rows (139 episodes) are readable from a 60 MB prefix; start–goal pairs (goal = the
   state 25 steps after the start, as in the LeWM eval config) are drawn from those 139 episodes, 50 per seed, seeds 0, 1, 2
   (the same pairs for every method of a seed). The action scaler (mean, std) is fitted on the same rows.
3. **Darkening 0.85** is not defined in the paper text: pixels × (1 − 0.85) on every observation and on the goal image,
   before ImageNet normalisation.
4. **Adapted parameters** (predictor-only scope): last predictor block, final norm and `pred_proj` ("output projection",
   BatchNorm in eval mode); Adam re-initialised each step, lr 5e-5, ρ = 1 (−0.1 for the pessimism control), ε = 1e-12, ramp
   √((k+1)/30) (k = 0 would zero the term). L_plan uses the 200 final-iteration CEM candidates from the planning state;
   OpTTA-30 uses the 30 lowest-cost of them. Success: the environment's `terminated` flag (block within 20 px and π/9) at
   any of the 5 executed steps; an episode stops on success; final success = success within 30 replans.
5. **Power profile.** The Orin stays in its quiet 25 W profile (no sudo); H1 milliseconds are for that profile, FP32, TF32 off.
   H1's "ms per replan" is CEM plus the adaptation step, CUDA-synchronised, over the first 100 replans of the H2 runs
   (an OpTTA-30 timing run of 5 episodes supplies its replans); the encode of the next frame and the environment step are excluded.

## 8. Open items and exploratory

Three points need the owner's wording. The OpTTA repository URL and licence are not recorded in our memo and are
confirmed at run time. H3 says "corrupted successes" for the false alarms and "69 rollouts" for the recall: which
successes carry the false-alarm rate (the held-out demos, or the 44 rollout successes) is not stated, and the proposal is
the 44 rollout successes, the same set as the recall, so that both come from one dataset. The wall time of H2 (9 runs ×
50 episodes, hours on the Orin) is logged, and the item is detached with `nohup`.

Exploratory, and labelled so in any result: other severities or corruptions, other buffer sizes, and any analysis after
the first number.
