# TOPReward vs elapsed time — protocol (WS-081 A4, folded into WS-077; frozen 2026-11-01 before any 8B score)

Owner, 2026-11-01 night: "TOPReward (WS-081 A4): frozen protocol, then go vast (≤ $5): Qwen3-VL-8B TOPReward on
ArmnetBench 2,047 + the 88 own, through LENGTH_PROTOCOL.md's analyses; fold the result into the WS-077 draft."
Only a 3-episode smoke test of the code with Qwen3-VL-2B on the Orin has been run (own episodes 0–2, all successes;
it checks that the pipeline returns finite log-probabilities; no AUROC was computed). Anything decided after the first
8B score is listed under Deviations and labelled exploratory.

## Question

Does a training-free video-language reward model, TOPReward (2602.19313; claims success-detection ROC-AUC 0.939 with
Qwen3-VL-8B on 279 LeRobot episodes, Table 3), separate failed from successful SO-101 rollouts better than episode
length, within policy, on the set where length gave 0.942 (WS-077)?

## Inputs (pinned)

- Model: `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, bf16, transformers 5.5.4 (the
  version of `lerobot-jetson:public`, where the code was tested with Qwen3-VL-2B @ `89644892`).
- ArmnetBench: `armnet/armnetbench_v01_lerobot_so101` @ `1db69c82b589c7172ee31a6b8f7ef58d1fc2c663`, view `top`, policy
  rollouts; the analysis set is LENGTH_PROTOCOL.md's: success_class ∈ {successful, failure}, teleop excluded
  (n = 2,047). Suboptimal episodes are scored and not analysed.
- Own rollouts: `ravediamond/so101-bowls-policy-rollouts` @ `bad1c8ea363789f5f55f9098e775f7361a0f019f`, view
  `front_top_camera`, labels `meta/rollout_labels.csv` (outcome success / failure): **69 episodes** (written here as 46 / 23 from older notes; the labels file at this revision says 44 / 25 — corrected 2026-11-01 before the analysis was run; the analysis reads the file). The other
  19 of G9a-own's 88 (M1 sessions) are on the old Mac only; they are not scored, and the own result is on 69.
- Code: `topreward_score.py` (this folder, committed with this file).

## Scoring (fixed; the paper does not state K, frames per prefix or the Yes/No wording, so these are our reading)

- Frames: the episode's segment of the camera video at 2 fps, last frame included, shortest edge 224 px; each query
  sees at most 16 frames (uniform, last kept; an odd count gets the first frame repeated for Qwen's temporal pairs).
- Prompt scored directly, no chat template (the paper's choice for open models):
  "<video> The above video shows a robot manipulation trajectory that completes the following task: {task}. Decide
  whether the above statement is True or not. The answer is:" → r+ = log p(" True"), r− = log p(" False").
- K = 10 uniformly spaced prefixes, first frame to last; margin m_k = r+_k − r−_k;
  s_top = mean(m over the last 3 prefixes) − mean(m over the first 3) (paper Eq. 7).
- Full-video Yes/No: "<video> Did the robot in the above video complete the following task: {task}? Answer Yes or No.
  The answer is:" → s_yn = log p(" Yes") − log p(" No") (Eq. 8).
- **Primary score:** s_succ = z(s_top) + z(s_yn), z over the analysed set of each dataset (Eq. 9). Higher = success.
- **Truncated score (50 % analysis):** the Yes/No margin on the clip cut at 0.5 × the task's longest episode (the cap
  of LENGTH_PROTOCOL.md (b), same T); s_cap = that margin (episodes shorter than the cap use the full clip).
- Instruction: the episode's task string, verbatim.

## Analyses (LENGTH_PROTOCOL.md's, same functions in `length_analysis.py`: AUROC = Mann–Whitney, 2,000 bootstrap
resamples stratified by policy, seed 0)

1. ArmnetBench, s_succ: pooled AUROC; within-policy pair-weighted AUROC; per policy; per task. All with 95 % CIs.
2. **Paired against length** (score −length), same episodes, same resamples: Δ_within = AUROC_within(s_succ) −
   AUROC_within(−length), and Δ_pooled.
3. 50 % truncation: within-policy AUROC of s_cap; paired Δ against length capped at 50 % (WS-077: 0.851).
4. Components reported separately (s_top alone, s_yn alone): descriptive.
5. Own 69: AUROC of s_succ with CI (bootstrap stratified by outcome); paired Δ against length; policy-identity
   baseline (per-policy success rate, an upper bound); non-TurboVLA subset descriptive.
6. Robometer-4B (G9a-AB pooled 0.818): reported beside, unpaired (its per-episode scores are not on the Orin).

## Decision rule (fixed)

On ArmnetBench, full episodes: **TOPReward beats elapsed time** if the 95 % CI of Δ_within lies above 0; **matches**
it if the CI contains 0; **is below** it if the CI lies below 0. The same three-way reading at the 50 % cut (s_cap vs
capped length). The own-69 result is reported and does not enter the decision (length is at chance there).

Episodes with a scoring error are excluded and counted; if more than 2 % of an analysed set errors, the errored
episodes are re-scored once before any analysis.

## Budget

`go vast` from the owner (2026-11-01, night cap $20 total, this item ≤ $5). One on-demand instance, ≥ 24 GB GPU,
hard deadline 3 h + 20 min grace with self-destroy, a local destroy timer, instance destroyed right after the verified
fetch. Every $ logged in the queue.

## Deviations

(none yet)
