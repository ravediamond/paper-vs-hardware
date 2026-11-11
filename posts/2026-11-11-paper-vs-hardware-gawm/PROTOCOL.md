# WS-168 · GAWM-S on the bowls data: PROTOCOL (draft for owner confirmation, 2026-11-06)

Status: **not frozen.** Every bar in §5 was proposed in chat on 2026-11-06 (Claude draft) and is not a bar until the owner
confirms it (`go WS-168`). No number of this study exists yet. Source: `journal/prompts/2026-11-06_paper_scan_lajepa_gawm_so101bench.md` Part B,
paper `~/workspace/papers/World models/2610.37091v1-gawm.pdf`, memo `journal/papers/scan-2026-11-06/MEMO.md` §2.

## 1. Question
Does the latent forecast of a compact world-action model (GAWM-S: frozen DINOv3 ViT-S/16, query adapter, predictor, ACT-style decoder)
pay off at small data on our arm, and does the gain grow when the training episodes shrink to a quarter?

Claims tested (paper numbers, memo §2): G1 capacity at our scale (Table I), G2 the gain of the world model grows as data shrinks
(Table II, p. 5: −9.5 pts [5.2, 14.2] at ¼ LIBERO data without the world model, +2.5 [−0.8, 6.1] at full data), G3 latency (Fig. 3, p. 6: 8.8 ms on an RTX PRO 5000).

## 2. Data
- Dataset `ravediamond/so101_bowls_all_trimmed` @ `4e5c6fda8dc5ce2252214bc086181d3df8b594b0`: 182 episodes, 61,462 frames, 30 fps, robot type `so_follower`
  (`data/FACTS.json` keys `so101_bowls_all_trimmed.revision`, `.episodes`, `.frames`, `.fps`, `.robot_type`). Two cameras (top, wrist), resized to 256².
- Split (fixed, same for every model): held-out = the 10 episodes with the highest index (172–181 of 0–181); train = the other 172.
- Quarter data: a fixed random ¼ of the 172 train episodes (seed 0); the episode list is committed as `projects/gawm/splits/quarter_seed0.json` before any training run.
- Seeds: three per model and split.
- Code: `https://github.com/gaoxiang12/ga-wm` @ `662e6d5c81194248156d139089716f3addcbc7ca` (2026-10-28); builds on StarVLA; a LeRobot v3 loader has to be written (Step 1).

## 3. Metric
- Held-out first-action L1: for every frame of the 10 held-out episodes the model predicts a chunk (H = 32) from the observation at that frame; L1 of the first action
  against the recorded action, in normalised action units (the paper's loss), averaged over the 6 action dimensions and over frames.
- Aggregation: per-episode mean, averaged over the three seeds, then the model difference per episode.
- Bootstrap: 2,000 resamples (template default; the prompt gives none), seed 0, unit of resampling = episode (10 episodes); paired difference GAWM-S − ablation.
- B2 compares two paired differences: gap at ¼ data minus gap at full data, with the same episode resample drawn for both.
- Latency (B3): median of 200 chunk queries, batch 1, FP16, MAXN_SUPER + jetson_clocks; encoder and the rest reported separately.
- Arm (B4): successes of 20 trials per policy, Wilson 95 % intervals.

## 4. Baselines and controls
- The paper's own ablation "w/o world model": same adapter, decoder and task code; forecast and all feature losses removed (Table II variant). Trained on the same splits and seeds.
- Identity null: first action = the recorded `observation.state` at that frame, same frames, same normalisation. Reported next to every L1.
- SmolVLA for the arm (B4) only: 9/10 on the arm, 238 ms per chunk in vla.cpp (EXPERIMENTS.md row 135).
- Same-set rule: every comparison uses the same 10 episodes and the same frames.

## 5. Bars (verbatim from the prompt; each is pass / fail / not proven)
- B1 full data (vast, cap $6): GAWM-S vs "w/o world model" (same adapter, decoder, task code; forecast and all feature losses removed — Table II's variant), both trained 20 k steps / batch 32
  on the bowls train split; held-out first-action L1 (normalised actions, the paper's loss) of GAWM-S ≤ the ablation's, episode-bootstrap CI of the paired difference; verdict "gain" if the CI excludes 0,
  "no gain" otherwise. Three seeds each. *(prompt Part B, Step 0)*
- B2 quarter data (same cap): both models on a fixed random ¼ of the train episodes (seed 0, list committed before training). Bar for the paper's claim: the GAWM-S − ablation gap at ¼ data is **larger** than
  at full data, paired CI excluding 0. This is the Shisso claim; B2 is the item's headline whatever B1 says. *(prompt Part B, Step 0)*
- B3 Orin latency ($0): full chunk query (DINOv3 ViT-S/16 on two 256² views, adapter, predictor, decoder, FP16, MAXN_SUPER + jetson_clocks, batch 1, median of 200) ≤ 60 ms (paper: 8.8 ms on an RTX PRO 5000;
  our ratio to a desktop GPU on SmolVLA is ~5–7×). Report encoder / rest separately. *(prompt Part B, Step 0)*
- B4 arm (go arm): GAWM-S full-data model, 20 trials ABAB against SmolVLA on the WS-163 layouts; successes ≥ SmolVLA − 2; Wilson CIs. Task code: one task → a learned constant vector
  (LiLa-WAM's visual transition token is not needed with one task). *(prompt Part B, Step 0)*

Sign convention for B1 and B2: L1 is an error, so "GAWM-S ≤ ablation" means a negative paired difference GAWM-S − ablation; "gap" in B2 is the ablation's L1 minus GAWM-S's L1 (positive = world model helps).

## 6. Checks before numbers
Two held-out episodes first: frame counts, action shape (6 dims, H = 32 chunk), the future-frame mask past the episode end (t+16, t+32); one hand-computed L1 on 5 frames; identity-null L1 on the same frames;
parameter counts of both configs (expect about 21.6 M frozen and 50–60 M trainable for GAWM-S; the exact count is logged, not a bar).

## 7. Deviations (empty) and open items
Deviations: none.
- O1: the "gap" sign convention above and averaging over the three seeds before the paired bootstrap are my wording of the prompt's bars, not new thresholds; owner to confirm.
- O2: B1 and B2 share a 12-run, $6 cap; the prompt's estimate is about 5 h and $2.5 on a 4090. The step count (20 k) and batch (32) are in B1; the learning-rate settings of Step 1 are not bars.
- O3: B3's 60 ms is a proposed ratio argument (about 5–7× the desktop GPU), not a measured SmolVLA ratio for this model.

## 8. Exploratory
Anything chosen after data exists (other encoder sizes, the forecast horizons other than (16, 32), the loss weights beyond the Step 1 defaults) is labelled exploratory.
