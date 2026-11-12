# PROTOCOL — A Platonic test on a robot: top vs wrist latents up to a rotation, predictor off vs on (WS-107)

- Date: 2026-11-10 (draft 2026-11-01). **Frozen before any number** (no training, no embedding, no alignment value exists for this
  question on this rig at the commit of this file).
- Sources: prompt `journal/prompts/2026-11-01_paper_jepa-platonic-cameras.md` § Design detail (bars, ladder, metric);
  memo `journal/papers/jepa-platonic-cameras/FEASIBILITY.md` §3, §5; owner decisions 2026-11-01 afternoon (queue log
  WS-107 14:12); paper 2610.39302v1 (`~/workspace/papers/`, text `runs/WS-107/paper_raw.txt`).
- Owner answers O1-O9: `AMENDMENT_2026-11-01.md` (106ae73), applied in §3-§6 and listed in §9. The two choices the
  amendment left open (B3 wrist denominator, seed pairing) were delegated by the owner on 2026-11-10 ("I take your
  recommendation") to the memo's proposals, in their conservative form: §4 and §6.5, §9 D1-D2.
- Stop point of WS-107: this file. Training, code and analysis are separate WS items (§9).

## 1. Question
On the SO-101 bowls data, do encoder-only JEPAs trained separately on the top and the wrist camera embed the same
frames into latents that agree (mutual 10-NN) better than a bad-view control, and does switching on W1's
action-conditioned predictor lower that agreement?

## 2. Data
- Dataset and split: `projects/vla-jepa-world-model/wm_train/split.json` — `ravediamond/so101_bowls_all_trimmed` @
  `4e5c6fda8dc5ce2252214bc086181d3df8b594b0`, rule "whole grid layouts until >= 10% of episodes", seed 1234,
  157 train / 25 val episodes, 5 val layouts. (The untrimmed parent is `data/FACTS.json` key `so101_bowls_all`,
  182 episodes.) Preprocessing: `prepare_data.py` transcode (H.264 224×224), as W1 (`wm_train/README.md` § For H3).
- Training uses the 157 train episodes only. **Evaluation set N** = every frame of the 25 val episodes at 5 fps
  (frameskip 6 from 30 fps, `config/lewm_so101.yaml`), each frame a (top_t, wrist_t) pair at one timestamp. N is
  counted at the first check (§6), not assumed (the prompt's "≈ 2,000" is an estimate).
- Second test set (8-page only, exploratory): the 69 published rollouts (`data/FACTS.json` `rollouts.labels.episodes`).
- Seeds (O9): every model is trained with seeds 3072 and 3073 (`config/lewm_so101.yaml` base seed 3072); E_crop1 and
  E_crop2 take 3072 and 3073 respectively. Bootstrap seed 0.

## 3. Models (W1's encoder architecture and recipe: ViT-Tiny, patch 14, embed 192, `config/lewm_so101.yaml`,
100 epochs; single view; same train frames for all)
| id | view | positives | predictor | ladder |
|---|---|---|---|---|
| E_top, E_top' | top | (top_t, top_t+1) at 5 fps; seeds 3072, 3073 | off (encoder-only, SIGReg) | 2, 3 |
| E_wrist, E_wrist' | wrist | (wrist_t, wrist_t+1); seeds 3072, 3073 | off | 3 |
| P_top, P_top' | top | as E_top; seeds 3072, 3073 | on: W1's action-conditioned predictor (history 3, num_preds 1, 36-dim macro-action) | 4 |
| P_wrist, P_wrist' | wrist | as E_wrist; seeds 3072, 3073 | on, as P_top | 4 |
| E_crop1, E_crop2 | top | each a bad-view JEPA, two-crop positives of one top frame (O1 b, O2: DINO-style crops, scale 0.4-1.0, since W1 has no augmentation term); seeds 3072, 3073; scored on full top frames | off | 1 |
| F_dino | top, wrist | none (frozen DINOv2-S from the Hub) | — | 5 |
| E_top×wrist | both | (top_t, wrist_t) | off | optional, exploratory, not in any bar |

"Predictor off" = encoder-only training mode (memo §3 gap 0): the SIGReg / LeWM encoder objective with an identity
predictor. Everything else (data, epochs, batch, optimiser, augmentation) is identical between E_x and P_x.
That is 10 trainings in the bars (E_top x2, E_wrist x2, P_top x2, P_wrist x2, E_crop x2); a void under §6.3 adds at most one
retraining per failed model.

## 4. Metric
- **Primary: mutual 10-NN** (paper §3 "Metrics", Huh et al. [22]): for each eval frame i, the 10 nearest neighbours
  of i in space A and in space B (cosine on whitened embeddings, CLS / mean-pooled latent as W1's encoder outputs);
  score_i = |NN_A(i) ∩ NN_B(i)| / 10; alignment = mean over i. Chance = 10 / N_eligible.
- **Same-episode exclusion (owner 2026-11-01):** the candidate neighbours of frame i are the frames of the *other*
  24 val episodes only; frames of i's own episode are never neighbours in either space.
- Frame set: 30 → 5 fps at phase 0 (raw frames 0, 6, 12, …); one common set for every model = frames with index ≥ 2
  at 5 fps (so the history-3 predictor models have full context and every comparison is on the same frames).
  N_eligible = that set's size, counted at §6.1.
- Whitening: ZCA, separately per model, mean and covariance from that model's embeddings on the full eval set, ε = 1e-6
  added to the eigenvalues; computed once and kept fixed inside the bootstrap (as the paper whitens before measuring,
  memo §5 Q6). Ties in distance broken by frame index.
- F_dino: the Hub model's own image processor on the 224×224 transcoded frames, CLS token.
- Secondary (reported, no bar): Procrustes residual ‖Q f_A − f_B‖ / ‖f_B‖ (paper Prop. 2), orthogonal Q fitted on
  the frames of 12 val episodes (the first 12 by episode index in `split.json`'s order), scored on the other 13.
- **Seed pairing (D2, owner-delegated 2026-11-10):** a ladder pair with two seeds per side is computed seed-matched,
  seed s of side A against seed s of side B, s in {3072, 3073}: alignment(3) = mean over s of mutual-10-NN(E_top s,
  E_wrist s); alignment(4) likewise with P_top s, P_wrist s; alignment(2) = mutual-10-NN(E_top 3072, E_top' 3073);
  alignment(1) = mutual-10-NN(E_crop1, E_crop2). Each bootstrap draw resamples the 25 episodes once and applies the same
  draw to both seeds.
- Aggregation and uncertainty: point value on all eligible frames; 95 % percentile CI from 2,000 bootstrap resamples,
  seed 0, **unit of resampling = val episode** (resample 25 episodes with replacement, recompute both alignments (each with the §4 seed pairing) on
  the resampled set, take the difference within each resample — paired). A duplicated episode is a distinct copy;
  exclusion is by original episode id (copies of i's episode are not neighbours of i). Percentiles 2.5 / 97.5 with
  numpy's default (linear) interpolation.

## 5. Ladder and bars
| # | pair | role |
|---|---|---|
| 1 | E_crop: control 1 (form O1) | lower anchor |
| 2 | E_top vs E_top' (same view, two seeds) | upper anchor |
| 3 | E_top vs E_wrist | the measurement |
| 4 | P_top vs P_wrist | the theorem's excluded case |
| 5 | F_dino top vs F_dino wrist | pretrained reference |
| 6 | alignment(3) per wrist-occlusion bin | B3 |

- **B1:** alignment(3) > alignment(1) with CI (the two cameras are better than bad views). — prompt § Design detail
  - pass: 95 % CI of alignment(3) − alignment(1) entirely > 0; fail: entirely < 0; not proven: CI contains 0.
- **B2 (one-sided, owner 2026-11-01):** alignment(4) < alignment(3) with CI (predictor lowers alignment, the
  theorem's direction). — prompt § Design detail; one-sided: owner 2026-11-01
  - pass: 95 % CI of alignment(3) − alignment(4) entirely > 0; fail: entirely < 0 (predictor raises alignment,
    reported as the opposite direction, memo §5 Q3); not proven: CI contains 0.
- **B3 (descriptive, no pass / fail):** alignment(3) vs wrist occlusion bin, monotone or not. — prompt § Design detail.
  Bins (O7): wrist bowl coverage 0-25 / 25-50 / 50-75 / 75-100 %; "monotone" = Spearman of alignment against bin
  midpoint, with a 95 % episode-bootstrap CI. Coverage definition and the not-evaluable rule: §6.5.
- Pass / fail only on B1–B2, on the primary metric.
- Upper anchor (2) and F_dino (5) are reported next to B1 / B2 with their CIs. **O5 (owner 2026-11-02):** if
  alignment(2) is not above alignment(1) with CI (95 % CI of the paired difference not entirely > 0), B1 and B2 are
  **not proven** (the anchors do not separate); the paper reports the ladder and stops.
- **O6 (owner):** the CI is the 95 % episode-bootstrap CI of the paired difference, two-sided interval; B2's direction is
  one-sided (the theorem's): pass = the interval excludes 0 on the predicted side, fail = excludes 0 on the other
  side, else not proven.

## 6. Checks before numbers (no alignment value is computed before these pass)
1. Two val episodes end to end: frame counts per camera equal; timestamps paired; shapes (N_ep, 192) per model.
2. Hand-computed example: mutual 10-NN on 12 synthetic points with known neighbours (identity space → 1.0; a random
   orthogonal rotation of it → 1.0; independent Gaussian → near chance); exclusion mask checked on two episodes.
3. Collapse checks per trained model (`config/lewm_so101.yaml`): min_emb_std 0.1, min_eff_rank_frac 0.1. A model
   failing them voids its pair (O8, owner): retrain that model once with the next seed (3074 for the 3072/3073 runs, same
   settings); a second failure = the model is reported as collapsed and the bar using it is void. No retraining with
   changed settings without a numbered deviation.
4. Checkpoints load strictly (AUDIT_W1_LOADS.md rule: strict `load_state_dict`, right transformers version).
5. Occlusion bins (B3), O3 + D1. Coverage of frame t of val episode e = (G9c HSV bowl-mask area on the wrist frame t) /
   (the same mask's area on the wrist frame 0 of e), the "first-frame area visible" cue (O3). The G9c masks are
   computed on the top camera (`G9C_PROTOCOL.md`), so a wrist-frame mask script with a hand-checked sample (two
   episodes) is a code item before binning. **D1 (owner-delegated 2026-11-10, the option that cannot make a pass easier):**
   the denominator must be non-degenerate: an episode whose wrist frame 0 has bowl-mask area 0 is excluded from B3 and
   listed; B3 is reported **not evaluable on the wrist** (no bins, no Spearman) if no val episode has a non-zero
   denominator or if any of the four bins is empty after the exclusion. No substitute denominator is allowed (not the
   top-view area, not the episode maximum). Values above 100 % go to the 75-100 % bin. B1 and B2 do not use B3.

## 7. Deviations
(none)

## 8. Exploratory (no bar, labelled as such when reported)
E_top×wrist; Procrustes residual; ρ the paper's way (still windows, memo gap 8); scene probe R² (bowl centroids,
gripper state, memo gap 9); rollout test set; per-direction curve (paper Fig. 6); wrist-encode latency share (gap 10);
any choice made after seeing data.

## 9. Resolved items (all before any number)
- **O1** (b): E_crop1 / E_crop2, seeds 3072 / 3073, scored on full frames. **O2** DINO-style crops, scale 0.4-1.0
  (W1's config and `wm_train/*.py` hold no augmentation term). **O3** G9c HSV bowl mask on the wrist frame, first-frame
  visible fraction, val episodes. **O4** W1 encoder's own frame embedding, mean-pooled if both exist; fixed in the
  code item and stated. **O5-O9** as in §4-§6 and §3. Source: `AMENDMENT_2026-11-01.md`, owner 2026-11-02.
- **D1** wrist denominator for B3: §6.5, owner-delegated to agent recommendation 2026-11-10.
- **D2** seed pairing for B1 / B2: §4, owner-delegated to agent recommendation 2026-11-10 (memo FEASIBILITY.md §11.4).
- No bar is missing: B1, B2 (pass / fail / not proven), B3 (descriptive) and the O5 gate are all defined above.
