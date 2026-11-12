# WS-107 results: A Platonic test on a robot (PROTOCOL.md, frozen at fa3636b)

Verdict: **B1 void, B2 void** (O8). Nothing in the pre-registered bars passes or fails. The top-view encoder-only model E_top collapsed twice under the
section 6.3 check (effective rank fraction below 0.1, seeds 3072 and 3073, then the single retraining with seed 3074), so every bar that uses E_top is void.
The P models (predictor on, phase 3) were trained afterwards on the owner's order of 2026-11-11 ("Go ws107 phase 3"); they pass 6.3, but B2 still uses E_top and stays void. Everything computed after the check, rung 4 included, is **exploratory**.

## Section 6 checks (before any alignment value)
- 6.1: all models give (1417, 192) embeddings on the same 1417 frames (25 val episodes, every 6th raw frame); eligible set (5 fps index >= 2) 1367 frames; chance 10-NN 0.0076 (`results/analysis/results.json`).
- 6.2: hand-computed and synthetic checks in `test_align.py` (identity and rotation score 1.0, independent Gaussian near chance, exclusion mask, copy-weighted bootstrap equals brute force) pass.
- 6.3 collapse checks (config thresholds min_emb_std 0.1, min_eff_rank_frac 0.1; the idm and arm-correlation flags of the config are reported in each `collapse_checks.json`, not used for the void):

| model | emb_std_mean | eff_rank_frac | status |
|---|---|---|---|
| E_top_s3072 | 0.879 | 0.084 | void |
| E_top_s3073 | 0.871 | 0.082 | void |
| E_wrist_s3072 | 0.936 | 0.139 | ok |
| E_wrist_s3073 | 0.943 | 0.139 | ok |
| E_crop1 | 0.941 | 0.149 | ok |
| E_crop2 | 0.941 | 0.148 | ok |
| E_top_s3074 (O8 retraining) | 0.875 | 0.082 | void (second failure: E_top reported as collapsed) |
| P_top_s3072 | 0.884 | 0.129 | ok |
| P_top_s3073 | 0.879 | 0.129 | ok |
| P_wrist_s3072 | 0.952 | 0.165 | ok |
| P_wrist_s3073 | 0.950 | 0.167 | ok |

- 6.4: strict `load_state_dict` in `extract_embeddings.py` (all keys matched for every model, `embeddings_meta.json`).
- 6.5: wrist bowl mask (`wrist_mask.py`, G9c thresholds unchanged, 224 px wrist frames): no val episode has first-frame area 0, all four bins non-empty. Hand-check of two episodes in the contact sheet: the masks follow the bowls but also pick up pink on the gripper; coverage can exceed 1.

## Alignment, exploratory (mutual 10-NN, ZCA, same-episode excluded, 2000 episode bootstraps, seed 0; point [95 % CI])
| rung | pair | value |
|---|---|---|
| 1 | E_crop1 vs E_crop2 | 0.514 [0.354, 0.516] |
| 2 | E_top s3072 vs E_top s3073 (void models) | 0.219 [0.074, 0.259] |
| 3 | E_top vs E_wrist, seed-matched (E_top void) | 0.099 [0.041, 0.115] |
| 4 | P_top vs P_wrist, seed-matched (exploratory: the pair is not usable for B2 because E_top is void; the P models themselves pass 6.3) | 0.255 [0.152, 0.286] |
| 5 | F_dino top vs wrist | 0.076 [0.050, 0.095] |
| | a3 - a1 (B1 quantity, void) | -0.436 to -0.287 |
| | a3 - a4 (B2 quantity, void; EXPLORATORY) | -0.200 to -0.092 |
| | a2 - a1 (O5 anchor gate) | -0.372 to -0.190 |

The lower anchor is the highest value: two top-view models trained on random crops of the same frames agree with each other far more than top and wrist
agree. The O5 anchor gate would also fail on these numbers (upper anchor below lower anchor), so B1 and B2 could not have passed even without the void.

## Rung 4, predictor on (phase 3, EXPLORATORY)
Four predictor-on models (W1's action-conditioned predictor; seeds 3072 and 3073 on the top and wrist views) were trained after B1 and B2 had been voided, so this section cannot change a verdict and is labelled exploratory throughout. Alignment(4), seed-matched mean of mutual 10-NN(P_top s, P_wrist s), is 0.255 (95 % episode-bootstrap CI 0.152 to 0.286), against 0.099 [0.041, 0.115] for the encoder-only pair (rung 3, uses the void E_top). The paired difference a3 - a4 is -0.156, CI [-0.200, -0.092], entirely below 0: the predictor-on pair agrees more than the encoder-only pair, the opposite of the direction B2 tests. Because rung 3 depends on the collapsed E_top, this may reflect the collapse of E_top rather than an effect of the predictor (the P_top models do not collapse: effective-rank fraction 0.129). Rung 4 stays below the crop control (0.514) and the upper anchor gate (O5) still fails. Procrustes residual (exploratory): 1.307 for the P pair against 1.339 for the E pair. No pass or fail is declared for B2: it is void.

## B3 (descriptive, exploratory because E_top is void)
Bins 0-25 / 25-50 / 50-75 / 75-100 % wrist bowl coverage, frames per bin [99, 140, 83, 1045], mean mutual-10-NN score per bin [0.106, 0.149, 0.127, 0.089];
Spearman of the per-frame score against bin midpoint -0.099, 95 % CI [-0.213, 0.060] (contains 0: no monotone trend shown).
Secondary Procrustes residual (a3, exploratory): 1.339.

## Spend
Phases 0-2: about 4 USD in 9 vast instances (launcher-log estimate 4.35 USD upper). Phase 3: 4.83 USD (vast credit 13.26 before launching, 8.43 after all four instances were destroyed), in 6 instances: two launches failed in setup (pypi timeout, destroyed by the launcher, about 0.15 USD together) and were relaunched once each. The four trainings ran 4.5 to 5.5 h each (0.41 s/step on an RTX 4080 SUPER, 0.45 to 0.50 s/step on RTX 3090) against the 3.6 USD expected in the memo, which assumed 0.41 s/step on every card; the 290 min limits were extended in place to 300-350 min once the speed was measured. WS-107 total about 9.2 USD of the 30 USD cap.
