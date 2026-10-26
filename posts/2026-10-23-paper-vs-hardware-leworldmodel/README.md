# LeWorldModel: learns the arm's dynamics, does not plan better than the commanded joints

Posts:
- https://shissorobotics.com/blog/2026/10/11/a-jepa-world-model-for-3-dollars-30/
- https://shissorobotics.com/blog/2026/10/13/the-world-model-loses-to-the-commanded-joints/
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-leworldmodel/

On the site: shissorobotics.com results 4 and 5.

## Claim

W1 (multi-view LeWM, ViT-tiny, 100 epochs) predicts 30 steps ahead with error 0.31 vs 0.89 for copying the last frame; ranking 8 action chunks, its best top-1 is 0.60 (bar 0.8) while an action-only baseline with no model reaches 0.68-0.90.

## Dynamics

Site result 4. Open-loop error at 30 steps: world model 0.31, copy the last frame 0.89 (`expected/action_sensitivity.json`, `by_k["30"]`). Script: `scripts/eval_action_sensitivity.py` (K = 1..30, open loop vs copy-last).

## Planning

Site result 5. Ranking 8 action chunks, top-1: world model best cell 0.60 against the 0.8 bar (`expected/g1b_range_w1.json`); an action-only baseline with no model reaches 0.68-0.90 (`expected/g1_investigation.json`, `pair.action_only`). Scripts: `scripts/g1b_range.py`, `scripts/g1_diag.py`, `scripts/g1_diag2.py`.

## Compute

**Compute.** External GPU: yes (vast.ai). G0 100-epoch training $3.30 + $2.96 of failed attempts; environment bring-up ~$0.28; G1 probes ~$0.06. Total ~$6.60 (G0 exact, the two small ones estimated). Retries: the $2.96 covers the failed G0 attempts listed in `wm_train/README.md`. G1b, G1 diagnosis, G2: Mac, $0.

## What is not proven

Open loop on the teleop val split only; ranking measured on teleop anchors (G1b), not in closed loop; one training seed.

See `REPRODUCE.md`.
