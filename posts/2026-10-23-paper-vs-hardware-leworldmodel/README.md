# LeWorldModel: learns the arm's dynamics, does not plan better than the commanded joints

Post: https://ravediamond.github.io/blog/2026/10/23/paper-vs-hardware-leworldmodel/ (shissorobotics.com results 4 and 5).

## Claim

W1 (multi-view LeWM, ViT-tiny, 100 epochs) predicts 30 steps ahead with error 0.31 vs 0.89 for copying the last frame; ranking 8 action chunks, its best top-1 is 0.60 (bar 0.8) while an action-only baseline with no model reaches 0.68-0.90.

## What is not proven

Open loop on the teleop val split only; ranking measured on teleop anchors (G1b), not in closed loop; one training seed.

See `REPRODUCE.md`.
