# La-JEPA S on 182 SO-101 bowl episodes: does the future-prediction loss matter, and how close is it to SmolVLA

Post: https://shissorobotics.com/blog/2026/11/11/paper-vs-hardware-la-jepa/

Paper's claim: partly tested. On this rig: unclear.

The authors' repository (mt-clemente/lajepa) had no commits on 2026-11-06 and 2026-11-07, so the model, the feature store, the trainer and the evaluation here are my implementation from the paper (section 3, Appendix B, Tables 9, 10, 15 and 16). Every choice the paper leaves open is a numbered row in the notes of the work-stack item, not in a bar. The policy has 80.2 M trainable parameters (`expected/runs/full_s0/run.json`, key `params`), against 0.10 B in the paper's Table 16, and trains on stored V-JEPA 2.1 ViT-L features without augmentation or LoRA, which is the paper's design-study variant and not its final recipe.

What is here:
- `PROTOCOL.md`: the question, the split and the four bars A1 to A4, confirmed by the owner on 2026-11-07, before any run (the file's status line still reads "draft").
- `scripts/lajepa.py`, `features.py`, `train.py`: the model, the feature store and the trainer; `selftest.py` checks the attention mask (action queries never read future tokens); `remote_run.sh` is the run on the rented GPU.
- `scripts/smolvla_heldout.py`: the SmolVLA reference scored on the same frames; `scripts/eval_stats.py`: the bootstrap for A1 and A2; `scripts/bench_orin.py`: the Orin latency timer (A3), which was written but not run.
- `expected/runs/<full|nofuture>_s<seed>/`: the six runs (future-loss weight 1 and 0, seeds 0 to 2) with per-episode held-out MAE (`heldout_mae.json`), parameters (`run.json`) and the training log.
- `expected/smolvla_heldout.json`: the reference, `expected/store.json`: the feature-store build, `expected/a1_a2.json`: the bootstrap output the post and its plot read.

Reproduce the analysis (numpy only), from `expected/`:
`python3 -I ../scripts/eval_stats.py --full runs/full_s0 runs/full_s1 runs/full_s2 --nofuture runs/nofuture_s0 runs/nofuture_s1 runs/nofuture_s2 --smolvla smolvla_heldout.json --out /tmp/a1_a2.json`
gives `a1_a2.json` bit for bit (1,000 episode resamples, seed 0). Training needs the dataset `ravediamond/so101_bowls_all_trimmed` at revision `4e5c6fda8dc5ce2252214bc086181d3df8b594b0`, the V-JEPA 2.1 ViT-L weights, and a 24 GB GPU.

Not done: the Orin latency (bar A3, at most 238 ms) was not measured, and the 20 + 20 arm trials against SmolVLA (A4) were not run. The SmolVLA reference was trained on all 182 episodes, including the 10 held-out ones (`in_sample` in `expected/smolvla_heldout.json`), so the A1 comparison is biased against La-JEPA.
