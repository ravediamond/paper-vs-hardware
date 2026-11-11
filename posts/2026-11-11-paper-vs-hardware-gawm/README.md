# GAWM-S on 182 SO-101 bowl episodes: does the world-model loss help at a quarter of the data

Post: https://shissorobotics.com/blog/2026/11/11/paper-vs-hardware-gawm/

Paper's claim: partly tested. On this rig: unclear.

The authors' repository (gaoxiang12/ga-wm at `662e6d5c81194248156d139089716f3addcbc7ca`) releases the model and its inference code but no trainer
("Training: Coming soon"). The trainer, the data loader and the "w/o world model" variant here are mine, so this is the authors' model trained my way,
not their training recipe.

What is here:
- `PROTOCOL.md`: the question, the split and the four bars, written before any number existed (bars confirmed by the owner before the runs).
- `splits/`: the training episodes (`train172.json`), the held-out episodes (`heldout10.json`) and the quarter subset (`quarter_seed0.json`), all committed before training.
- `scripts/prepare.py`, `scripts/gawm_so101.py`, `scripts/remote_run.sh`: data conversion, loader, trainer, ablation (`NullWM`) and the held-out first-action L1 with the identity null; `scripts/latency.py`: the Orin latency test; `scripts/analyse.py`: the paired episode bootstrap.
- `expected/runs/<model>_<split>_s<seed>/heldout_l1.json` and `run.json`: the 12 runs (2 models x 2 splits x 3 seeds) with per-episode L1 and parameter counts.
- `expected/analysis.json`: the bootstrap output the post and its plot read. `expected/latency.json`: the Orin timing.

Reproduce the analysis (needs numpy only): `python3 -I scripts/analyse.py` rewrites `expected/analysis.json` from the 12 run files (2,000 episode resamples, seed 0). Training needs the dataset
`ravediamond/so101_bowls_all_trimmed` at revision `4e5c6fda8dc5ce2252214bc086181d3df8b594b0` and a 24 GB GPU; the run command is in `scripts/remote_run.sh`.

Not done: the arm trials against SmolVLA (bar B4) were not run, and the latency number comes from a model with random core weights in the precision it was trained in (frozen encoder bf16, core fp32), not FP16.
