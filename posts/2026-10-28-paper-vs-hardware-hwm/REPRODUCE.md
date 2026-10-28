# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python3 l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. The bars are recomputed from the per-horizon means in
`expected/gate0.json`; the rest is read from the pinned files.

## L1: rerun the analysis

Pins in `env/PINS.md`. Two steps:

1. Latents (Orin or any CUDA box, in the W1 container with this repo mounted at `/pvh`):
   `python scripts/hwm_extract.py --data <so101-bowls-h264-224> --w1 <weights_final.pt> --out out` (~20 min on the Orin).
   Check before training (the deviation note's rule): copy-last MSE at 30 frames ≈ 0.9 and W1 below it.
2. High level (CUDA GPU, ~25 min on an RTX 3090): `python scripts/hwm_train.py out results`, which writes
   `results/gate0.json` with the bars.

## L2: rerun on hardware

Not applicable: gate 0 is offline; gate 1 was not run (it needs H1).
