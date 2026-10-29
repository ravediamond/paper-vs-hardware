# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python3 l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Means, anisotropies and B1–B3 are recomputed from the per-joint R² in
`expected/gate0.json`.

## L1: rerun the analysis

W1: `ravediamond/lewm-w2-so101-bowls` `w1/weights_final.pt` @ `d0336db6`; data `ravediamond/so101_bowls_all_trimmed`
@ `4e5c6fda`, G1b split. `scripts/gate0_extract.py` (W1 container, strict load) writes the latents; `scripts/gate0_analyze.py`
(numpy) writes `gate0.json`.

## L2

Gate 1 (random-walk data on the arm and a retrain) is not run.
