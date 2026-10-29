# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python3 l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. The bars are recomputed from `expected/gate0.json` and `gate1.json`.

## L1: rerun the analysis

W1 latents of the 69 rollouts (`ravediamond/so101-bowls-policy-rollouts`, W1 `weights_final.pt` @ `d0336db6`), the
frozen G9c-3 judge (`scripts/surveyor_judge.py`, `expected/judge.csv`), then `scripts/surveyor_gate0.py`,
`scripts/surveyor_imagine.py`, `scripts/surveyor_gate1.py` (numpy).

## L2

Not applicable: no arm run.
