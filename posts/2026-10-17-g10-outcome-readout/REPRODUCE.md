# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: W1 / W2 weights (public since 2026-10-26); rollouts `ravediamond/so101-bowls-policy-rollouts` @ `bad1c8ea`.

`python scripts/g10_eval.py --help`. Status: inputs public since 2026-10-26 (W1 / W2′); L1 rerun 2026-11-05 on the Orin (W1 image, `rig/w1/L1_W1.md`): **FAIL-ENV** (the script keeps `<local path>` constants and the original repo layout).

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
