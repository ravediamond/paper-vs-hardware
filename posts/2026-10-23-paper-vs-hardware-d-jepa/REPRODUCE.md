# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: W1 weights (public since 2026-10-26); data @ `4e5c6fda`.

`python scripts/djepa_operator.py --help`. Status: inputs public since 2026-10-26 (W1 / W2′); L1 rerun 2026-11-05 on the Orin (W1 image, `rig/w1/L1_W1.md`): **REPRO** (451 of 452 values equal, only the timing differs).

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
