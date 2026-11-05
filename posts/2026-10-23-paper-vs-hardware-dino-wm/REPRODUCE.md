# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: W2′ weights in `ravediamond/lewm-w2-so101-bowls` folder `w2prime/` (public since 2026-10-26); DINOv2-S.

`python scripts/g1_suite.py --help`. Status: inputs public since 2026-10-26 (W1 / W2′); L1 rerun 2026-11-05 on the Orin (`rig/w1/L1_W1.md`): G1b **DIFF** (same verdict, accuracy within the CI, max cell difference 0.018); the §30 probes are not rerun.

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
