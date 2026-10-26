# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Tolerance: the rounding the post uses (given per line).

## L1: rerun the analysis

Pins: numpy for L0. L1/L2: vla.cpp `f7e0f7f` (2026-10-01) built with CUDA sm_87; checkpoint `ravediamond/smolvla_2026-09-24_14-34-40` @ `c1d76f62` step 020000 (public since 2026-10-26); the latency itself needs the Orin and the arm.

Tick-based numbers are L0 (the tick files are in `expected/`). Re-timing is L2 (Orin + arm, 2 x 10 trials of 25 s).

## L2: rerun on hardware

The robot-side protocol (episodes, bars, controls) is in the post. Not re-verified by this repo.
