# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: vla.cpp main after #38 (`65d16f5`); the ACT checkpoint `ravediamond/act_2026-09-24_12-19-31` (private).

The port lives upstream: `scripts/convert_act_to_gguf.py` and `src/models/act.cpp` in VinRobotics/vla.cpp. Parity: `vla_predict_check` (vla.cpp CONTRIBUTING). Status: **pending visibility** (ACT checkpoint private).

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
