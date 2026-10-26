# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: numpy for L0; the arm run itself needs the rig (L2).

No L1: the numbers are arm trials (L2). Tick files are in `expected/`.

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
