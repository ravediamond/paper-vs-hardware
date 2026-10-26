# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: W1 / W2 weights (`ravediamond/lewm-w2-so101-bowls`, **private until gate 2**); data `so101_bowls_all_trimmed` @ `4e5c6fda`.

`python scripts/g6a.py --help`, `python scripts/g6b_eval.py --help`. Status: **pending visibility**.

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
