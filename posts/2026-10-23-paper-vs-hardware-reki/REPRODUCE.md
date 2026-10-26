# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: rollouts `ravediamond/so101-bowls-policy-rollouts` @ `bad1c8ea` (public); CPU.

`python scripts/run_g14_oscillation_audit.py --help` on the public rollouts (CPU, minutes).

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
