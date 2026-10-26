# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: RoboJEPA checkpoints (public, upstream); vast.ai A100 for Gate A; Orin NX for Gate B.

Gate A needs a CUDA GPU with the 1B model: not rerun here. Gate B: `python scripts/bench_orin.py` on an Orin.

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
