# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: transformers 5.5.4 vs 5.19.0 in the lerobot-jetson image; W1 weights (**private until gate 2**).

`python scripts/strict_load_log.py <W1 weights_final.pt>` under both transformers versions. Status: **pending visibility**.

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
