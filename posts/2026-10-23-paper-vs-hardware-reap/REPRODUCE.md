# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Tolerance: the rounding the post uses (given per line).

## L1: rerun the analysis

Pins: lerobot-jetson container + transformers 5.19.0; rollouts @ `bad1c8ea`; W1 weights (**private until gate 2**); onsets from Gemini 3.8 Flash in `expected/onsets.csv` (v1 only; v2 uses none).

```bash
python scripts/run_reap_v2.py --help
```
Status: **pending visibility** (W1 weights private).

## L2: rerun on hardware

The robot-side protocol (episodes, bars, controls) is in the post. Not re-verified by this repo.
