# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: Flora-Rewarder-27B (NVIDIA, noncommercial licence), Qwen3.5-27B / 9B on an >= 80 GB GPU; Gemini API.

Needs an >= 80 GB GPU (vast.ai) and an API key: L0 only here.

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
