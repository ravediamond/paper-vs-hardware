# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: W1 weights (**private until gate 2**); rollouts @ `bad1c8ea` (public); Gemini 3.8 Flash for G15 / G16-B / tau.

W1-A / W1-B / G11-B: `python scripts/run_w1_offline_screening.py --w1 <W1 weights_final.pt>`. Status: **pending visibility**. G15 / G16-B / tau need an API key (not rerun here).

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
