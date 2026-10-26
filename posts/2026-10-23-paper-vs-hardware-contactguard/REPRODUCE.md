# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Tolerance: the rounding the post uses (given per line).

## L1: rerun the analysis

Pins: lerobot-jetson container + transformers 5.19.0; rollouts `ravediamond/so101-bowls-policy-rollouts` @ `bad1c8ea` (public); W1 weights as in the LeWorldModel folder (**private until gate 2**).

```bash
python scripts/run_w1_offline_screening.py --w1 weights/w1/weights_final.pt --out out/w1_screening
python scripts/run_w1c_offline.py --w1 weights/w1/weights_final.pt --out out/w1c
```
Status: **pending visibility** (W1 weights private).

## L2: rerun on hardware

The robot-side protocol (episodes, bars, controls) is in the post. Not re-verified by this repo.
