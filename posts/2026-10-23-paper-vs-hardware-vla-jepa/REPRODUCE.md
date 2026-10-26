# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: lerobot-jetson container; checkpoint `ravediamond/vla_jepa_so101_bowls_trimmed` @ `c65c98bb` (61.6 GB, **private**).

`python scripts/sample_diversity.py` / `latency.py` on the Orin with the checkpoint. Status: **pending visibility**.

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
