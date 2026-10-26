# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: lerobot `d3793b7` (fixed) and `2e88044` (pre-fix) side by side; checkpoint `ravediamond/vla_jepa_so101_bowls_trimmed` @ `c65c98bb` (**private**); data `so101_bowls_all_trimmed` @ `4e5c6fda`.

`python scripts/eval_rope_compat.py compare --ckpt <snapshot> --root <so101_bowls_all_trimmed> --split split.json --windows-per-episode 8 --out results.json` (Orin, ~10 min). Status: **pending visibility** (checkpoint private).

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
