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

On the public rollouts (CPU, seconds; the labels ship with the dataset in `meta/rollout_labels.csv`):

```bash
hf download ravediamond/so101-bowls-policy-rollouts --repo-type dataset --revision bad1c8ea --local-dir rollouts
python scripts/run_g14_oscillation_audit.py --root rollouts --labels rollouts/meta/rollout_labels.csv --out out/g14
```
Expected: `out/g14/g14_episodes.csv` and `g14_summary.json` identical to `expected/` (blob columns empty without
`--blob`). **Checked 2026-10-31 cold on an Orin NX (lerobot-jetson:public, no token): identical, 4 s.**

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
