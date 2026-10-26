# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Tolerance: the rounding the post uses (given per line).

## L1: rerun the analysis

Pins: Robometer scoring: one >= 48 GB GPU (A6000 used), so L0 only for that part. Judge: CPU, rollouts `ravediamond/so101-bowls-policy-rollouts` @ `bad1c8ea` (public).

Judge (CPU, ~3 min), on the public rollouts at the pinned revision:

```bash
hf download ravediamond/so101-bowls-policy-rollouts --repo-type dataset --revision bad1c8ea --local-dir rollouts
cd scripts && python g9c3_test.py --config g9c_config.json --rule g9c2_rule.json --out ../out \
  --orin ../rollouts --catalog ../l1/empty_catalog.csv
```

Expected: 69 judged, 0 false positives, 5 false negatives, exactly episodes **7, 23, 57, 59, 61** (the Orin rows of
`errors_on_judged` in `expected/g9c3_test_results.json`); accuracy 0.928 on these 69. The published 94.3 % is on 88
rollouts: the other 19 are M1 sessions that are not in the public dataset (empty catalog here).
**Checked 2026-10-26 on an Orin NX (CPU, lerobot-jetson container): pass, same five episodes, 177.7 s.**

Robometer scoring needs a >= 48 GB GPU: L0 only here.

## L2: rerun on hardware

The robot-side protocol (episodes, bars, controls) is in the post. Not re-verified by this repo.
