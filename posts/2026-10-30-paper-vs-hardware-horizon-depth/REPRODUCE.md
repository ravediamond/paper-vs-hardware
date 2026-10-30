# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python3 l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. A1 and A2 are recomputed from `expected/a1_cloud.json` and
`expected/a2_latency.json`; the A4 readings from the two per-second CSVs.

## L1: rerun the gates

A1 (PyTorch, lerobot-jetson container, ~40 min on the Orin at 25 W): `scripts/hd_cloud.py`, policy
`ravediamond/smolvla_2026-09-24_14-34-40` @ `c1d76f6`, 20 frames of `ravediamond/so101_bowls_all` (episode list in
`scripts/run_g13b_rtc_offline.py`). A2 and A4 (vla.cpp with `--num-steps`): `scripts/hd_latency.py frames` (container,
once), then `hd_latency.py run` at MAXN_SUPER + `jetson_clocks`, and `scripts/hd_thermal.py run` once per power mode.

## L2

Not applicable: the arm gate runs only if A1 and A2 pass.
