# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python3 l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. The bars are recomputed from `expected/gate0.json` and
`expected/gate1_latency.json`.

## L1: rerun the gates

Gate 0 (PyTorch, ~3 h on an Orin NX at 25 W): `scripts/aac_gate0.py` in the lerobot-jetson container, policy
`ravediamond/smolvla_2026-09-24_14-34-40` @ `c1d76f6`, the 20 G13b episodes of `ravediamond/so101_bowls_all`
(`scripts/run_g13b_rtc_offline.py` holds the episode list and the SmolVLA wrapper).

Gate 1 (vla.cpp, ~5 min): Orin NX at MAXN_SUPER + `jetson_clocks`, nothing else on the GPU;
`python3 scripts/hd_latency.py frames` (in the container, once), then `python3 scripts/aac_latency.py run` (host;
starts vla-server with `--flash-attn 1 --mm-prec default` on the SmolVLA GGUF).

## L2

Not applicable: no arm run (the bowls task is at ceiling for SmolVLA).
