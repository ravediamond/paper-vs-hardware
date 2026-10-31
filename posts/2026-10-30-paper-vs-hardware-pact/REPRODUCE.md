# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python3 l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Gate A's cycle time and both delays are recomputed from
`expected/a2_latency.json` (WS-057 A2, the same session) and `expected/ver_ratio.json`; B-P1 and B-P2 from
`expected/gate_b.json`.

## L1: rerun the gates

Gate B (PyTorch, Orin NX at 25 W, ~12 min of checks + ~72 min): `scripts/pact_flowres.py` in the lerobot-jetson
container, policy `ravediamond/smolvla_2026-09-24_14-34-40` @ `c1d76f6`, the 20 G13b episodes of
`ravediamond/so101_bowls_all` (`--episodes 2 --checks --k 8 --out out/pact_checks` first: 31 boundaries, recorded future below the median candidate on 68 %; then `--k 8,12 --out out/pact`),
then `python scripts/pact_score.py out/pact/rows.jsonl out/pact/gate_b.json`.
**Checked 2026-10-31 cold (`ghcr.io/ravediamond/lerobot-jetson:public`, no token): `rows.jsonl` byte-identical to the
original run, `gate_b.json` identical to `expected/gate_b.json` (34 / 34 numbers).**

Gate A (Orin NX at MAXN_SUPER + `jetson_clocks`, nothing else on the GPU): `scripts/hd_latency.py frames` then
`hd_latency.py run` (vla.cpp, the A2 latency file), `scripts/pact_latency.py ratio` in the container (the PyTorch
ratio), `pact_latency.py combine`.

## L2

Not applicable: gate C (arm) runs only after B-P3 passes, and B-P3 is not computable at the measured delay.
