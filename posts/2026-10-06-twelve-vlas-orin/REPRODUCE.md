# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: vla.cpp `f7e0f7f` era build (CUDA sm_87, MAXN_SUPER); GGUFs from `huggingface.co/vrfai`.

Needs a Jetson Orin NX: `bash scripts/bench_chain.sh` (see vla.cpp `docs/benchmark/` for the template).

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
