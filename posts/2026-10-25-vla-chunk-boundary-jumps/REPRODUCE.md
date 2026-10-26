# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Tolerance: the rounding the post uses (given per line).

## L1: rerun the analysis

Pins: vla.cpp fork `ravediamond/vla.cpp` branch `feat/smolvla-prefix-overwrite` @ `805570d` (public), CUDA sm_87; GGUF converted from `ravediamond/smolvla_2026-09-24_14-34-40` @ `c1d76f62` (public since 2026-10-26); data `ravediamond/so101_bowls_all` @ `f989020a` (public).

```bash
VLA_LIBRARY=<vla.cpp>/build-orin/libvla.so PYTHONPATH=<vla.cpp>/bindings/python \
  python scripts/check_vlacpp_prefix_overwrite.py --gguf <smolvla-so101-bowls-020000.gguf> \
  --episodes-from expected/g13b_summary.json --out out/vlacpp_overwrite
```
Status: inputs public since 2026-10-26; the token-free rerun is **not yet re-verified** here. Needs a CUDA GPU (Orin used).

## L2: rerun on hardware

The robot-side protocol (episodes, bars, controls) is in the post. Not re-verified by this repo.
