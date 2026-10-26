# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Each line says whether the number is recomputed from rows or read from the pinned file.

## L1: rerun the analysis

Pins: JetPack 6.2 (L4T R36.4.3), CUDA 12.6; the image is published as `ghcr.io/ravediamond/lerobot-jetson:public`.

Build the image as in `scripts/JETSON_ORIN_BUILD_GUIDE.md`, then `python -c "import torch; print(torch.cuda.is_available())"` inside it.

## L2: rerun on hardware

The robot-side protocol is in the post. Not re-verified by this repo.
