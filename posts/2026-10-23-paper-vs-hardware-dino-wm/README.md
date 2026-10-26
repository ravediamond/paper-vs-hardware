# Paper vs hardware 2: DINO-WM (W2 / W2′ on frozen DINOv2 patches)

Posts:
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-dino-wm/

## Claim

A predictor on frozen DINOv2-S patches (W2) encodes the arm well (pose R² 0.92) but fails G1b (best-val 0.55-0.61 < 0.8); the retrain W2′ reproduces W2 (R² 0.9193, open loop / copy-last 0.633).

## Compute

**Compute.** External GPU: yes (vast.ai): W2 < $1, W2′ ~$0.65, G1 probes ~$0.06. Total ~$1.7 (W2 bounded, not exact). Retries: none recorded.

## What is not proven

One training each (W2 lost, W2′ retrained with seed 3073); G1b measures ranking, not closed loop.

See `REPRODUCE.md`.
