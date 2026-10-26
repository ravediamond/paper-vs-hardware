# Why your VLA's actions jump at every chunk boundary, and a 0.4 ms fix

Post: https://shissorobotics.com/blog/2026/10/25/vla-chunk-boundary-jumps/ (shissorobotics.com result 1, second sentence).

## Claim

Gradient-free prefix overwrite in vla.cpp's SmolVLA denoising loop cuts the median boundary jump by 71.0 / 73.3 / 71.5 % (d = 8 / 12 / 20, 254 boundaries each) at +0.4 ms per chunk; task success not tested.

## What is not proven

Smoothness offline on recorded episodes; task success with the overwrite is untested (G13a).

See `REPRODUCE.md`.
