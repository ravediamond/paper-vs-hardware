# Paper vs hardware 3: RoboJEPA (action conditioning holds, the encoder is too slow for the Orin)

Posts:
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-robojepa/

## Claim

Pretrained RoboJEPA predicts our bowls latents 10.5-15.8 % better than copy-last and uses the actions (8.9-10.4 % better than shuffled); the 2-view encoder takes 526 ms on the Orin NX (bar 200 ms): no-go on device.

## Compute

**Compute.** External GPU: yes (vast.ai, Gate A): ~$1.2. Gate B on the Orin NX and the Mac: $0. Retries: none recorded.

## What is not proven

1B on 30 clips only (partial sample); encoder latency in eager fp16, TensorRT not tried.

See `REPRODUCE.md`.
