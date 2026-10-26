# LeRobot on a Jetson: the container instead of pip install torch

Posts:
- https://shissorobotics.com/blog/2026/09/09/lerobot-on-jetson-no-pip-install-torch/

## Claim

LeRobot runs on a Jetson Orin NX (JetPack 6.2) from a container with a JetPack-built torch and a source-built torchcodec; pip's aarch64 torch wheels have no Jetson CUDA.

## Compute

**Compute.** External GPU: no (Orin NX only, image builds on the board). Cost: $0.

## What is not proven

One board (Orin NX 16 GB) and one JetPack release.

See `REPRODUCE.md`.
