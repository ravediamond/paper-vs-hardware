# SmolVLA on the Orin: 974 ms to 238 ms per chunk

Post: https://ravediamond.github.io/blog/2026/10/23/paper-vs-hardware-smolvla/ (shissorobotics.com result 1).

## Claim

SmolVLA chunk inference 974 ms in PyTorch, 390 ms in vla.cpp, 238 ms with `--flash-attn --mm-prec default`, same checkpoint, Orin NX, MAXN_SUPER; sync loop blocked 39 % -> 14 %.

## What is not proven

n = 10 trials per engine; success 9 / 9 / 8 of 10 says nothing either way; sync loop only.

See `REPRODUCE.md`.
