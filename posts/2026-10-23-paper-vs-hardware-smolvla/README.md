# SmolVLA on the Orin: 974 ms to 238 ms per chunk

Posts:
- https://shissorobotics.com/blog/2026/10/01/smolvla-4x-faster-on-jetson-with-vlacpp/
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-smolvla/

On the site: shissorobotics.com result 1.

## Claim

SmolVLA chunk inference 974 ms in PyTorch, 390 ms in vla.cpp, 238 ms with `--flash-attn --mm-prec default`, same checkpoint, Orin NX, MAXN_SUPER; sync loop blocked 39 % -> 14 %.

## Compute

**Compute.** External GPU: none for these posts (Orin NX; 2 x 10 vla.cpp trials + 10 PyTorch trials of 25 s on the arm). The SmolVLA fine-tune's cost is not recorded in the research repo.

## What is not proven

n = 10 trials per engine; success 9 / 9 / 8 of 10 says nothing either way; sync loop only.

See `REPRODUCE.md`.
