# TurboVLA: our own claim, withdrawn (the data, not the port)

Post: https://ravediamond.github.io/blog/2026/10/23/paper-vs-hardware-turbovla/ (shissorobotics.com result 6).

## Claim

Upstream's own trainer on our bowls data (path A, 35k steps) moves its actions 0.020 deg when the task text is swapped, against a 0.105 control on the released LIBERO checkpoint: the instruction is ignored with the authors' code too.

## What is not proven

One seed; 25 val episodes; the LIBERO control is 0.105 against a 0.095 line (just above).

See `REPRODUCE.md`.
