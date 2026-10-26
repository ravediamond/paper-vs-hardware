# TurboVLA: our own claim, withdrawn (the data, not the port)

Post: https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-turbovla/ (shissorobotics.com result 3).

## Claim

Upstream's own trainer on our bowls data (path A, 35k steps) moves its actions 0.020 deg when the task text is swapped, against 0.105 normalised units (≈ 5.5° of SO-101 pan) for the released LIBERO checkpoint on LIBERO-Goal: the instruction is ignored with the authors' code too.

## What is not proven

One seed; 25 val episodes; the LIBERO control is 0.105 against a 0.095 line (just above); the two numbers are in different units (degrees on our arm, normalised LIBERO action units for the control) and are compared through the 0.095 ≙ 5° line, not directly.

See `REPRODUCE.md`.
