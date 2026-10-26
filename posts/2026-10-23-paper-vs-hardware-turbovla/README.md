# TurboVLA: our own claim, withdrawn (the data, not the port)

Posts:
- https://shissorobotics.com/blog/2026/10/08/turbovla-a-policy-that-ignored-its-instruction/
- https://shissorobotics.com/blog/2026/10/23/paper-vs-hardware-turbovla/

On the site: shissorobotics.com result 3.

## Licence of the checkpoint

The path-A checkpoint is not Apache-2.0 like this repo's code: its weights are initialised from TurboVLA's released LIBERO checkpoint (DINOv3 backbone), so use and redistribution fall under the [DINOv3 License](https://huggingface.co/H-EmbodVis/TurboVLA/blob/main/DINOv3_LICENSE.md), a copy of which ships with the checkpoint.

## Claim

Upstream's own trainer on our bowls data (path A, 35k steps) moves its actions 0.020 deg when the task text is swapped, against 0.105 normalised units (≈ 5.5° of SO-101 pan) for the released LIBERO checkpoint on LIBERO-Goal: the instruction is ignored with the authors' code too.

## Compute

**Compute.** External GPU: yes (vast.ai). Path A (upstream trainer, 35k steps) ≈ $2.0 + ~$0.8 of failed starts. The LIBERO control ran on the Orin NX ($0). The port-era trainings behind the 10-08 post (several variants, 0/3 on the arm) have no cost recorded in the research repo. Total recorded: ~$2.8.

## What is not proven

One seed; 25 val episodes; the LIBERO control is 0.105 against a 0.095 line (just above); the two numbers are in different units (degrees on our arm, normalised LIBERO action units for the control) and are compared through the 0.095 ≙ 5° line, not directly.

See `REPRODUCE.md`.

`expected/port/` holds the port-era README (the 10-08 post's numbers); path A supersedes it.
