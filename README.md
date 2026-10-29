# paper-vs-hardware

This repository backs **[shissorobotics.com](https://shissorobotics.com)**, a register of robot-learning papers
re-tested on a ≈ $930 rig (an SO-101 arm, two webcams and a Jetson Orin NX), each with its bar, verdict, cost and
code: **[the register](https://shissorobotics.com/papers/)**. Every post there has a folder here with the code, pinned
data and checkpoints to reproduce it.

| folder | posts | compute | cost (USD, retries incl.) | L0 | L1 |
|---|---|---|---|---|---|
| [2026-09-09-lerobot-jetson-container](posts/2026-09-09-lerobot-jetson-container/) | 09-09 LeRobot on a Jetson | Orin NX | 0 | ✔ | build (L2-like) |
| [2026-09-29-three-policies-so101](posts/2026-09-29-three-policies-so101/) | 09-29 three policies · 10-26 p99 is not a tail | Orin NX + Mac, arm | 0 (training n/r) | ✔ | L2 (arm) |
| [2026-10-23-paper-vs-hardware-smolvla](posts/2026-10-23-paper-vs-hardware-smolvla/) | 10-01 SmolVLA 4x · PvH SmolVLA | Orin NX, arm | 0 (training n/r) | ✔ | L2 (arm) |
| [2026-10-06-twelve-vlas-orin](posts/2026-10-06-twelve-vlas-orin/) | 10-06 twelve VLAs | Orin NX | 0 | ✔ | Orin needed |
| [2026-10-23-paper-vs-hardware-turbovla](posts/2026-10-23-paper-vs-hardware-turbovla/) | 10-08 TurboVLA · PvH TurboVLA | vast.ai GPU | ~2.8 (incl. ~0.8 failed starts; port-era n/r) | ✔ | inputs public (2026-10-26, DINOv3 License); rerun not yet re-verified |
| [2026-10-23-paper-vs-hardware-vla-jepa](posts/2026-10-23-paper-vs-hardware-vla-jepa/) | 10-08 VLA-JEPA $10 · PvH VLA-JEPA | vast.ai GPU | ~9.7 | ✔ | pending checkpoint visibility |
| [2026-10-23-paper-vs-hardware-leworldmodel](posts/2026-10-23-paper-vs-hardware-leworldmodel/) | 10-11 $3.30 world model · 10-13 WM loses · PvH LeWorldModel | vast.ai GPU | ~6.6 (incl. 2.96 failed attempts) | ✔ | inputs public (2026-10-26); rerun not yet re-verified |
| [2026-10-12-act-vlacpp-port](posts/2026-10-12-act-vlacpp-port/) | 10-12 ACT port | Orin NX | 0 (training n/r) | ✔ | pending checkpoint visibility |
| [2026-10-14-g6-progress-latents](posts/2026-10-14-g6-progress-latents/) | 10-14 latent distance · PvH H-JEPA | Mac | 0 | ✔ | inputs public (2026-10-26); rerun not yet re-verified |
| [2026-10-23-paper-vs-hardware-robometer](posts/2026-10-23-paper-vs-hardware-robometer/) | 10-16 judging a rollout · PvH Robometer · 10-26 audit your labels | vast.ai GPU | ~8.55–9.55 (incl. G9b ~2.3–3.3 never trained, ~1 OOM re-score) | ✔ | judge ✔ public; Robometer L0 only |
| [2026-10-17-g10-outcome-readout](posts/2026-10-17-g10-outcome-readout/) | 10-17 what the WM is for | Mac | 0 | ✔ | inputs public (2026-10-26); rerun not yet re-verified |
| [2026-10-19-six-monitors-w1](posts/2026-10-19-six-monitors-w1/) | 10-19 six monitors · PvH VLA-Corrector | Orin NX + Gemini API | ~2.1–3.1 API | ✔ | inputs public (2026-10-26); rerun not yet re-verified; API |
| [2026-10-20-flora-rewarder-input-format](posts/2026-10-20-flora-rewarder-input-format/) | 10-20 input format | vast.ai GPU + Gemini API | ~4.95 (2.2 GPU + 2.75 API) | ✔ | GPU ≥ 80 GB: L0 only |
| [2026-10-22-voided-gates](posts/2026-10-22-voided-gates/) | 10-22 two gates voided | Orin NX | 0 | ✔ | inputs public (2026-10-26); rerun not yet re-verified |
| [2026-10-23-paper-vs-hardware-contactguard](posts/2026-10-23-paper-vs-hardware-contactguard/) | PvH ContactGuard · PvH FAIL-Detect/SAFE | Orin NX | 0 | ✔ | inputs public (2026-10-26); rerun not yet re-verified |
| [2026-10-23-paper-vs-hardware-dino-wm](posts/2026-10-23-paper-vs-hardware-dino-wm/) | PvH DINO-WM | vast.ai GPU | ~1.7 (W2 < 1 bounded) | ✔ | inputs public (2026-10-26); rerun not yet re-verified |
| [2026-10-23-paper-vs-hardware-d-jepa](posts/2026-10-23-paper-vs-hardware-d-jepa/) | PvH D-JEPA | Orin NX | 0 | ✔ | inputs public (2026-10-26); rerun not yet re-verified |
| [2026-10-23-paper-vs-hardware-reap](posts/2026-10-23-paper-vs-hardware-reap/) | PvH ReAP · 10-26 audit your labels | Orin NX + Gemini API | < 0.1 API | ✔ | inputs public (2026-10-26); rerun not yet re-verified |
| [2026-10-23-paper-vs-hardware-reki](posts/2026-10-23-paper-vs-hardware-reki/) | PvH REKI | Orin NX + vast.ai GPU, arm | ~13.05 (training 11.25 + smoke 0.60 over 3 attempts + serving 1.20) | ✔ | ✔ public rollouts (CPU); REKI arm trials: L2 |
| [2026-10-23-paper-vs-hardware-robojepa](posts/2026-10-23-paper-vs-hardware-robojepa/) | PvH RoboJEPA | vast.ai GPU | ~1.2 | ✔ | GPU (1B encoder) |
| [2026-10-25-vla-chunk-boundary-jumps](posts/2026-10-25-vla-chunk-boundary-jumps/) | 10-25 chunk boundaries · PvH Real-Time Chunking | Orin NX | 0 | ✔ | inputs public (2026-10-26); rerun not yet re-verified |
| [2026-10-26-vla-jepa-rope-compat](posts/2026-10-26-vla-jepa-rope-compat/) | 10-26 upstream fix 47 % worse | Orin NX | 0 | ✔ | pending checkpoint visibility |
| [2026-10-28-paper-vs-hardware-hwm](posts/2026-10-28-paper-vs-hardware-hwm/) | PvH HWM | Orin NX + vast.ai GPU | ~0.4 (one invalid run incl.) | ✔ | latents not published: rebuild on the Orin, then GPU |
| [2026-10-29-paper-vs-hardware-lejepa](posts/2026-10-29-paper-vs-hardware-lejepa/) | PvH LeJEPA identifiability | Orin NX | 0 | ✔ | W1 container |
| [2026-10-29-paper-vs-hardware-surveyor](posts/2026-10-29-paper-vs-hardware-surveyor/) | PvH SURVEYOR | Orin NX | 0 | ✔ | W1 latents, numpy |
| [2026-10-29-vla-jepa-surprise-robot-day](posts/2026-10-29-vla-jepa-surprise-robot-day/) | 10-29 VLA-JEPA surprise robot day | Orin NX, arm | 0 | ✔ | L2 (arm) |

**All 39 posts:** ~$46.3–47.3 on vast.ai and ~$4.9–5.9 of API (Gemini; Flora on vast is in the GPU figure), failed or never-finished attempts included where the research notes record them: G0 ($2.96), HWM's red-only first run (~$0.2), TurboVLA path A (~$0.8), Robometer G9b (~$2.3–3.3, never trained), Robometer OOM re-score (~$1). Not recorded, so not in the total: the training of the ACT, SmolVLA and port-era TurboVLA checkpoints. One post has no run behind it ([the station as a harness](https://shissorobotics.com/blog/2026/10/22/the-station-as-a-harness/)). Arithmetic: `costs.md` in the research repo (not public); every figure is in the folder's README.

## Replication levels

- **L0, check the numbers:** recompute the post's numbers from the small files in `expected/` (seconds, CPU,
  standard library or numpy). Every folder has `l0_check.py`. Where `expected/` holds rows (per-seed, per-frame,
  per-layout, per-tick), the number is recomputed from them; where it only holds a summary (the Robometer pooled AUROC,
  the LeWorldModel open-loop errors), L0 reads the pinned result file and each line says so. Per-rollout scores for a
  true AUROC recomputation are L1.
- **L1, rerun the analysis:** from the public dataset and checkpoint at the pinned revision, rerun the script that
  produced the numbers (CPU or one consumer GPU; time given per folder).
- **L2, rerun on hardware:** the robot-side protocol, for someone with an SO-101. Described, not re-verified here.

Each folder: `README.md` (claim, post link, what is not proven), `REPRODUCE.md` (per level, expected output,
tolerance), `scripts/` (only what the post needs, copied from the research repo), `expected/`, `env/` (pins).
Large files stay on Hugging Face (`ravediamond/...`), always referenced by commit hash. The rig: [`rig/`](rig/).

Licence: Apache-2.0 for the code here. Data and checkpoints keep the licence on their Hugging Face card.
