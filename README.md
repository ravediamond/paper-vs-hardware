# paper-vs-hardware

Code, pinned data and checkpoints to reproduce each post of the *paper vs hardware* series: robot-learning papers
tested on a $300 SO-101 arm with two webcams and a Jetson Orin NX. Posts: https://ravediamond.github.io/papers/ ·
site: https://shissorobotics.com

| post | claim (short) | L0 | L1 |
|---|---|---|---|
| [LeWorldModel](posts/2026-10-23-paper-vs-hardware-leworldmodel/) | world model 0.31 vs copy-last 0.89; plans no better than an action-only baseline | ✔ | pending checkpoint visibility |
| [ContactGuard](posts/2026-10-23-paper-vs-hardware-contactguard/) | imagined-latent probe 0.70 vs current latent 0.82 | ✔ | pending checkpoint visibility |
| [ReAP](posts/2026-10-23-paper-vs-hardware-reap/) | risk head 0.93 → 0.789 vs 0.708 shuffled with controls | ✔ | pending checkpoint visibility |
| [Robometer](posts/2026-10-23-paper-vs-hardware-robometer/) | Robometer-4B 0.818 pooled; deterministic judge at rest, 0 false positives | ✔ | judge ✔ (public data); Robometer L0 only |
| [TurboVLA](posts/2026-10-23-paper-vs-hardware-turbovla/) | authors' trainer on our data: task swap 0.020° vs 0.105 control | ✔ | pending checkpoint visibility |
| [SmolVLA](posts/2026-10-23-paper-vs-hardware-smolvla/) | 974 → 238 ms per chunk on the Orin with vla.cpp | ✔ (tick files) | L2 only (Orin + arm) |
| [Chunk boundaries](posts/2026-10-25-vla-chunk-boundary-jumps/) | prefix overwrite cuts the boundary jump 71–73 % at +0.4 ms | ✔ | pending checkpoint visibility |

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
