# Reproducibility, checked cold (2026-10-31)

Every post in this repository promises two levels: **L0**, recompute its bars from the published rows
(`l0_check.py`, standard library, seconds), and **L1**, rerun its gate on the published hardware and software
(`REPRODUCE.md`). On 2026-10-31 all of it was run the way a stranger would: a fresh clone of this repository
(@ `a83fcc9`, 29 post folders), the public Hugging Face Hub with no token, the published container
`ghcr.io/ravediamond/lerobot-jetson:public` (LeRobot 0.6.2, torch 2.11.0, transformers 5.5.4), on one Jetson Orin NX
16 GB, JetPack 6.2 / L4T R36.4.3, CUDA 12.6, at the 25 W power mode. Fixes found on the way were merged on
2026-11-01 (merge `c3d04d0`); no `expected/` file was edited.

**Tolerance rule.** A post's own tolerance where it states one; otherwise a number is reproduced if it lies inside the
published 95 % CI, or within 5 % relative when no CI is published (timings: within 10 %). Classes: `REPRO`,
`REPRO-WITHIN-CI`, `DIFF` (outside tolerance, both values recorded), `FAIL-ENV` / `FAIL-FILE` / `FAIL-SCRIPT`
(a packaging problem), `SKIP` (needs a larger GPU, the arm, a private checkpoint or the MAXN power mode).

**Result.** L0: **29 / 29 PASS**. L1: **6 REPRO** (5 exact or byte-identical, 1 identical but one timing at 0.2 %),
**0 DIFF**, 12 FAIL-ENV (one cause: the world-model environment is not yet published), 11 SKIP. No published number
was contradicted. GPU time used: about 5.5 h.

| # | post | L0 | L1 | fixed on the branch | content finding | not reproducible on the rig, why |
|---|---|---|---|---|---|---|
| 1 | 09-09 lerobot-jetson-container | PASS (nothing numeric) | REPRO | — | — | — |
| 2 | 09-29 three-policies-so101 | PASS | — | — | — | arm (L2) |
| 3 | 10-06 twelve-vlas-orin | PASS | SKIP | — | — | MAXN power mode, not run; GGUFs unpinned |
| 4 | 10-12 act-vlacpp-port | PASS | SKIP | — | — | ACT checkpoint private |
| 5 | 10-14 g6-progress-latents | PASS | FAIL-ENV | — | — | W1 env unpublished; `--help` only |
| 6 | 10-17 g10-outcome-readout | PASS | FAIL-ENV | — | — | W1 env; `--help` only |
| 7 | 10-19 six-monitors-w1 | PASS | FAIL-ENV | — | — | W1 env; G15 / G16-B / tau need an API key |
| 8 | 10-20 flora-rewarder | PASS | SKIP | — | — | ≥ 80 GB GPU (vast) |
| 9 | 10-22 voided-gates | PASS | FAIL-ENV + FAIL-SCRIPT | — | — | W1 env; weights path hard-coded as a placeholder |
| 10 | 10-23 contactguard | PASS | FAIL-ENV | — | — | W1 env |
| 11 | 10-23 dino-wm | PASS | FAIL-ENV | — | — | W1 env; `--help` only |
| 12 | 10-23 d-jepa | PASS | FAIL-ENV | — | — | W1 env; `--help` only |
| 13 | 10-23 leworldmodel | PASS | FAIL-ENV | — | — | W1 env; `--help` only |
| 14 | 10-23 reap | PASS | FAIL-ENV | — | — | W1 env; `--help` only |
| 15 | 10-23 reki | PASS | REPRO | full L1 command | — | — |
| 16 | 10-23 robojepa | PASS | SKIP | — | — | gate A on an A100; gate B time unstated |
| 17 | 10-23 robometer | PASS | REPRO (judge) | — | — | Robometer scoring ≥ 48 GB |
| 18 | 10-23 smolvla | PASS | — | — | — | arm (L2) |
| 19 | 10-23 turbovla | PASS | SKIP | — | — | unstated; upstream code + unnamed val cache |
| 20 | 10-23 vla-jepa | PASS | SKIP | — | — | checkpoint private |
| 21 | 10-25 vla-chunk-boundary-jumps | PASS | SKIP | — | — | unstated; vla.cpp fork build; imports `so101_harness` |
| 22 | 10-26 vla-jepa-rope-compat | PASS | SKIP | — | — | checkpoint private |
| 23 | 10-28 hwm | PASS | FAIL-ENV + FAIL-FILE | — | — | W1 env; H.264 224 data copy unpublished |
| 24 | 10-29 lejepa | PASS | FAIL-ENV | — | — | W1 env |
| 25 | 10-29 surveyor | PASS | FAIL-ENV | — | — | W1 env |
| 26 | 10-29 vla-jepa-surprise-robot-day | PASS | — | — | — | arm (L2) |
| 27 | 10-30 adaptive-chunking | PASS | REPRO (gate 0) | `so101_harness.episodes` + G13b list; time | — | gate 1 MAXN |
| 28 | 10-30 horizon-depth | PASS | REPRO (A1, byte-identical) | same; A1 time ~40 → ~80 min | — | A2, A4 MAXN |
| 29 | 10-30 pact | PASS | REPRO (gate B, identical) | same; the gate B scorer (`pact_score.py`); checks at k = 8 | — | gate A MAXN |

## Fixed in this pass (merged)

- `reki`: the L1 command in REPRODUCE was `--help` only; the full command is now given.
- `horizon-depth`, `adaptive-chunking`, `pact`: the scripts imported `so101_harness.episodes` and the G13b episode
  list from a private repository; both now ship in each folder.
- `pact`: the gate B scorer is published (`pact_score.py`); checks run at k = 8 as published.
- `horizon-depth`, `adaptive-chunking`: stated wall-clock corrected (A1: about 80 min at 25 W, not 40).

## Not yet reproducible from this repository, by cause

- **World-model environment (12 posts):** the W1 posts need the layer the local `lerobot-jetson-wm` image adds
  (`stable-worldmodel`, `stable-pretraining`, `transformers 5.19.0`) and a standalone loader. To be published.
- **MAXN power mode:** twelve-vlas, and the latency halves of horizon-depth, adaptive-chunking and pact.
- **Larger GPU:** flora-rewarder (≥ 80 GB), robojepa gate A (A100), Robometer scoring (≥ 48 GB).
- **Private checkpoints:** act-vlacpp-port, vla-jepa, vla-jepa-rope-compat.
- **The arm (L2):** three-policies, smolvla, vla-jepa-surprise-robot-day.
