# OpTTA on an Orin NX: replan cost, one darkening severity, and the prediction-only half on the SO-101 monitor

Post: https://shissorobotics.com/blog/2026/11/12/paper-vs-hardware-optta/

Paper's claim: partly tested. On this rig: unclear.

The paper is "Adapt for Planning, Not just Prediction: Optimistic Test-Time World Model Adaptation" (Mete, Kalathil, Kumar; arXiv 2610.35636). I found no released OpTTA code or checkpoint, so the OpTTA and AdaJEPA here are my implementation from the paper's Algorithm 1, Equations 6 to 8 and Appendix A.1 Table 3 (`scripts/optta_pusht.py`). The public parts used are the LeWM model code, the PushT environment and the official `quentinll/lewm-pusht` checkpoint (commits in `PROTOCOL.md` section 7).

What is here:
- `PROTOCOL.md`: the question, the four bars H1 to H4 (confirmed 2026-11-07 before any number) and the deviations; the status line is the one written then.
- `scripts/optta_pusht.py`: own AdaJEPA and OpTTA on LeWM PushT; `run_h2.sh`: the detached driver for the nine H2 runs; `aggregate.py`: the seed-paired bootstrap that writes `expected/summary.json` (H1 and H2).
- `scripts/run_h3_extract.sh`, `extract_w1_h3.py`, `tta_w1.py`: H3 (prediction-only adaptation of the SO-101 surprise monitor under a lighting shift, 69 recorded rollouts) and H4 (time of one adaptation step).
- `expected/summary.json` (H1 ms per replan, H2 success), `expected/h3.json`, `expected/h4.json` (25 W profile), `expected/h4_maxn.json` (MAXN_SUPER), `expected/h2/h2_<method>_s<seed>.json` (per-run outcomes). The plot on the post reads `expected/summary.json`, key `H2.optta_minus_adajepa.pp`.

Reproduce the aggregation: `scripts/aggregate.py` reads `projects/optta/results/h2_<method>_s<seed>.json` (line 6 sets that path); copy `expected/h2/` there, or change `R`, then run it from the folder above `projects/`. The H2 runs take about 14 hours on the Orin at 25 W; the H3 latents need the board's docker image and the rollouts dataset, which are not in this folder.

Limits: every PushT success rate is under 10 %, so the H2 comparison has little power; the H3 numbers come from one corruption (gamma 1.5, gain 0.7) on one dataset; the paper's optimism term needs a planner and the SO-101 policy has none that works, so H3 and H4 do not test optimism.
