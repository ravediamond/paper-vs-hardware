# G6a report: on-robot done / stall check with world-model latents

Protocol: [`../../G6A_PROTOCOL.md`](../../G6A_PROTOCOL.md) (frozen 2026-10-13, amended once before any number). Raw numbers: `g6a_results.json`. Code: `g6a.py`, `g6a_orin_export.py`, `g6a_orin_bench.sh`.
Data: 182 episodes, 157 train / 25 held-out; two tasks (blue-in-pink, pink-in-blue). Rest tolerance 0.96 (p95 of first-frame distances). 7 train and 2 val episodes do not end at rest (kept). Every episode has 1 grasp cycle.

## Variant table (held-out 25 episodes, nearest sub-goal distance, 95% CI by bootstrap over episodes)

Pass bars: T1 median Spearman <= -0.8; T2 AUROC >= 0.95.

| Variant | T1 median Spearman | T2 AUROC | TPR / FPR at done threshold | T2 secondary (mid-episode negatives) |
|---|---|---|---|---|
| W1 (192-d, both views) | -0.499 [-0.609, -0.383] | 0.973 [0.934, 1.000] | 0.92 / 0.163 | 0.9998 |
| W2 8x8 top, all cells | -0.381 [-0.541, -0.286] | 1.000 [1.000, 1.000] | 0.76 / 0.00 | 0.999 |
| W2 8x8 top, work32 | -0.421 [-0.648, -0.309] | 1.000 [1.000, 1.000] | 0.84 / 0.00 | 0.999 |
| W2 8x8 both, all | -0.320 [-0.352, -0.300] | 0.981 [0.943, 1.000] | 0.84 / 0.00 | 1.000 |
| W2 8x8 both, work32 | -0.351 [-0.385, -0.308] | 0.986 [0.959, 1.000] | 0.88 / 0.00 | 1.000 |
| W2 16x16 top, all | -0.387 [-0.564, -0.306] | 1.000 [1.000, 1.000] | 0.80 / 0.00 | 0.999 |
| W2 16x16 top, work128 | -0.420 [-0.733, -0.307] | 1.000 [1.000, 1.000] | 0.84 / 0.00 | 0.996 |
| W2 16x16 both, all | -0.318 [-0.365, -0.281] | 0.965 [0.920, 0.997] | 0.84 / 0.058 | 1.000 |
| W2 16x16 both, work64 | -0.332 [-0.419, -0.289] | 0.961 [0.906, 0.997] | 0.84 / 0.043 | 1.000 |

Chosen W2 variant (best train leave-one-episode-out T2 AUROC, 0.9998, chosen on train only): **W2 16x16, top view, all cells**. Selected work-area N: 8x8 top 32, 8x8 both 32, 16x16 top 128, 16x16 both 64.
Mean-of-3-nearest distance gives the same picture (see json).

## Readings

- **T1 fails for every variant** (best -0.50 for W1, W2 -0.32 to -0.42, bar -0.8). Distance to the nearest sub-goal does not fall monotonically with time. Likely because the nearest bank frame changes along the episode and the arm path is non-monotone in latent space; not investigated (exploratory only).
- **T2 passes for W2 top-view variants** (AUROC 1.00, CI [1,1]); W1 0.973. Caveat: negatives are at-rest frames before the first grasp, which is an easy contrast (arm at rest, bowls not moved). Mid-episode negatives also separate (>= 0.996).
- Done threshold (p95 of train end-frame distances) catches 76-92% of end frames at 0-16% FPR.
- **Stall rule is not usable as is.** Frozen choice W = 8 s, eps = 0.02 x d_ref (d_ref 9.34); no setting reached zero train false stalls, so the fallback applied. False-stall rate: 0.38 on train, 0.32 on held-out (indicative). Shorter windows are at 0.99-1.00. Consistent with T1: distance stays flat in normal operation.

## T3 / T4 (gate rollouts with deliberate failures): pending

The 81-rollout archive is on the owner's old Mac. Even when run, only 8 failures have video, so T3/T4 are **indicative only** (owner decision 3), not part of the decision. Robometer comparison dropped until G9a.

## T5 (Orin NX, W2 encoder TensorRT FP16): pending

Jetson was not reachable today. Export verified on the Mac (`g6a_orin_export.py --check`). A poller runs `g6a_orin_bench.sh` in `~/g6a` on the Jetson (only folder written) if it comes up. "SmolVLA loaded" measurement pending.

## Decision (section 6)

No adoption decision today. T1 fails for all variants and the stall rule has a 32-38% false-stall rate, so a W2 distance-based **stall** monitor is not supported. A **done** detector looks feasible (T2 pass) but is only tested on an easy contrast; the decisive test (T3/T4 on brick rollouts with deliberate failures, G4) is still to run.

## Owner decisions after this report (2026-10-14)

- **Distance-based stall detection is dropped** (the T1 fail explains the false-stall rate). Future stall detection = Robometer progress plateau, to be tested in G9a.
- **The T2 pass is a sanity check only** (easy contrast). The done detector stays **unadopted** until near-miss failures exist.
- **T5 stays on the poller, low priority.**
- **Order once the old-Mac archive lands:** G9a first (Robometer zero-shot on the 73 rollouts with video: 65 ok / 8 fail), then T3/T4 in the same pass; both reported as indicative (8 failures).

## Deviations

1. Owner decisions 1-6 (2026-10-13), recorded in the protocol before computing: W2 best-val as candidate, W3 skipped, W1 floor; features recomputed locally; T3/T4 indicative only (< 15 failures); T1/T2 only for W1 and W2; T5 limited to `~/g6a`, encoder alone; report in `results/g6a/`.
2. Stall tie-break amended to the largest eps (4287613), before any number.
3. Grasp-cycle counting fixed (counted the final re-close; now counts close-then-open cycles). Does not enter any metric.
4. Mac MPS used for all compute; no vast spend.
5. Plot of distance vs time for gate rollouts pending with T3/T4. Held-out successes: `g6a_distance_vs_time_val.png`.
