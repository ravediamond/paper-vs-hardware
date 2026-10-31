# REMORA readout (2610.39125) on the SO-101 rollouts: protocol (frozen 2026-10-31, before any number)

Paper: Wang, Niu, Zhu et al., *Tracking Task Progress Through Errors and Recovery: A Benchmark and Recurrent
Distributional Model*, arXiv 2610.39125v1 (29 Oct 2026), Maniformer / Tsinghua / Zhejiang / SJTU. Code "upon
acceptance"; dataset promised. Analysis: `journal/queue/2026-10-31_two_papers.md` §2. Draft: `journal/drafts/2610.39125.md`.

## Question

The paper's model is out of reach (π0 backbone, 8 × H200). Its two model-free parts are not: an error-aware progress
target that decreases inside error intervals, and a causal nearest-peak readout of a categorical progress
distribution. On the 69 published rollouts, with the per-frame evidence this rig already computes (G15), does the
nearest-peak readout track progress through the re-grasp and dithering episodes (G14 windows) better than the
monotone stage machine and than independent per-frame readouts, without raising premature or missed events?

## Decision rule

- P1 pass and P2 pass and P3 pass → the nearest-peak readout replaces the monotone stage machine in the harness's
  progress slot (v1.5 verifier plugin, "progress" signal), DECISIONS.md entry; the stage-at-rest judge is unchanged.
- P1 fail → nothing adopted ("does not transfer": a readout cannot fix the evidence).
- P1 pass, P2 or P3 fail → not adopted; published as "replicates the readout, no decision value here".

## Fixed setup

- Data: `ravediamond/so101-bowls-policy-rollouts`, 69 episodes, outcomes
  `data/rollouts/so101-bowls-policy-rollouts_labels.csv` (44 / 25), G15 stage labels
  `shared/gate/data/2026-10-19_g15/transition_labels.csv` (grasp, place frame; −1 = absent), G15 dev / test split from
  `g15_summary.json` (unchanged), G14 reversal windows from `shared/gate/data/2026-10-18_g14/` (per episode, the
  frame ranges where the REKI criterion fired; 14 episodes).
- Evidence: exactly the G15 features (`scripts/run_g15_pace_offline.py`, `extract_features`, modes proprio and
  masks, query every 3 frames), no new feature.
- Stages: 3 — reach (before grasp), hold (grasp to place), placed (after place). Progress atoms z = {0, 0.5, 1}.
- The categorical distribution p_t: two logistic regressions on the dev half only — P(grasped | evidence_t) and
  P(placed | evidence_t) — combined into p_t over the three stages (reach = (1−g)(1−p), hold = g(1−p), placed = p),
  renormalised. Fitted once, coefficients committed to `results/p_t_fit.json` before the test half is read. No
  recurrence, no memory: this is the paper's "explicit window A2" setting with a window of one query.
- Readouts, same p_t:
  - R0: the G15 monotone stage machine (existing code, its fitted thresholds) — the baseline.
  - R1: per-frame argmax.
  - R2: per-frame median of p_t over the atoms (the paper's A5 ablation).
  - R3: causal nearest-peak (paper Eq. 3): peaks = atoms that are local maxima of p_t with p ≥ 0.2; ŷ_t = the peak
    closest to ŷ_{t−1}; ŷ_0 = 0. Ties → the lower atom.
- Error-aware target ỹ_t (paper §3.2): linear from 0 at the first frame to 0.5 at the grasp label, 0.5 to 1 from grasp
  to place, flat at the last stage boundary reached; inside each G14 window, decrease at β = min(α, v/L) per frame,
  then recover with the recomputed slope. For episodes with no place label the target ends at the last reached
  boundary. Built once, committed before any readout is scored.
- Statistics: episode bootstrap, 2,000 resamples, seed 0, 95 % CIs on paired differences; McNemar for per-event
  counts where paired.

## Metrics (test half, 35 episodes as G15's split)

1. **Premature / missed** on the Gemini grasp and place labels, G15's definitions and tolerance (event fires more than
   the tolerance before the label = premature; never fires or fires after the tolerance = missed), per readout.
2. **Flips per episode**: number of changes of the readout's stage value.
3. **Regression recall**: fraction of G14 windows (test-half episodes) inside which the readout decreases at least once;
   **false regressions**: fraction of episodes with no G14 window in which the readout decreases at least once.
4. **MAE and curvature** vs ỹ_t (paper Eq. 5, 7), all frames inside the labelled span.

## Bars

**P1 (the readout does not hurt detection):** R3 premature ≤ R1 premature and R3 missed ≤ R1 missed + 2 points,
paired on episodes. FAIL if R3 missed > R1 missed + 5 points.

**P2 (the readout stops jumping):** R3 flips per episode ≤ 0.5 × R1 flips (paired CI upper bound ≤ 0.7).

**P3 (the readout sees regressions the stage machine cannot):** R3 regression recall ≥ 0.5 on the test-half G14
windows AND false regressions ≤ 0.2; R0 recall is 0 by construction (reported).

**Descriptive, no bar:** MAE and curvature vs ỹ_t for R0–R3 (the paper's primary metric; here the target is built from
the rig's own labels, so it is reported, not passed); the 7 stalled / retried-grasp failure episodes by hand (10 + 10
video check), with the readout's trace next to the video, as the G14 check that was never done.

## Checks before any number is read

(1) p_t fit on the dev half reproduces G15's dev-half grasp detection within 2 points when read with R0 (same
evidence, same thresholds); (2) ỹ_t is in [0, 1], non-decreasing outside the G14 windows, decreasing inside them;
(3) R3 on a synthetic p_t with one jump between two equal peaks does not flip (rule sanity).

## Notes

Three atoms are coarse; the test is of the rule, not of the resolution, and the draft says so. G14 windows are a
mechanical criterion (reversal at ≥ 3 chunks), not human error labels; the hand check is reported beside them. The
paper's memory and backbone claims are not tested and the register row says "not tested here". Anything decided after
this commit is exploratory.

## Cost

$0. ≈ 1 h Orin or Mac (features are cached from G15 if present, else recomputed from the parquet + video, ~30 min).

## Outputs

`projects/remora-readout/README.md`, `projects/remora-readout/results/` (`p_t_fit.json`, `target.jsonl`,
`readouts.jsonl` per frame, `summary.json`), EXPERIMENTS.md §2 row, draft sections 5–8.
