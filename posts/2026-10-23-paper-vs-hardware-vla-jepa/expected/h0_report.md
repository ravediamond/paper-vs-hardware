# Pre-checks: is there headroom for world-model reranking of VLA-JEPA samples?

Project overview: [`projects/vla-jepa-world-model/README.md`](../README.md).

Date: 2026-10-08. Hardware: Jetson Orin NX 16 GB (MAXN_SUPER), lerobot-jetson:rc (lerobot 0.6.2). Policy:
`ravediamond/vla_jepa_so101_bowls_trimmed` (final step). Data: `ravediamond/so101_bowls_all_trimmed`.
Scripts in this folder, raw outputs in [`out/`](out/), step A details in [`inventory.md`](inventory.md).

## Context

**Paper.** H-JEPA (Zhang, Terver et al., arXiv 2610.06805, Oct 2026). Level 1 is LeWM: a ViT-Tiny encoder and an
action-conditioned predictor trained in latent space, with no pixel reconstruction and SIGReg against collapse. Each
higher level re-encodes every 2nd latent of the level below and predicts further ahead with "macro-actions". The
paper plans top-down by gradient descent: the top level plans toward the goal latent, and its predictions become
subgoals for the level below. Findings this project relies on:

1. Distance to the goal in a higher-level latent is a better progress signal than in level 1 (their Table 1: just
   scoring a flat planner in the level-2 space improves success).
2. Higher levels keep slow task state and drop fast detail, but only when the data has a clear fast/slow split; this
   mostly fails on short manipulation tasks (Cube, Push-T).
3. When scenes vary across episodes the encoder can collapse onto the background and ignore the arm; an
   inverse-dynamics loss (predict the action from two consecutive latents) prevents it.
4. Stagewise training worked best: broad "play" data for level 1, goal-directed demos for level 2.
5. No closed-loop real-robot test; their real-robot results (DROID) are offline only.

**How we use it.** No planning with H-JEPA: the VLA proposes N action chunks, the world model rolls each out, and
the chunk whose predicted outcome is closest to a goal frame's latent is executed.

**Hypotheses, in order.**

- **H0 (these pre-checks):** the VLA's samples vary enough that picking among them can help, beyond a
  no-world-model consensus pick.
- **H1:** a flat LeWM (+ inverse dynamics) trained on our SO-101 data raises real-robot success over the VLA alone and
  over a random pick among the N samples.
- **H2:** scoring in the level-2 latent of a 2-level H-JEPA beats level 1 (finding 1), most likely on the chained
  multi-stage bowls task rather than a single-stage one (finding 2).
- **H3 (evaluator):** the world model's predicted goal distance on logged rollouts predicts real success, enabling
  offline policy evaluation.

## Verdict on H0: no-go for now, pending the harder-layout baseline (step E)

> Update 2026-10-08: H0 closed as no-go (shared bias). H3, the world model as an evaluator of rollouts, is now the
> main hypothesis: plan in [`h3_plan.md`](h3_plan.md). Step E gained `--avg` (execute the mean of the N chunks),
> `--capture-goal`, and a session runbook: [`session_runbook.md`](session_runbook.md).

| Criterion (heuristic) | Go if | Measured | |
|---|---|---|---|
| Oracle gain, `oracle_err / mean_err` | <= ~0.7 overall, or clearly lower where failures happen | **0.76** overall (null 0.92); **0.83** in the grasp phase, where the 2026-10-08 failure happened; arm joints only 0.85 | no |
| Where the spread is | not almost all in the gripper | per joint, the gripper has the largest normalized spread and most of the headroom: gripper 0.58 vs arm 0.85 oracle/mean | mostly gripper |
| Latency, N = 8 vs N = 1 per decision | <= ~1.5x | **1.84x** (643 vs 349 ms); N = 4 is 1.31x | no |
| Baseline success | < ~85% | 90% (9/10) on the current task; the harder-layout set (step E) is the new reference | pending |

Offline numbers are **optimistic (seen during training)**: the VLA was trained on all 182 episodes.

What the numbers say:

- **The samples are a tight bundle sharing one bias.** The 8 chunks differ by 0.033 (normalized,
  mean pairwise distance), about 3% of how far the arm moves in one chunk (1.01). In degrees: arm
  spread 1.08, error to the expert 2.22, best of 8 1.83.
  The spread is mostly step-to-step jitter of a few tenths of a degree, not different strategies, and the samples
  share the same offset from the expert ([`overlays.png`](out/sample_diversity/overlays.png)). Picking the best of
  the bundle cannot remove a bias every sample has.
- **Real headroom above chance is small.** Best-of-8 reaches 0.76 of a random pick's error,
  against 0.92 for random perturbations with the same spread: +0.15 is
  all that structure in the samples adds. In the grasp phase it is
  +0.11, on the arm joints
  +0.11.
- **A free baseline already takes a large share.** Averaging the 8 chunks captures
  **45%** of the oracle's gain (the medoid 26%), with no world
  model and no extra rollout. A world model would have to beat that.
- **More noise widens the spread but not usefully.** At noise x2 the oracle ratio drops to
  0.64, but every sample's error grows so much that best-of-8
  (0.058) is worse than best-of-8 at the deployed noise (0.044). Fewer denoising steps
  reduce the headroom.
- **Latency:** the head runs in float32 and takes ~104 ms per sample, almost linear in N past
  4; N = 8 adds ~0.3 s per decision to a synchronous loop
  that already stalls ~28% of the time.

**Recommendation.** Do not train a world model yet. Run step E on the harder layouts first: it answers whether the
baseline drops below 85% and, from the logged 8 chunks at the failing decisions, whether the samples there differ in
a way that matters (e.g. grasp height or gripper timing). If they do, the next cheapest test is mean-chunk ensembling,
which needs no model; a world model is worth it only for headroom beyond that.

## Step A: inventory

Full table: [`inventory.md`](inventory.md). Key points:

- 182 episodes, 61,462 frames, 30 fps, 2 cameras 640x480 (front_top, wrist), episodes 8.6-15.7 s (median 11 s), two
  tasks (blue-in-pink 90, pink-in-blue 92). Actions: 6 absolute joint targets (degrees; gripper 0-100, 0 = closed).
- Policy: chunk 30 = executed 30 (1 s), 4 Euler denoising steps, images 224x224, backbone bf16, head float32, world
  model off at inference. Actions normalized MIN_MAX to [-1, 1], clipped before unnormalizing.
- Sampling: `torch.randn` in `VLAJEPAActionHead.predict_action` (`action_head.py:325`), no seed, generator, scale or
  temperature; deterministic given the noise. `common.sample_head` reimplements the loop with explicit noise and
  steps, identical to the original (max |diff| 0).
- Batching: `_encode_qwen` once per observation, head batched over N; verified equal to N single calls (6e-7).

## Step B: sample diversity and oracle (optimistic, seen during training)

50 observations from 50 episodes, 10 per phase. Phases from the gripper trace (`common.phase_windows`): start =
first 0.5 s, approach = gripper open until 0.5 s before it closes on the bowl, grasp = +-0.5 s around that close,
transport = holding, release = +-0.5 s around the opening over the target. N = 8 samples per observation, seeds
fixed. Distance between chunks: mean over the 30 steps of the L2 norm over the joints. Null = 8 random perturbations
of the mean chunk with the samples' per-step, per-joint std (32 draws). Consensus = medoid sample. Capture = share of
the oracle's improvement over a random pick that a baseline gets. "sample0" = the plain VLA's chunk.

#### all joints, normalized units

| phase | n | spread | scale_ref | mean_err | sample0 | oracle | meanchunk | consensus | oracle/mean | null | gain>null | consensus capture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start | 10 | 0.031 | 1.112 | 0.059 | 0.057 | 0.045 | 0.055 | 0.057 | **0.76** | 0.94 | +0.18 | 13% |
| approach | 10 | 0.026 | 1.100 | 0.053 | 0.056 | 0.042 | 0.050 | 0.051 | **0.79** | 0.91 | +0.12 | 14% |
| grasp | 10 | 0.019 | 0.751 | 0.052 | 0.051 | 0.043 | 0.051 | 0.052 | **0.83** | 0.94 | +0.11 | 7% |
| transport | 10 | 0.021 | 0.530 | 0.043 | 0.036 | 0.031 | 0.040 | 0.043 | **0.73** | 0.92 | +0.19 | -4% |
| release | 10 | 0.067 | 1.568 | 0.080 | 0.064 | 0.057 | 0.059 | 0.066 | **0.71** | 0.88 | +0.16 | 60% |
| all | 50 | 0.033 | 1.012 | 0.057 | 0.053 | 0.044 | 0.051 | 0.054 | **0.76** | 0.92 | +0.15 | 26% |

#### arm joints, normalized units

| phase | n | spread | scale_ref | mean_err | sample0 | oracle | meanchunk | consensus | oracle/mean | null | gain>null | consensus capture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start | 10 | 0.031 | 1.054 | 0.059 | 0.056 | 0.045 | 0.055 | 0.057 | **0.76** | 0.94 | +0.18 | 13% |
| approach | 10 | 0.014 | 0.459 | 0.032 | 0.033 | 0.029 | 0.031 | 0.032 | **0.91** | 0.96 | +0.05 | 8% |
| grasp | 10 | 0.014 | 0.722 | 0.038 | 0.036 | 0.034 | 0.037 | 0.038 | **0.89** | 0.96 | +0.08 | 6% |
| transport | 10 | 0.016 | 0.367 | 0.033 | 0.035 | 0.030 | 0.031 | 0.032 | **0.89** | 0.96 | +0.06 | 42% |
| release | 10 | 0.024 | 1.213 | 0.041 | 0.040 | 0.034 | 0.037 | 0.040 | **0.84** | 0.94 | +0.10 | 19% |
| all | 50 | 0.020 | 0.763 | 0.041 | 0.040 | 0.034 | 0.038 | 0.040 | **0.85** | 0.95 | +0.11 | 16% |

#### gripper joints, normalized units

| phase | n | spread | scale_ref | mean_err | sample0 | oracle | meanchunk | consensus | oracle/mean | null | gain>null | consensus capture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start | 10 | 0.002 | 0.110 | 0.003 | 0.003 | 0.003 | 0.002 | 0.003 | **0.82** | 0.85 | +0.03 | -30% |
| approach | 10 | 0.018 | 0.896 | 0.031 | 0.034 | 0.020 | 0.029 | 0.030 | **0.65** | 0.85 | +0.19 | 13% |
| grasp | 10 | 0.007 | 0.051 | 0.018 | 0.018 | 0.012 | 0.018 | 0.018 | **0.66** | 0.87 | +0.21 | 4% |
| transport | 10 | 0.007 | 0.204 | 0.011 | 0.003 | 0.002 | 0.011 | 0.013 | **0.20** | 0.80 | +0.60 | -20% |
| release | 10 | 0.054 | 0.816 | 0.057 | 0.041 | 0.033 | 0.038 | 0.043 | **0.58** | 0.81 | +0.23 | 59% |
| all | 50 | 0.018 | 0.415 | 0.024 | 0.020 | 0.014 | 0.020 | 0.021 | **0.58** | 0.83 | +0.25 | 27% |

#### all joints, raw units (degrees; gripper 0-100)

| phase | n | spread | scale_ref | mean_err | sample0 | oracle | meanchunk | consensus | oracle/mean | null | gain>null | consensus capture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start | 10 | 1.740 | 68.779 | 3.194 | 3.042 | 2.250 | 2.966 | 3.111 | **0.70** | 0.95 | +0.24 | 9% |
| approach | 10 | 0.938 | 33.326 | 2.046 | 2.129 | 1.763 | 1.957 | 2.017 | **0.86** | 0.94 | +0.08 | 10% |
| grasp | 10 | 0.832 | 39.220 | 2.407 | 2.332 | 2.082 | 2.337 | 2.362 | **0.87** | 0.96 | +0.09 | 14% |
| transport | 10 | 0.937 | 22.513 | 1.992 | 1.874 | 1.634 | 1.882 | 1.982 | **0.82** | 0.94 | +0.12 | 3% |
| release | 10 | 2.220 | 83.877 | 2.951 | 2.539 | 2.302 | 2.371 | 2.582 | **0.78** | 0.92 | +0.14 | 57% |
| all | 50 | 1.333 | 49.543 | 2.518 | 2.383 | 2.006 | 2.303 | 2.411 | **0.80** | 0.94 | +0.14 | 21% |

#### arm joints, raw units (degrees)

| phase | n | spread | scale_ref | mean_err | sample0 | oracle | meanchunk | consensus | oracle/mean | null | gain>null | consensus capture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| start | 10 | 1.737 | 68.006 | 3.193 | 3.041 | 2.249 | 2.966 | 3.109 | **0.70** | 0.95 | +0.24 | 9% |
| approach | 10 | 0.715 | 22.250 | 1.686 | 1.708 | 1.542 | 1.618 | 1.686 | **0.91** | 0.96 | +0.05 | -0% |
| grasp | 10 | 0.730 | 38.770 | 2.130 | 2.044 | 1.881 | 2.066 | 2.064 | **0.88** | 0.96 | +0.08 | 26% |
| transport | 10 | 0.829 | 19.351 | 1.797 | 1.857 | 1.603 | 1.681 | 1.738 | **0.89** | 0.96 | +0.07 | 31% |
| release | 10 | 1.408 | 80.268 | 2.279 | 2.165 | 1.859 | 2.026 | 2.092 | **0.82** | 0.94 | +0.12 | 44% |
| all | 50 | 1.084 | 45.729 | 2.217 | 2.163 | 1.827 | 2.072 | 2.138 | **0.82** | 0.95 | +0.13 | 20% |

#### per-joint std across samples, raw units

| phase | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---|---|---|---|---|---|
| start | 0.30 | 0.83 | 0.76 | 0.35 | 0.21 | 0.05 |
| approach | 0.19 | 0.30 | 0.26 | 0.16 | 0.13 | 0.34 |
| grasp | 0.21 | 0.31 | 0.26 | 0.16 | 0.10 | 0.14 |
| transport | 0.21 | 0.32 | 0.36 | 0.20 | 0.12 | 0.14 |
| release | 0.39 | 0.68 | 0.48 | 0.23 | 0.19 | 1.02 |
| all | 0.26 | 0.49 | 0.42 | 0.22 | 0.15 | 0.34 |


Plots: [`oracle_gain_by_phase.png`](out/sample_diversity/oracle_gain_by_phase.png),
[`spread_per_joint.png`](out/sample_diversity/spread_per_joint.png),
[`overlays.png`](out/sample_diversity/overlays.png). Per-observation data: `out/sample_diversity/metrics.json`.

![oracle gain by phase](out/sample_diversity/oracle_gain_by_phase.png)
![overlays](out/sample_diversity/overlays.png)

## Step C: noise-scale and denoising-step sweep (optimistic)

Same 50 observations and noise; all joints, all phases, normalized units.

| setting | spread | mean_err | oracle | consensus | mean chunk | oracle/mean | null | gain>null | consensus capture | mean-chunk capture | grasp oracle/mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x1.0, 4 steps (deployed) | 0.033 | 0.057 | 0.044 | 0.054 | 0.051 | **0.76** | 0.92 | +0.15 | 26% | 45% | 0.83 |
| x1.5, 4 steps | 0.052 | 0.065 | 0.045 | 0.061 | 0.053 | **0.70** | 0.89 | +0.20 | 22% | 58% | 0.74 |
| x2.0, 4 steps | 0.102 | 0.090 | 0.058 | 0.074 | 0.063 | **0.64** | 0.87 | +0.23 | 51% | 82% | 0.65 |
| x1.0, 2 steps | 0.027 | 0.057 | 0.048 | 0.055 | 0.053 | **0.84** | 0.94 | +0.10 | 20% | 45% | 0.90 |
| x1.0, 1 step | 0.036 | 0.062 | 0.057 | 0.061 | 0.057 | **0.92** | 0.95 | +0.03 | 23% | 106% | 0.94 |

Raising the noise trades every sample's error for spread: at x2 even the best of 8 is worse than the best of 8 at x1.
Fewer Euler steps shrink the headroom. No setting gives a better oracle than the deployed one.
Plot: [`sweep.png`](out/sample_diversity/sweep.png). Full tables per setting: `out/sample_diversity/summary.md`.

## Step D: latency on the Orin NX

50 timed runs after 5 warm-up, one real observation, ms median (p95):

| N | preprocess | backbone (once) | head (batched N) | postprocess | total | vs N = 1 |
|---|---|---|---|---|---|---|
| 1 | 1.4 (1.4) | 242.9 (244.0) | 104.1 (104.6) | 0.6 (0.7) | 349.3 (350.8) | 1.00x |
| 4 | 1.4 (1.4) | 243.1 (243.5) | 211.7 (212.2) | 0.7 (0.7) | 457.2 (458.1) | 1.31x |
| 8 | 1.4 (1.4) | 243.4 (243.7) | 397.2 (397.5) | 0.7 (0.7) | 643.1 (643.4) | 1.84x |

The deployed single-chunk path measures 350.4 ms, matching. The backbone is paid once; the float32 head dominates
the extra cost. Running the head in bf16 might roughly halve it (untested). A world-model rollout of 8 chunks would
add to this.

## Step E: real-robot baseline with logging (script only, not run)

[`robot_baseline.py`](robot_baseline.py) drives the gate harness (preset `vla_jepa`, rows in
`outputs/gate/trials.csv`) on the harder layouts listed in [`layouts_hard.csv`](layouts_hard.csv) (template to fill
with your ~20 layouts: trial, layout_id, description). Before each trial it prints the layout to set up. At every
decision it samples 8 chunks from one backbone pass, executes chunk 0 (plain VLA behaviour) and logs both camera
frames, the raw state, all 8 chunks (normalized and raw), timestamps and inference time. After a failure it prints
the trial's decision timeline and asks at which decision it failed. Per trial: layout, success, failure type
(missed grasp / dropped / wrong bowl / timeout / other), failure decision, number of decisions, episode length.
Outputs in `outputs/prechecks/robot_baseline/<session>/`.

```bash
IMAGE=lerobot-jetson:rc LEROBOT_SRC=image shared/gate/jetson_run.sh --python projects/vla-jepa-world-model/prechecks/robot_baseline.py \
    --layouts projects/vla-jepa-world-model/prechecks/layouts_hard.csv
```

Dry-run verified on the Jetson with a fake arm (`--fake-robot --trials 2 --duration 4`): prompts, layout
announcements, failure-decision question, `trials.jsonl`, frames, `decisions.json` and `chunks.npz`
`[decisions, 8, 30, 6]` all written. Timing caveat: sampling 8 chunks makes each decision ~643 ms instead of ~349 ms,
so the arm pauses longer at chunk boundaries than in the 2026-10-08 run; `--n 1` gives a timing-identical baseline.

## Step F: world-model preparation (no training)

**F.1 stable-worldmodel.** It exists: `stable-worldmodel` 0.1.1 on PyPI,
[galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), and it ships `LeWM`
(`stable_worldmodel/wm/lewm`). Its LeRobot connector (`data/formats/lerobot.py`, `LeRobotAdapter`, scheme
`lerobot://<repo_id>`) is read-only and expects: lerobot >= 0.5 on Python 3.12; a `primary_camera_key` when there
are several cameras (mapped to `pixels`); `action` -> `action`, `observation.state` -> `proprio`, other columns via
`key_aliases`; windowing with `frameskip` (stride between observations) and `num_steps`, the `frameskip` raw actions
between two observations being concatenated. **Loaded our dataset on the Jetson** with
`primary_camera_key="observation.images.front_top_camera"`, wrist camera aliased, `frameskip=6`, `num_steps=4`:
57,276 windows; `pixels` and `wrist` [4, 3, 480, 640] float, `proprio` [4, 6], `action` [4, 36]. Format issues:
actions and state are raw (un-normalized) and frames full size, so normalization and resizing are on the training
side; installing it next to lerobot pulled numpy 2.5 (lerobot needs < 2.3), so it needs its own environment or a
pinned numpy; importing it warns about missing Atari environments (harmless).

**F.2 temporal alignment (proposal).** World model at **5 fps** = `frameskip` 6 on the 30 fps data, so one
world-model step = **6 raw actions** (a 36-dim macro-action). One VLA chunk (30 actions, 1 s) = **5 world-model
steps**: scoring a chunk = rolling the world model 5 steps from the current frame and comparing the 5th latent with
the goal latent. A trial (median 11 s) is ~55 world-model steps, ~11 decisions. For a 2-level H-JEPA, level 2 at
every 2nd level-1 latent runs at 2.5 fps, so a chunk spans 2.5 level-2 steps; chunks of 24 or 36 actions would align
with level-2 steps.

**F.3 goal frames and layouts.** The last frame of every episode (both cameras, 320x240) is in
`out/layouts/goal_frames/`, indexed by `out/layouts/index.csv` (episode, task, layout, bowl positions, goal frame
paths). Layouts were not discrete: bowl positions in the first front frame are spread over the whole mat
([`scatter.png`](out/layouts/scatter.png)); fine clustering at 0.15 image widths gives 76 groups of 1-9 episodes.
The layout label is therefore a coarse grid, each bowl's cell being left / centre / right x top / bottom of the
front image (top = robot side), e.g. `B:left-top P:right-bottom`. 30 grid layouts; contact sheets (start frames over
goal frames) for the 20 with >= 3 episodes: `out/layouts/layout_<id>_<label>.jpg`. **Please confirm or relabel.**

| layout | label | episodes |
|---|---|---|
| 0 | B:right-bottom P:left-bottom | 19 |
| 1 | B:right-bottom P:left-top | 19 |
| 2 | B:left-top P:right-bottom | 16 |
| 3 | B:left-bottom P:right-bottom | 15 |
| 4 | B:right-top P:left-bottom | 11 |
| 5 | B:left-top P:centre-bottom | 10 |
| 6 | B:centre-bottom P:left-top | 9 |
| 7 | B:right-top P:left-top | 9 |
| 8 | B:right-top P:centre-bottom | 8 |
| 9 | B:centre-bottom P:right-top | 7 |
| 10 | B:left-bottom P:right-top | 7 |
| 11 | B:centre-top P:left-bottom | 6 |
| 12 | B:left-bottom P:centre-bottom | 6 |
| 13 | B:left-bottom P:left-top | 6 |
| 14 | B:centre-top P:left-top | 5 |
| 15 | B:centre-bottom P:centre-top | 4 |
| 16 | B:right-bottom P:right-top | 4 |
| 17 | B:left-bottom P:centre-top | 3 |
| 18 | B:left-top P:right-top | 3 |
| 19 | B:right-top P:right-bottom | 3 |
| 20 | B:centre-bottom P:left-bottom | 2 |
| 21 | B:right-bottom P:centre-top | 2 |
| 22 | B:centre-bottom P:right-bottom | 1 |
| 23 | B:centre-top P:centre-bottom | 1 |
| 24 | B:centre-top P:right-bottom | 1 |
| 25 | B:centre-top P:right-top | 1 |
| 26 | B:left-top P:centre-top | 1 |
| 27 | B:left-top P:left-bottom | 1 |
| 28 | B:right-bottom P:centre-bottom | 1 |
| 29 | B:right-top P:centre-top | 1 |

![bowl positions](out/layouts/scatter.png)

## Caveats

- Offline numbers are on training data (no validation split): optimistic. The ~20 new demos on harder layouts with a
  distractor will be the held-out set.
- Distance to one demonstration is not success: several chunks can succeed. Step B is a sanity check; step E's
  logs decide.
- The 90% baseline is 10 trials on one task, under changing sunlight.
- The phase heuristic relies on the gripper's open/close pattern; all 182 episodes follow it.

## Files

| file | what |
|---|---|
| `common.py` | policy loading, seeded head sampler, normalization, phase windows, distances |
| `inventory.py` -> `inventory.md` | step A |
| `sample_diversity.py` -> `out/sample_diversity/` | steps B and C (metrics.json, summary.md, plots) |
| `latency.py` -> `out/latency/` | step D |
| `robot_baseline.py`, `layouts_hard.csv` | step E (not run) |
| `layouts.py` -> `out/layouts/` | step F.3: layouts, goal frames, index |

All run with `IMAGE=lerobot-jetson:rc LEROBOT_SRC=image shared/gate/jetson_run.sh --python projects/vla-jepa-world-model/prechecks/<script>.py`.
