# TurboVLA on the SO-101 — everything in one place

Status, experiments, evidence, open problem, and how to run things, for the TurboVLA port
([`ravediamond/lerobot-policy-turbovla-so101`](https://github.com/ravediamond/lerobot-policy-turbovla-so101)) on a real
SO-101 arm. Every number is from an actual run unless marked as an estimate. Consolidated on 2026-10-01 from five earlier
docs (`TURBOVLA_SO101_WRITEUP.md`, `TURBOVLA_BOWLS_DIAGNOSIS.md`, `turbo-vla-findings.md`, `TURBOVLA_SO101_ROLLOUT.md`,
`TURBOVLA_SO101_TRAINING.md`; last versions at commit `356dd7c`). The TurboVLA paper itself stays in
[`2607.27205v1-turbo-vla.md`](2607.27205v1-turbo-vla.md).

**Reading order:** §0 is self-contained and current — start there (and give §0 to an agent first). §1–7 are the
experiment history in order, §8 the detailed evidence for the bowls task, §9–11 the practical reference.

| § | Content |
|---|---|
| 0 | Current state: facts, ruled out, open problem, next experiments, constraints, superseded explanations |
| 1 | Background: TurboVLA, the plugin, why a real robot |
| 2 | Experiment 1 (pick_cups, frozen backbones): proprioceptive shortcut |
| 3 | Methodological takeaway: the input-sensitivity gate |
| 4 | Cross-check against the TurboVLA paper |
| 5 | Checkpoint conversion (libero warm start) |
| 6 | Experiment 2 (pick_cups, unfrozen DINOv3): passes the gate |
| 7 | Experiment 3 (bowls, first real rollouts): never starts → task coin-flip → copycat: training barely needs the instruction |
| 8 | Experiment 3, detailed evidence and solution list |
| 9 | How to: train (HF Jobs), gate, run rollouts (M1 / Orin NX), hardware facts |
| 10 | Artifacts index (Hub repos, data folders, scripts) |
| 11 | Remaining deliverables and the #4400 status message |

---

## 0. Current state — read this first

> **Correction / confirmation, 2026-10-23 (NEXT_ANGLES §15).** The port differed from upstream in four places (Grounding DINO / LIBERO init of the fusion, the bi-attention module, where the state token enters, decoder and projection details; confirmed in [`FAITHFUL_PROTOCOL.md`](FAITHFUL_PROTOCOL.md)). Upstream's own code, initialised from its LIBERO checkpoint (666/672 tensors) and trained on this data, gives **task swap 0.020°, per-layout 39 %**, while the released checkpoint shows 0.105 (≈ 5.5°) on LIBERO-Goal. Pre-registered verdict: **fact 7 is the full story** — the paper's recipe ignores the instruction on this data too. **The data claim stands; the port claim is withdrawn**: the divergences are real but did not change instruction use (TurboVLA's own Table 3, LIBERO-Goal 97.4 → 11.6 % without language, shows the architecture reads language when the data forces it). Details: [`data/2026-10-22_faithful/README.md`](data/2026-10-22_faithful/README.md).

**Status (2026-10-08, research closed).** No TurboVLA checkpoint is deployable, and the TurboVLA line of work stops here.
On the two-task bowls dataset every variant either ignores the instruction or encodes it without acting on it. The last
attempt, a fresh run with the paper's own optimizer recipe + EMA (`turbovla_so101_bowls_v2`, fact 8), went 0/3 on the
arm with the same failure. On the same task and rig, ACT (single-task, 10/10 M1, 9/10 Orin NX), SmolVLA (two-task, 9/10
on both) and VLA-JEPA (9/10 Orin NX) work. Results are published as they stand; the untested hypotheses below (H1–H4)
are left as the hand-off for anyone picking this up.

**Setup.** SO-101 follower + 2 cameras (C920 front, icSpring wrist); M1 Mac (MPS) and Jetson Orin NX (CUDA, Docker
`lerobot-jetson:public`); lerobot `main` @ `240ea44cf`; plugin @ `daaff77` (policy type `turbovla_so101`: DINOv3 ViT-B +
BERT-base, 6 bidirectional vision↔text fusion layers, 4-layer action decoder with 12 learned queries, L1 loss, ~209M
params). Dataset `ravediamond/so101_bowls_all`: 182 teleop episodes, "Put the blue bowl in the pink bowl." /
"Put the pink bowl in the blue bowl.", both bowl layouts for both tasks (36/54 and 36/56), so only the instruction
disambiguates the task.

**Measured facts.**
1. *Idle start (solved).* Demos idle ~1.6 s at rest; a 12-step (0.4 s) chunk makes "stay" the target at 99% of episode
   starts → the first checkpoint never left the rest pose. Trimming idle starts fixed it (episode-start reach 1.3° → 24°).
2. *Instruction ignored.* Task-swap (blue↔pink instruction on the same frame) 0.1–0.25° for every TurboVLA variant vs
   7–13° for SmolVLA; per-layout probe (does the left-bowl instruction steer further left) 42–64% (chance) vs 86% SmolVLA.
   On `pick_cups` (Exp. 2, one-word-different instructions, 3 tasks) TurboVLA reached 4.30°.
3. *On the arm:* task coin-flip and hesitation — 1/10 correct tries with the trimmed model, 1/4 with the single-task
   fine-tune (C), 0/2 with the relabelled one (A); hovering between the bowls, gripper chatter. Failed tries have
   plan-to-plan jumps of 1.8–4.7° vs 0.3–0.4° in completed ones (mode averaging of an L1 head between two tasks).
4. *The text path is alive.* Decoder cross-attention puts 7–21% of its mass on text (3–10× uniform); text tokens get
   24–33× the per-token gradient of image patches; the colour token differs 46–63% between instructions after
   `language_proj`.
5. *The instruction is washed out before the actions.* In fusion layers 2–5 the instruction-specific part of the
   text→vision update is 1–4%; the action chunk differs ~0.5% between instructions.
6. *An explicit task loss is learned but not used.* An auxiliary "which bowl" head on the decoder's final states (what
   `action_head` reads) reaches ≥ 98% accuracy within 1k steps, yet task-swap peaks at 2.3° (10k steps) and decays;
   per-layout converges to the control's (chance).

7. *Sanity check on training frames (the decisive one).* On paired pre-grasp training frames, swapping the instruction
   raises the L1 loss by only 0–4% (run 1 +4.3%, control +1.5%); the predicted blue−pink difference is 1–2% of the true
   target difference. Why: in the demos the side decision (next-0.4 s pan move of 7° towards the target bowl) happens
   only in the 0.1–0.3 band of pre-grasp progress (~20% of pre-grasp frames, ~6% of all frames); before it the target is
   the same for both tasks (lift), after it the arm's own pose already points to the target bowl (pan turned 14–18°).
   The 0.4 s L1 target is predictable from the state almost everywhere, so training barely rewards reading the
   instruction: a **copycat / causal-confusion** solution (continue the arm's current direction). On the arm, the
   direction at the decision point is set by small motions → coin flip, hesitation when they waver.

8. *The paper's optimizer recipe doesn't change it (2026-10-08).* `turbovla_so101_bowls_v2`: fresh 35k steps from the
   libero warm start on the untrimmed data, chunk 12, AdamW betas (0.9, 0.95), grad clip 1.0, wd 1e-10 (plugin v0.2.0
   fixed the defaults that had silently differed from the paper's Table 8) and EMA 0.999. Offline it matches the
   original untrimmed checkpoint on every probe (start reach 1.3°, task-swap 0.19°, per-layout 47%; EMA identical to
   ±0.2°). On the arm 0/3: from the rig's live rest pose it does move (live reach 27–28°, so the offline start probe
   was pessimistic here), but it went **right in all three trials whatever bowl was there** — swapping the bowls didn't
   change the side — and hovered at the rim with an open gripper for the full 25 s, never grasping. A position prior,
   not the instruction or the colour, picks the side. Data: `projects/turbovla/data/2026-10-08_gate/`,
   `projects/turbovla/data/2026-10-08_orin_nx/`.

**Ruled out (with evidence in §7–8).** Rollout pipeline and inference path (normalisation, scaling, colour order, task
plumbing, camera order); dataset labels (1 bad episode of 182, fixed); bowl layout ↔ task correlation; camera mapping;
calibration; latency; a dead text path (fact 4); training order / fine-tuning from a checkpoint that ignored text (a
fresh start from the libero warm start: task-swap 0.5–0.7°); relabelling to distinct words (0.18° → 0.25°); paired
batches (doubled the instruction-specific fusion share, still ~3%).

**Superseded explanations — do not reuse.** (a) "The two instructions are the same tokens reordered, so set-like
attention can't tell them apart" — disproved: distinct wording changed nothing, and the colour token differs strongly at
the input. (b) "The text path is dead" — disproved by fact 4. (c) "Training from a fresh trunk will fix it" —
disproved by the fresh control run. (d) "The action head can't read a task signal that's in its input" (the
"readout" framing of §7.6) — superseded by fact 7: the signal is readable, training just almost never needs it.

**Open problem.** The task is represented at three successive levels — text tokens, fused tokens (layer 0), decoder
states (98% linearly decodable) — but the actions don't depend on it. Fact 7 explains why: with a 12-step (0.4 s) L1
target and the state as input, the instruction is needed on only ~6% of training frames and partly leaked by the arm's
pose even there, so the cheapest fit continues the arm's own motion. The open question is now practical: which change
makes the instruction necessary on enough frames?

**Hypotheses to test.** (H4, leading) *Copycat:* a longer action horizon (50 steps, as SmolVLA; 100, as ACT) puts the
side decision into the targets of most early frames, and random dropout of the state input removes the leak — either
should raise task-swap. (H1) *Readout:* the action head's gradient for the task direction is weak relative to the
dominant visual/state directions; making the action computation conditional by construction would fix it. (H2)
*Objective:* L1 regression on a bimodal target rewards hedging between the two modes; a multimodal head (flow matching /
diffusion, or a CVAE latent as in ACT) would separate them. (H3) *Fusion:* averaging over 8–11 text tokens dilutes the
1–2 contrastive tokens; a conditioning path that concentrates them (learned attention pooling + FiLM) would help — but
fact 6 says the decoder already has the task, so H3 alone is unlikely to be enough.

**Candidate next experiments (cheapest first).** A 50-step-horizon run from the resized libero init
(`projects/turbovla/scripts/resize_turbovla_chunk.py`, init built and smoke-tested) or state dropout in the training script (tests H4);
per-task action heads (two linear heads selected by the instruction —
a minimal test of H1); FiLM of the decoder states by the attention-pooled task representation; a flow-matching action
head (H2); for the TurboVLA-vs-ACT comparison, a from-scratch single-task run (ACT's protocol). Gate every candidate
offline before arm time (§9: start reach ≥ 10°, task-swap ≥ 5°, per-layout ≥ 75%), and run
`projects/turbovla/scripts/text_path_diagnostic.py` to see where the signal dies.

**Constraints.** HF Jobs credit ~$3.3 left; t4-small ≈ $0.07 per 1k steps is the cheapest per step (35k steps ≈ 6.3 h ≈
$2.5). Any new config field breaks loading on the Jetson's plugin (pinned in `jetson_setup.sh`, now v0.2.0 `2d08dc3`; strict config parsing) — training-time
changes must live in `projects/turbovla/scripts/train_turbovla_paired.py` (extra weights are fine: `strict=False` loading); architecture
changes need a plugin branch. Two-machine rig (M1, Orin NX); the arm is shared with other work.

---

## 1. Background — TurboVLA, the plugin, why a real robot

### Motivation

[TurboVLA](https://github.com/H-EmbodVis/TurboVLA) is a lightweight
vision-language-action (VLA) model: a DINOv3 vision encoder + BERT text encoder
feeding a small fusion trunk, rather than a full multi-billion-parameter VLM
backbone (the approach used by e.g. SmolVLA, pi0, GR00T). The paper reports 97.7%
on LIBERO at ~0.2B total parameters and ~31ms inference on an RTX 4090 — the pitch
is that a full VLM backbone may be unnecessary overhead for many manipulation
tasks, and a much smaller, purpose-built architecture can compete.

This is a genuinely interesting empirical question with no clean answer yet:
**does the "small paired encoders, no VLM" architecture actually work on a real
robot, on real hardware, or does it only work in simulation benchmarks?**
LIBERO/RoboTwin sim results don't answer that. This work is an attempt to get a
real answer, end to end: port the architecture into a real robotics stack
(LeRobot), train it on real teleop data, run it on a real robot arm, and report
what actually happens — including a real failure, diagnosed rather than hidden.

### What was built

**[`lerobot-policy-turbovla-so101`](https://github.com/ravediamond/lerobot-policy-turbovla-so101)**
— a third-party LeRobot policy plugin (installable via
`pip install lerobot-policy-turbovla-so101`, registers as `policy.type=turbovla_so101`
through LeRobot's `register_third_party_plugins()` mechanism, no core-repo PR
required). Built on top of
[`ez1540/turbovla-lerobot-plugin`](https://github.com/ez1540/turbovla-lerobot-plugin)
(the original LeRobot port of the architecture, Apache-2.0, credited in this repo's
README/NOTICE — not a GitHub fork, since the intent from the start was to add
something the original didn't have).

What this fork adds beyond the original port:
- **Temporal ensembling** (ACT's Algorithm 2 — exponential blending of overlapping
  chunk predictions across timesteps). Chosen deliberately over Real-Time Chunking
  (RTC, used by SmolVLA/pi0): RTC corrects an *iterative flow-matching* denoising
  trajectory mid-generation, but TurboVLA's decoder — like ACT's — predicts a
  whole action chunk in a single parallel forward pass with nothing to correct
  mid-trajectory. Temporal ensembling is the technique that actually applies to
  this architecture's inference pattern.
- CI (ruff + pytest on Python 3.12/3.13) and a PyPI release pipeline (Trusted
  Publishing, no stored token).
- **Real hardware validation** — the actual differentiator this writeup is about.
  The original port shipped with zero hardware validation and placeholder results.

Model size confirmed empirically (not just from the paper): **209,286,150 total
parameters**, of which only 14,143,494 (~14M, ~6.8%) were trainable in the first
training run below — the rest is the frozen DINOv3 + BERT backbones.

#### Infrastructure validation before any training (2026-08-31)

Before training anything, the plugin's inference plumbing was smoke-tested on real
Jetson Orin hardware (part of a parallel, separate infra effort —
[`ravediamond/lerobot-jetson`](https://github.com/ravediamond/lerobot-jetson), a
Docker image providing a working CUDA-enabled torch/torchcodec build for Jetson,
since no official or community channel ships one). Result: the plugin installs and
runs correctly on real GPU hardware, `select_action()` returns a valid action
tensor, **steady-state inference latency ~4.0ms** (median 3.9ms over 10 calls,
after ~400-500ms cold-start/CUDA-JIT warmup) — a genuinely fast number, consistent
with the architecture's efficiency claim, at least for raw forward-pass speed.
(This was a random-init smoke test, not the trained checkpoint — included here
only to establish that the inference path itself is fast and correct before any
training-quality questions come into play.)

### Project history before the first rollout (from the 2026-09 rollout handoff)

- Originated from [huggingface/lerobot#4400](https://github.com/huggingface/lerobot/issues/4400), closed by maintainer
  `pkooij` ("plugin approach is right for now, thanks"); real-rollout results were **explicitly promised there** — the
  final step is a comment on that closed issue (§11).
- 2026-08-21: plugin built and published (CI green on Python 3.12/3.13, PyPI live, test suite passing, temporal
  ensembling added).
- 2026-08-31: dry run on the Jetson Orin — `register_third_party_plugins()` discovers the plugin, forward pass works,
  ~4 ms steady-state inference. Bug found: `load_pretrained_backbones=False` still fetches the gated
  `facebook/dinov3-vitb16-pretrain-lvd1689m` config; worked around with `facebook/dinov2-base` for that dry run only.
- The handoff's original rollout command (`--robot.port=/dev/ttyACM0`, cameras `up` / `side`) was **wrong for this
  rig** and is superseded by §9.

## 2. Experiment 1 — pick_cups, frozen backbones: a proprioceptive shortcut

### Experiment 1 — training run

- **Dataset**: [`ravediamond/pick_cups_all`](https://huggingface.co/datasets/ravediamond/pick_cups_all)
  — 180 teleop episodes, 3 differently-colored cups, language-conditioned task
  strings ("Pick up the blue cup" / "Pick up the rose cup" / "Pick the grey cup").
  Chosen specifically because TurboVLA (unlike ACT, used in an earlier,
  architecturally different demo — huggingface/lerobot#4241's chunk-safety work)
  takes language input, so a multi-object/multi-instruction dataset is directly
  usable here.
- **Config**: `policy.type=turbovla_so101`, `freeze_vision_backbone=True`,
  `freeze_language_backbone=True` (both defaults, not overridden), 20,000 steps,
  batch size 8, single T4 GPU (via HF Jobs, `t4-small` flavor).
- **Compute**: 41,342 training frames, 697 epochs over the dataset by the end.
  ~4.0 steps/sec throughout. **Total training time: 1h 23m.** Only 1.34GB GPU
  memory used at any point (T4's 16GB was far more than needed for this config).
- **Result**: final training loss (L1, normalized) **0.098**, smooth monotonic
  decrease the whole run (0.834 → 0.751 → 0.633 → 0.137 → 0.132 → 0.121 → 0.109 →
  0.100 → 0.098 across the run), no divergence or instability at any point.
- **Checkpoint published**: [`ravediamond/turbovla_so101_pick_cups`](https://huggingface.co/ravediamond/turbovla_so101_pick_cups),
  public, verified byte-for-byte intact (837MB `model.safetensors`, HTTP 200 on a
  full download, matching the Hub API's reported storage size — this was
  double-checked because the job's own dashboard status showed `ERROR`, caused by
  a job-cleanup timeout that fired *after* training and the Hub push both
  completed cleanly, confirmed from the job's own log ordering).

**At this point, by every training-time signal available, the run looked
successful**: loss went down smoothly, no NaNs, no crashes, checkpoint saved and
pushed intact. This is exactly the kind of result that would normally be reported
as "training complete, ready to evaluate."

### The rollout — what actually happened on hardware

Real SO-101 rollout, on a Mac with the arm physically connected (Apple Silicon,
MPS backend — not the eventual Jetson deployment target, but the diagnostic below
doesn't depend on which device runs the forward pass).

**The arm did not move meaningfully.** Small twitches only — no reaching motion
toward any cup. Camera feeds were confirmed live and correctly framed (overhead
view of the three cups, a wrist close-up). State readings were sane. The
observation/action traces in a live visualization were essentially flat.

### The diagnostic

A standalone script was written to reproduce LeRobot's exact rollout inference
path (`build_dataset_frame` → `prepare_observation_for_inference` → the policy's
own pre/post-processor pipeline → `predict_action_chunk`) outside the interactive
rollout loop, so individual inputs could be probed in isolation. It:

1. Loaded the checkpoint and connected to the real robot, grabbed one real
   observation, then immediately disabled arm torque (so nothing moves for the
   rest of the diagnostic — this is a pure inference probe, not a live rollout).
2. Ran inference on that one observation three ways: the real task string, a
   *different* task string (different target cup, opposite side of the mat), and
   with both camera images zeroed out entirely (black frames).

#### Verified correct (ruled out as the bug)

| Check | Result |
|---|---|
| State normalization | Raw degrees → `MEAN_STD` normalized to ≈[−1.4, 1.4] — correct |
| Image scaling | uint8 [0,255] → float [0,1] — correct |
| Camera key/order matching | Matches training config exactly (`wrist_camera`, `front_top_camera`) |
| Task string reaching the model | Confirmed present in the batch passed to the policy |
| Temporal ensembling | Confirmed off (`temporal_ensemble_coeff=None`) — not a smoothing artifact |
| Control-loop rate | ~6-7Hz observed on MPS vs 30Hz target — real, but a separate latency issue, not the cause of the frozen output (jerky ≠ frozen) |

#### The actual finding

The model's predicted 12-step action chunk barely deviates from the robot's
current pose — total motion across the entire predicted chunk was
**`[-0.08, 1.34, -0.59, -0.43, -0.15, 0.07]` degrees** across the six joints. For
comparison, the training dataset's own action distribution has a standard
deviation of up to **40.9 degrees** on one joint — so the demonstrations
themselves contain large, real motions; the *model* just isn't predicting any.

**Input-sensitivity probe** (same one real observation, only one input changed
each time):

| Perturbation | Max change in predicted action (degrees) |
|---|---|
| Swap task string to a different target cup (opposite side of the mat) | **0.48°** |
| Zero out both camera images entirely (black frames) | **1.92°** |
| Natural variation across the predicted chunk's own 12 steps | ~1.6° (worst joint) |

For reference, the natural variation *within the model's own uncontested output*
(across the 12 predicted timesteps) is roughly the same magnitude as what happens
when you delete all visual input or swap the language target. **The model is
statistically indifferent to whether it can see anything or has been told which
cup to pick.**

### Diagnosis — a proprioceptive shortcut, not a pipeline bug

**The checkpoint learned `action ≈ state_t` — copy the current joint pose through
almost unchanged — rather than a policy conditioned on vision and language.**

Why this happened, reasoned from the actual config and data:

- In `pick_cups_all`, the **action distribution is nearly identical to the state
  distribution** (absolute joint-position targets from slow, careful teleoperation
  — action mean `[1.9, -42.3, 50.1, 66.0, -2.7, 4.1]` vs. state mean
  `[1.8, -43.0, 52.5, 66.2, -2.5, 4.6]`, essentially overlapping).
- TurboVLA's architecture prepends a learned state token alongside the vision
  patch tokens and text tokens into its fusion trunk.
- With `freeze_vision_backbone=True` and `freeze_language_backbone=True`, only the
  small fusion trunk (`dim_model=256`) and a few projection layers are trainable
  — 14M of 209M parameters.
- A model that simply copies the state token through to the output achieves a low
  L1 loss (**exactly the 0.098 observed**) on this dataset **without ever learning
  to use the frozen vision/language features for control**, because the loss
  landscape doesn't distinguish "predicted the demonstrated action" from
  "predicted something numerically close to the current state" when those two
  targets are nearly the same thing in the data.
- The frozen backbones do produce real features — but the trainable trunk was
  never forced to learn how to use them, because the shortcut was cheaper and
  available from the very first gradient step.

**This is a known failure mode in imitation learning generally** (shortcut
learning / proprioceptive leakage), but it's worth stating plainly here because
**the loss curve gave no indication anything was wrong** — smooth, monotonic,
textbook-looking convergence, on a checkpoint that turned out to have not learned
the task at all. Loss-going-down is necessary but nowhere near sufficient
evidence that a policy learned to use its sensory inputs.

## 3. The methodological takeaway (candidate for the blog post's actual thesis)

**A trained policy should pass an input-sensitivity gate before its loss curve is
trusted as evidence of anything.** Concretely: after training, run inference on a
held-out observation with (a) real inputs, (b) camera images zeroed, and (c) the
language instruction changed to a different valid target. If the predicted action
doesn't change meaningfully between these, the model has not learned to condition
on vision or language, regardless of how good the loss number looks. This is a
five-minute check that would have caught this specific failure before any
hardware time was spent on it, and it generalizes: any VLA-style architecture
with a proprioceptive state input and an action space close to the state space
is at risk of this exact shortcut, not just TurboVLA specifically.

## 4. Cross-checking against the TurboVLA paper itself

The [TurboVLA paper](https://arxiv.org/abs/2607.27205) (`2607.27205v1-turbo-vla.md`
in this repo) was read after the diagnosis above, specifically to check whether the
paper's own reported training recipe and ablations are consistent with what was
observed here — not to re-derive the diagnosis, but to sanity-check it against the
authors' own numbers and to find any concrete gaps worth closing before Experiment
2.

**1) The paper's own ablation independently confirms the failure mode.** Table 3
("Effect of language conditioning") removes language entirely from a correctly
trained TurboVLA and reports average LIBERO success dropping from 97.7% to 70.8%,
with LIBERO-Goal specifically collapsing from 97.4% to 11.6% — because that suite
has multiple valid behaviors compatible with the same scene, and only language
disambiguates them. This is the paper's own proof that language conditioning is
supposed to produce *dramatic* behavioral differences when the architecture is
working. The 0.48° change observed here from swapping the target cup isn't just
"lower than expected" — it's off by orders of magnitude from what the architecture
is designed to do, independent of anything about this specific SO-101 checkpoint.
Strong external confirmation of §2's diagnosis, not just consistent with it.

**2) Training-scale gap, worth naming honestly.** The paper's LIBERO recipe:
**80,000 steps, effective batch size 256, four RTX 4090 GPUs** (§2.2/5.3).
Experiment 1 here: 20,000 steps, batch size 8, a single T4. That's 4x fewer steps
and 32x smaller batch — a real difference in optimization scale, independent of
the freezing question, and worth stating as a contributing factor rather than
omitting it.

**3) The more important finding — the paper's real-world results don't train from
scratch.** §2.2's real-world AgileX Piper experiments **initialize from a
checkpoint already pretrained on LIBERO**, then fine-tune on only **4×65 = 260 real
teleoperated demonstrations for 12,500 steps** — reaching 80-92.5% success across
four tasks. That data/step scale is comparable to Experiment 1 here (180 episodes,
20,000 steps), but the paper's recipe starts from a policy that already learned
real vision-language grounding during LIBERO pretraining; Experiment 1 trained
end-to-end from scratch on real SO-101 data alone, with no equivalent pretraining
stage. This is a stronger argument than the frozen-backbone theory alone against
"just collect more real data" — the paper's own demonstrated path to real-world
success on a similarly small amount of real data specifically depends on starting
from a pretrained backbone, not on data volume by itself.

This connects to an already-known, previously deprioritized gap: weight conversion
from the original authors' released checkpoints
([`H-EmbodVis/TurboVLA`](https://github.com/H-EmbodVis/TurboVLA)) into this
plugin's LeRobot format was flagged as unsolved when the plugin was first built
(the SO-101 port trains from scratch via `lerobot-train` instead, same as the
upstream port it's based on) — deprioritized at the time in favor of shipping a
working plugin fast. This cross-check suggests that gap may now be higher-leverage
than the freeze/unfreeze fix on its own.

## 5. Checkpoint conversion — done, verified (2026-09-08)

Investigated (1) below directly rather than leaving it as an open question. Two
real checkpoints exist in
[`H-EmbodVis/TurboVLA`](https://huggingface.co/H-EmbodVis/TurboVLA) on the Hub:
`checkpoints/libero/turbovla_libero.pth` (864MB, sim-pretrained — what the paper's
real-world fine-tune started from) and a separate RoboTwin bimanual checkpoint
(wrong action space for a single-arm robot, not used here).

**The checkpoint's own embedded `model_config` confirms, for this exact
checkpoint** (not inferred from the RoboTwin config as a proxy): `vision.frozen:
False`, `text.frozen: True` — the original authors trained with the vision
backbone unfrozen, only BERT frozen. Directly confirms the freeze-policy
divergence suspected in §4.

**Key-name and shape comparison, done by actually loading both state dicts:**
- **`vision_encoder` (211 keys, the DINOv3 backbone) transfers cleanly** — same
  parameter count, same shapes, only a naming-prefix difference
  (`vision_encoder.backbone.{embeddings,layer.N}.*` in the source vs.
  `vision_encoder.{embeddings,model.layer.N}.*` in this plugin). A mechanical
  remap, not an architecture mismatch.
- `text_encoder.bert.*` matches standard HF BERT naming too, but since BERT was
  frozen in both the source run and this plugin's defaults, it's stock
  `bert-base-uncased` either way — nothing to gain by transferring it.
- `vision_projection`, `vision_language_interaction`, `action_head` are
  **structurally different modules** between the original H-EmbodVis codebase and
  this LeRobot port (different projector/attention/head implementations, not just
  different names — e.g. the source's `vision_projection` is a norm+MLP block,
  this plugin's `vision_proj` is a single `nn.Linear`). These can't be transferred
  as-is; they train from scratch either way, same as Experiment 1.
- Action/state dims also differ (source: 7-D action / 8-D state, LIBERO's
  7-DoF-arm+gripper convention; SO-101: 6-D both) — expected, another reason the
  action head can't transfer directly regardless of module-shape differences.

**Conversion script written and run**: `convert_turbovla_checkpoint.py` in this
repo. Builds a fresh SO-101-shaped `TurboVLAPolicy` (stock DINOv3/BERT init,
`freeze_vision_backbone=False`), overwrites just the `vision_encoder` submodule
with the remapped, VLA-fine-tuned weights from the source checkpoint, verified
zero missing / zero unexpected keys on load. Saved and **pushed to
[`ravediamond/turbovla_so101_dinov3_libero_init`](https://huggingface.co/ravediamond/turbovla_so101_dinov3_libero_init)**
— usable directly as `--policy.pretrained_path` for a warm-started retrain.

## 6. Experiment 2 — pick_cups, libero warm start, unfrozen DINOv3: passes the gate

### What's next (Experiment 2, not yet run)

Revised priority order, folding in both the paper cross-check and the conversion
result above:

1. **Retrain from the warm-started checkpoint**
   (`ravediamond/turbovla_so101_dinov3_libero_init`) rather than from scratch —
   the vision encoder now starts from VLA-fine-tuned weights instead of generic
   ImageNet-pretrained DINOv3, matching the paper's own recipe for this
   component specifically.
2. **Keep the vision backbone unfrozen** (`freeze_vision_backbone=false`) —
   removes the shortcut's advantage regardless of the warm start, since the trunk
   now has to actually make the features earn their keep. Use a low learning rate
   for the backbone specifically (the config already separates
   `optimizer_lr_backbone`, default 10x lower than the main LR).
3. **Train longer, closer to the paper's own scale where feasible.** The paper
   used 80k steps / batch 256 / 4 GPUs for LIBERO — not directly reproducible on a
   single T4, but 20k steps / batch 8 is a real underscale by comparison. Estimate
   (not yet run): 30,000-40,000 steps, 2.5-4 hours on a T4, less on a stronger GPU
   given added backward-pass compute from the unfrozen backbone.
4. **Still not "more data" as the first lever**, per §4 above — the
   paper's own real-world recipe getting strong results from a comparable data
   scale (260 demos) argues against data volume being the primary bottleneck here.
   Only revisit dataset size/diversity if a checkpoint that's both
   pretrained-initialized and unfrozen still shows the same shortcut.
5. **Re-run the vision-sensitivity gate from §2 as a hard pass/fail check**
   before trusting the next checkpoint enough to spend more hardware time on a
   physical rollout.
6. Only after passing that gate: real rollout (ideally on the Jetson — the
   intended deployment target, and where the ~4ms latency number from §1
   is actually relevant), real success-rate numbers over multiple trials, fill in
   the plugin README's `Results` section, and report back on the originating
   discussion, [huggingface/lerobot#4400](https://github.com/huggingface/lerobot/issues/4400)
   (closed, but rollout results were explicitly promised there).

### Experiment 2 — run, and it passes the sensitivity gate (2026-09-08/09)

Launched on HF Jobs (T4), from
`ravediamond/turbovla_so101_dinov3_libero_init`, `freeze_vision_backbone=false`,
35,000 steps, batch 8. One real bug hit and fixed along the way: the conversion
script's first output only saved model weights, not the pre/post-processor
pipeline files — `--policy.pretrained_path` needs both, since it loads the full
policy state including normalization config, not just raw tensors. Fixed by also
building and saving `make_pre_post_processors(cfg)`'s default pipeline alongside
the model (verified safe: `lerobot-train` overrides normalizer stats with the
real dataset's stats regardless of what's saved in the pretrained processor,
whenever not resuming from a checkpoint — so an unfitted placeholder normalizer
in the warm-start checkpoint doesn't silently break normalization).

**Training stats**: 1.54-1.58 steps/sec throughout (vs. Experiment 1's 4.0
steps/sec — real backward-pass cost from the now-unfrozen DINOv3 ViT-B, larger
slowdown than the ~20-30% initially guessed). Peak memory 3.99GB (vs. 1.34GB in
Experiment 1), still comfortable on a T4's 16GB. Total wall time ~6h17m (vs.
Experiment 1's 1h23m). Completed all 35,000 steps and pushed to
[`ravediamond/turbovla_so101_pick_cups_v2`](https://huggingface.co/ravediamond/turbovla_so101_pick_cups_v2)
— confirmed via the Hub push timestamp landing ~2h20m after the last locally-polled
log line, same "real completion, cosmetic `Job timeout` during cleanup" pattern as
Experiment 1 (§2), not re-litigated in detail here since it's the identical
non-issue.

**Sensitivity gate, re-run without the physical robot** (`sensitivity_gate_v2.py`
— pulls one real observation from the dataset instead of connecting to the SO-101,
same methodology as §2's diagnostic otherwise):

| Perturbation | Experiment 1 (broken) | Experiment 2 |
|---|---|---|
| Swap task string to a different valid target | 0.48° | **4.30°** (~9x) |
| Zero both camera images | 1.92° | **20.42°** (~11x) |
| Natural motion within one predicted chunk | ~1.6° (worst joint) | up to **5.7°** |

A clear, large jump on every axis — the model is now responsive to both vision and
language, the exact thing that was missing in Experiment 1. This alone doesn't
prove correctness (a model can be input-sensitive and still wrong), so a second,
independent check was run.

**Ground-truth comparison, also dataset-only, no robot** (`ground_truth_compare_v2.py`
— compares the model's predicted 12-step action chunk at a real training-episode
frame against the actual human-teleoperated actions recorded at those same
timesteps): per-step error stayed in the **1.2-1.9° range** (declining toward the
end of the chunk), against a ground-truth trajectory whose own motion on the
dominant joint was **6.42°** across the same chunk — error at roughly 20-30% of
the real motion's magnitude, not comparable to or exceeding it. The model isn't
just reacting to inputs; what it predicts tracks a real recorded demonstration
reasonably closely.

**Both checks pass. Caveat stated plainly, not glossed over**: this frame is
in-distribution (part of the training set) — a good match here is a legitimate
sanity check that the model learned to imitate plausibly, not proof it
generalizes to new scenes, and it is not a substitute for a real physical
rollout. Per §6's plan, the actual remaining step is the SO-101 rollout
itself (success rate + latency, ideally on the Jetson), which is what would
finally let the plugin README's `Results` placeholder be filled with a real
number and the promised comment posted on #4400.

## 7. Experiment 3 — bowls, first real rollouts (2026-09-29 → 10-01)

The first time a TurboVLA checkpoint drove the real arm through full trials. Full detail, evidence tables and the
solution list: §8; raw trial data:
[`projects/turbovla/data/2026-10-01_orin_nx/`](data/2026-10-01_orin_nx/).

**Setup.** New dataset `so101_bowls_all` (182 episodes, 2 tasks: "Put the blue bowl in the pink bowl." /
"Put the pink bowl in the blue bowl."), the same one SmolVLA was trained on; ACT used only the blue → pink half.
`turbovla_so101_bowls`: Experiment 2's recipe (libero warm start, unfrozen DINOv3, frozen BERT, 35k steps, batch 8,
chunk 12). Same rig and harness as the ACT/SmolVLA gate runs, where ACT went 10/10 (M1) / 9/10 (Orin) and SmolVLA
9/10 on both ([`shared/gate/data/`](../../shared/gate/data/)).

**7.1 It passed the Experiment 2 gate and still never moved.** Offline it used vision (black cameras → 16°) and
tracked the demos within ~1° mid-episode. On the arm it stayed at the rest pose for 25 s (commanded joints moved
< 1.3°). Cause: the demonstrator idles ~1.6 s at rest before moving; a 12-step chunk looks 0.4 s ahead, so at rest
the correct target is "stay" — 99% of episode starts teach it. The model learned the data exactly; the gate never
looked at a rest frame. Fix: trim each episode to start 0.1 s before the first move
([`trim_idle_start.py`](../../shared/tools/trim_idle_start.py), 8,435 of 69,897 frames) and fine-tune 5k steps →
`turbovla_so101_bowls_trimmed`; a new **episode-start probe** in the gate went 1.3° → 24° and the arm now always
starts. (A 50-step chunk alone would not have fixed it: 42% of starts still "stay".)

**7.2 Then it did the wrong task half the time, and hesitated.** 10 exploratory tries on the Orin NX: 1 correct
pick-and-place, 3 with the task reversed (pink into blue), 1 missed grasp, the rest hovering, chattering the gripper,
or wavering between the two bowls. Two measured causes:

- **It ignores the instruction (task-swap 0.2°), and here it has to use it**: both bowl layouts occur equally for both
  tasks, so the scene does not give the task away. First hypothesis: the two instructions are the same 11 BERT tokens
  reordered (set similarity 0.986), invisible to set-like attention — contrast: on `pick_cups` the instructions
  differed by one word and task-swap was 4.30°. **The text-path diagnostic (§7.5) later showed this is not the
  bottleneck**: the colour token differs by 46–63% between instructions after the projection, and the difference is
  lost inside the fusion.
- **Mode averaging.** The L1 action head, facing two equally likely futures, predicts between them; re-planning every
  0.4 s it flips. On the arm: the two completed tries had smooth plan hand-overs (0.3–0.4° jump), every failed try
  jumped 6–10× more (1.8–4.7°).

**7.3 Fixes tried the same day** (HF Jobs, t4-small, 2 h cap each, ~1.2 h of training each):

| Variant | What | Offline gate | Arm (Orin NX) |
|---|---|---|---|
| C `turbovla_so101_bowls_blue_in_pink` | single task: fine-tune on the 90 blue → pink episodes (ACT's data) | start 24° ✓; tracks blue demos 1.1° vs pink 3.0° | **1/4** gate trials: still went for pink twice, hesitated once |
| A `turbovla_so101_bowls_relabeled_paired` | relabel to "Pick up the blue/pink bowl." (different tokens), fix mislabelled ep 108, **paired batches** (same layout + progress, both tasks per batch) | task-swap **0.25°** ✗, per-layout 64% ✗ | 0/2 (explore): grasped *between* the bowls |
| B `turbovla_so101_bowls_b_textpos` | A + learned text positional embeddings (plugin branch `text-pos-embed`) + BERT unfrozen at 1e-6 | cancelled at 32% (not evaluated) | — |

C and A both started from the two-task checkpoint: 5–7k fine-tuning steps did not undo "ignore the instruction"
(A) nor the pink → blue habit (C). A from-scratch single-task run (ACT's protocol) is the clean comparison still owed.

**7.4 Gate v3** ([`sensitivity_gate_v3.py`](scripts/sensitivity_gate_v3.py)), calibrated against SmolVLA, which
works on the arm: episode-start reach (≥ 10°), task-swap (≥ 5°; SmolVLA 7.4°, TurboVLA 0.1–0.25°), and a
**per-layout probe** — same frame, both instructions, does the left-bowl instruction steer further left (shoulder pan)
than the right-bowl one: SmolVLA 86%, every TurboVLA variant 42–64% (chance). An offline plan-consistency probe was
built and **dropped as a verdict**: replaying demos it rates ACT worst (16°) though ACT is the smoothest on the arm —
it measures drift from the demonstrator, not indecision; plan consistency has to be measured on arm rollouts.

**7.5 Is the text path dead? No — the instruction is washed out inside the fusion.**
([`text_path_diagnostic.py`](scripts/text_path_diagnostic.py), CPU, minutes; outputs in
[`projects/turbovla/data/2026-10-01_gate/`](data/2026-10-01_gate/)). On 8 (blue, pink) frame pairs:

| | trimmed (old wording) | A (relabelled + paired) |
|---|---|---|
| input: blue vs pink at the colour token, after `language_proj` | 0.51 / 0.46 | 0.63 |
| fusion layer 0: instruction-specific share of the text→vision update | 26% | 53% |
| fusion layers 2–5 (text update = 0.35–0.81 of the vision stream): instruction-specific share | 1.1–1.7% | 2.6–3.7% |
| decoder cross-attention mass on text (uniform 2–2.7%) | 8–21% | 7–21% |
| gradient per text token / per image patch | 24× | 33× |
| action chunk, blue vs pink | 0.36% | 0.48% |

Not dead: text gets 3–10× its uniform share of attention, a large share of the decoder's output, and strong
gradients. But each patch averages over 8–11 text tokens of which 1–2 differ; by the deep fusion layers the
instruction-specific part is a few percent of a large, near-constant text update, and ~0.5% survives to the actions.
Paired batches doubled that share (1.1–1.7% → 2.6–3.7%) — the right direction, far too weak. This points at the
**fusion design** (no mechanism that singles out the contrastive token), not at a dead path from the initial
checkpoint. Open: the same fusion did respond to a one-word difference on `pick_cups` (4.30°), where the distinct
wording was there from the start of training — so training the relabelled data from a fresh trunk is the cheapest
remaining test before changing the architecture (e.g. pooled-text FiLM conditioning of the trunk, or a dedicated
instruction token the decoder must route through).

**7.6 A fresh start with an auxiliary task head — the decoder learns the task, the actions still don't use it.**
Two runs in parallel from the libero warm start on the relabelled data with paired batches: run 1 with an auxiliary
2-way "which bowl" head on the decoder's final query states — the exact representation `action_head` reads, never the
text tokens (where it would solve the task from the colour word and leave the actions alone) — trained on pre-grasp
frames only (after the grasp, vision alone tells which bowl is held), weight 0.2; run 2 identical without it (the
control that separates "fresh start" from "aux head"). Every 5k-step checkpoint was gated offline while training ran
(data: [`projects/turbovla/data/2026-10-01_fresh_runs/`](data/2026-10-01_fresh_runs/)):

![learning curves](figures/fresh_runs_curve.png)

| step | task-swap, run 1 / run 2 | per-layout, run 1 / run 2 |
|---|---|---|
| 5k | 1.91° / 0.56° | 39% / 33% |
| 10k | **2.29°** / 0.47° | **56%** / 36% |
| 15k | 2.02° / 0.50° | 53% / 56% |
| 20k | 1.38° / 0.67° | 53% / 53% |

- **A fresh start alone does nothing for the instruction** (run 2: 0.5–0.7° throughout) — the training-order
  hypothesis (§7.5, "open") is rejected.
- **The aux head works and changes almost nothing**: ≥ 98% accuracy from step 1k on, so the task is plainly encoded
  in the decoder states, yet run 1's task-swap peaked at 2.3° (10k) and fell back as training went on, and its
  per-layout probe converged to the control's, at chance. Both stopped at ~22k (pre-agreed kill rule, $3.2).
- Together with §7.5 the task information is now shown to be present at **three successive levels** — text tokens
  after projection (46–63% blue/pink difference), fused tokens (layer 0), decoder states (98% linearly decodable) —
  and to vanish only in the **readout to actions**. The bottleneck is not representing the task; it is the action head
  learning to *depend* on it. Next (outside this timebox): make the action computation conditional by construction —
  task-specific action heads or FiLM on the decoder states from the task representation, or a multimodal (flow /
  diffusion) head that cannot average the two tasks.

**7.7 Sanity check: does it use the instruction on its own training frames? Barely — because training barely needs it.**
([`projects/turbovla/scripts/instruction_use_check.py`](scripts/instruction_use_check.py), CPU/MPS, minutes.) On 96 paired pre-grasp
training frames (same layout, same progress):

| | L1, correct instruction | swapped | \|Δpred\| / \|Δtrue\| | cos(Δpred, Δtrue) |
|---|---|---|---|---|
| run 1 (aux), 20k | 0.0728 | +4.3% | 0.023 | +0.10 |
| run 1 (aux), 10k (48 pairs) | 0.1112 | −0.2% | 0.023 | +0.12 |
| run 2 (control), 20k | 0.0680 | +1.5% | 0.010 | +0.13 |

The loss barely depends on the instruction, even at the start of the motion (−0.7% at progress 0–0.1), yet the model
fits those frames well — the target must be predictable without the task. The demos show why (shoulder pan, signed
towards the bowl to pick, all 182 episodes):

| pre-grasp progress | next-0.4 s pan move to the target side | pan already turned to the target side |
|---|---|---|
| 0.0–0.1 | 0.6° (same lift for both tasks) | 0.0° |
| **0.1–0.3** | **7.1°** (the side decision) | 0.35° (> 1° in 46%) |
| 0.3–0.6 | 1.3° | 13.9° |
| 0.6–1.0 | 0.1° | 18.1° |

The instruction is necessary only in the decision band (~6% of all training frames), and partly leaked by the pose
even there. With a 0.4 s horizon and the state as input, the cheapest fit is to continue the arm's current direction —
the copycat / causal-confusion failure of imitation learning. It explains the arm behaviour (the bowl is set by small
motions at the decision point; hesitation when they waver), why the aux head's task knowledge goes unused (the action
loss almost never needs it), and why SmolVLA (50-step chunks) and ACT (100) are fine. This supersedes the "readout"
framing of §7.6: the task is readable; training doesn't ask for it.

**7.8 Takeaways, extending §3.**

1. **Probe the states a rollout actually starts from.** Experiment 2's gate sampled mid-episode frames; the failure
   lived at the rest pose. A short action horizon plus idle demo starts produces a fixed point the loss never sees as
   an error.
2. **A live input path is not a used input.** TurboVLA's text path carries attention (7–21% of the decoder's,
   3–10× uniform) and strong gradients, yet the instruction-specific part of the signal is ~2–4% of what the fusion
   writes and ~0.5% of the action difference: the model uses the text as constant context. Sensitivity probes on the
   *output* (task-swap, per-layout) are what catch this; attention maps alone would have said "it reads the text".
3. **When the scene cannot disambiguate the task, a point-estimate (L1) head averages tasks** — on the arm that is
   hovering between objects, not a clean wrong choice. ACT avoided it by being single-task; SmolVLA by reading the
   instruction (and a flow-matching head).
4. **Neither a fresh start nor an explicit task loss made the actions use the instruction.** Relabelled, paired
   fine-tuning moved task-swap 0.18° → 0.25°; a fresh start, 0.5–0.7°; an auxiliary head that decodes the task from
   the decoder states at ≥ 98% peaked at 2.3° and fell back. Information that is linearly decodable one layer before
   the actions can still be unused by them.
5. **Check what training actually asks for.** A signal can be in the data, in the model, and still unnecessary for the
   loss: with short-horizon targets and proprioception, the robot's own motion can stand in for the instruction on
   almost every frame. Measure the loss with the instruction swapped, per phase of the task.
6. **Measure what reaches the actions.** Attention to language (§7.5), gradients into the text path, a perfectly
   accurate auxiliary task head — each would have read as "the model uses the instruction"; only the output-side
   probes (task-swap, per-layout) showed it doesn't. That is the general lesson for any VLA evaluation.

**Status (end of the timebox).** No TurboVLA checkpoint passes the gate on the two-task bowls data. The instruction is
represented in the model but training barely requires it: with a 0.4 s horizon and the state as input, the arm's own
motion predicts the target on ~94% of frames (§7.7, copycat). On the same task ACT (single-task, 100-step chunks) and
SmolVLA (two-task, 50-step chunks) reach 9–10/10. Next, if resumed: a 50-step-horizon run or state dropout (H4 in §0).
B (text positional embeddings + unfrozen BERT on top of A) was cancelled at 32% and not evaluated.

## 8. Experiment 3 — detailed evidence and solution list

The full diagnosis as it was built during the day: problem statements, evidence tables, the solution list A–G, then dated updates. Where an early explanation was later disproved, it is marked **[superseded]** in place; §0 is the current summary.

### TL;DR

Three problems, stacked. The first is fixed; the other two explain everything seen on the arm since.

| # | Problem | Symptom on the arm | Root cause | Status |
|---|---|---|---|---|
| 1 | **Never starts** | stays at the rest pose for the whole 25 s | demos idle ~1.6 s at rest, but the policy only plans 0.4 s ahead → "at rest, stay" | **fixed** by trimming the idle starts + 5k-step fine-tune |
| 2 | **Does the wrong task half the time** | puts the pink bowl into the blue one | it ignores the instruction, and the two instructions are the same words in a different order | open |
| 3 | **Hesitates / chatters** | hovers, gripper opens/closes, goes back up, never grasps | L1-regression head averages two equally likely plans; it re-plans every 0.4 s and flips between them | open |

Problems 2 and 3 have the same origin: **the model cannot tell the two tasks apart, so for the same scene it sees two
equally valid demonstrations — and an L1 head averages them.** Most promising fix: make the instructions
distinguishable (relabel) and let the language path actually train, then re-check with the offline gate before the arm.

### Setup and timeline

| When | What | Result |
|---|---|---|
| 2026-09-29 | `turbovla_so101_bowls` trained on HF Jobs (t4-small, 35k steps, chunk 12, DINOv3 unfrozen, BERT frozen) | |
| 10-01 morning | offline gate v3 (no robot) | uses vision (black cams → 16° change), tracks demos mid-episode (~1° error), **ignores language (0.2°)** |
| 10-01 | first arm trial (M1) | **barely moved** in 25 s — not registered |
| 10-01 | diagnosis + fix: idle-start trimming, 5k-step fine-tune → `turbovla_so101_bowls_trimmed` | gate episode-start probe: **1.3° → 24°, PASS** |
| 10-01 afternoon | 10 exploratory arm tries on the Orin NX (`outputs/explore`, not gate trials) | 1 success, 3 wrong task, 1 missed grasp, 5 hesitation/stall |

### Problem 1 — never leaves the rest pose (fixed)

**Symptom.** In the first 25 s trial every joint's commanded position moved < 1.3°; the arm stayed at rest. The
harness and the rollout were fine (567 control ticks, 48 chunk inferences in 25 s).

**Root cause.** The demonstrator waits at rest a median **1.6 s** (max 2.2 s) before moving. TurboVLA predicts
`chunk_size = 12` actions = **0.4 s**. Training targets at the first frame of an episode:

| data | horizon | planned motion from rest (median) | episodes where the target is "stay" (< 3°) |
|---|---|---|---|
| original | 12 (0.4 s) | 1.1° | **99%** |
| original | 50 (1.7 s) | 4.4° | 42% |
| trimmed | 12 | 20.7° | 0% |

The model learned the data correctly: at rest, the right 0.4 s-ahead answer is "stay", and nothing in the camera or
state tells it when the operator started moving. On the arm it stays, sees the same observation, stays again. ACT
(3.3 s horizon) and SmolVLA (1.7 s) plan past the idle.

**Fix applied.** [`shared/tools/trim_idle_start.py`](../../shared/tools/trim_idle_start.py) cuts each episode to start 0.1 s
before the first > 3° move (median 1.53 s cut, 8,435 of 69,897 frames) → `ravediamond/so101_bowls_all_trimmed`
(private); 5k-step fine-tune of the existing checkpoint on it (HF Jobs, 55 min). The gate's new **episode-start
probe** ([`sensitivity_gate_v3.py`](scripts/sensitivity_gate_v3.py)) went from 1.3° (FAIL) to 24° (PASS), and on
the arm the policy now starts every time. A longer horizon alone would *not* have fixed it (42% of starts still
"stay" at 50 steps).

### Problem 2 — does the wrong task (task ambiguity)

**Symptom.** With the instruction "blue into pink", the fine-tuned policy completed a pick-and-place 3 times: once
blue → pink (correct), twice pink → blue (reversed). Reversed or wrong-bowl behaviour from both starting layouts.

**Evidence that it ignores the instruction.**

- Offline gate: swapping the instruction blue↔pink changes the predicted actions by **0.16–0.21°** (median over 8
  frames). SmolVLA: 11–13°. Blacking out the cameras changes them by 16–22°. The policy is a vision policy with a
  language input it does not use.
- The scene does not give the task away. Starting layout vs task in the training data is balanced:

  | demos | blue on the left (B\|P) | pink on the left (P\|B) |
  |---|---|---|
  | blue into pink | 36 | 54 |
  | pink into blue | 36 | 56 |

  For any starting scene there are about as many "blue → pink" as "pink → blue" demonstrations, so **only the
  instruction can disambiguate** — and it is not used. Expected outcome: a coin flip on the task.

**[superseded — see "Text-path diagnostic" below and §0] Why it ignores the instruction — the two instructions are nearly identical to the model.**

"Put the **blue** bowl in the **pink** bowl." and "Put the **pink** bowl in the **blue** bowl." are the same 11
BERT tokens in a different order. TurboVLA feeds BERT's per-token outputs (frozen) into cross-attention, and
attention over a set of keys is order-agnostic — only BERT's contextual shift on the colour words can tell the two
apart, and it is small:

| measure (frozen `bert-base-uncased`, last hidden state) | value |
|---|---|
| same multiset of tokens | yes |
| set similarity, blue→pink vs pink→blue (mean best-match cosine) | **0.986** |
| set similarity, blue→pink vs an unrelated instruction | 0.718 |
| "blue" token in sentence A vs "blue" token in sentence B | 0.94 cosine |

With BERT frozen, a 256-dim projection and DINOv3 (unfrozen) carrying a strong, easy visual signal, training has
almost no gradient incentive to find that 1–2% difference: predicting the average of both tasks is "good enough" for
the loss on frames before the grasp, and after the grasp vision alone tells which bowl is held. (SmolVLA reads the
instruction with a full VLM whose language model is trained to use word order; ACT never had the ambiguity — it was
trained on blue → pink only.)

**Not the cause** (checked):
- **Dataset labels.** Last frame of every episode: the moved bowl ends on top. `so101_bowl_blue_in_pink`: 90/90
  correct. `so101_bowl_pink_in_blue`: 91/92 correct — **episode 18 is mislabelled** (blue moved into pink, labelled
  pink into blue) = **episode 108 of `so101_bowls_all`**. One bad episode of 182 cannot explain a ~50% wrong-task
  rate, but it should be fixed. The merge into `so101_bowls_all` preserved labels (first 90 = blue → pink).
- Layout correlation (table above), harness, inference latency (stable ~119 ms/chunk), camera mapping (verified
  every session), calibration (motors match file).

### Problem 3 — hesitation, gripper chatter, hovering

**Symptom.** Hovering above a bowl with the gripper open and never descending; gripper opening/closing repeatedly;
going back up; moving between the two bowls; once carrying a bowl above the other and never releasing.

**Evidence: the plans disagree with each other.** The policy plans 12 actions, executes them, then plans again
(every ~0.5 s on the Orin). Within a plan the motion is smooth; at the hand-over to the next plan, the commanded pose
jumps:

| try | plan-to-plan jump (median) | step inside a plan | gripper reversals | outcome |
|---|---|---|---|---|
| 3 | **0.3°** | 0.2° | 3 | completed (wrong task) |
| 4 | **0.4°** | 0.1° | 3 | ✓ completed (right task) |
| attempt 1 | 2.8° | 0.4° | 8 | carried a bowl, never released |
| attempt 2 | 4.7° | 0.4° | 5 | gripper chatter, no descent |
| 5 | 2.4° | 0.7° | 6 | wrong task |
| 6 | 1.8° | 0.3° | 4 | wrong task |
| 7 | 2.7° | 0.3° | 3 | missed grasp |
| 8 | 2.4° | 0.3° | 1 | hesitated between bowls |
| 9 | 2.7° | 0.4° | 7 | hesitated between bowls |
| 10 | 2.6° | 0.4° | 1 | hovered with gripper open |
| demos (reference) | — | — | 3 | — |

The two tries that completed a full pick-and-place are exactly the two with smooth hand-overs (0.3–0.4°); every
failed try jumps 6–10× more at the hand-over than within a plan.

**Root cause: mode averaging.** TurboVLA's action head is trained with an **L1 loss** (point estimate). When the same
observation has two valid futures (go to blue / go to pink; descend now / a bit later), the loss is minimised by a
prediction *between* them — a pose above neither bowl, a gripper half-way. Each new plan re-weights the two modes from
a slightly different observation, so successive plans flip or blend → hovering, chatter, oscillation. The 0.4 s
horizon makes it worse: 2.4 re-plans per second = 2.4 chances per second to flip. ACT avoids it (single task, plus a
CVAE latent and a 3.3 s chunk); SmolVLA avoids it (instruction resolves the task, flow-matching head can represent
two modes instead of averaging).

Problem 2 *creates* most of the multimodality Problem 3 suffers from: fix the task ambiguity and most of the
averaging disappears.

### Potential solutions

Ordered by cost. "Verify" = what to check offline (gate) before the arm.

#### A. Make the two instructions different words — relabel (cheap, high expected impact) — **[tried: no effect, see updates below]** — **[tried: no effect, see updates below]**

Rewrite the tasks so the token sets differ, e.g. name only the object to pick (the target is then implied):

- "Put the blue bowl in the pink bowl." → **"Pick up the blue bowl."** (or "Move the blue bowl.")
- "Put the pink bowl in the blue bowl." → **"Pick up the pink bowl."**

Now the instructions differ by a whole token instead of word order, which cross-attention over BERT tokens can see
directly. Relabel with `lerobot-edit-dataset --operation.type=modify_tasks` (no video re-encode):
`task_replacements` maps the two old instructions to the new ones, `episode_tasks` fixes episode 108 in the same
pass; fine-tune 5–10k steps from `turbovla_so101_bowls_trimmed` (HF Jobs, ~1 h).
**Verify:** task-swap probe ≫ 0.2° (target: comparable to vision, ≥ 5–10°). Use the new phrasing at rollout.
**Risk:** if the language path is still too weak (frozen BERT, small projection), the swap stays small → then B.

#### B. Let the language path train (moderate)

- **Unfreeze BERT** with a low LR (e.g. 1e-6–5e-6, like the vision backbone's 5e-6), or at least its last layers, so
  it can learn to weight the colour words / word order.
- Add **learned positional embeddings to the text tokens** before fusion, so order survives the set-like attention.
- **Instruction dropout / swap augmentation**: train some batches with the wrong instruction paired with a high loss
  weight on the first post-start actions, or an auxiliary head that must predict "which bowl to pick" from the fused
  features — forces the model to read the instruction.
- **Balanced paired batches**: same starting layout, both tasks, in the same batch — the only way to lower the loss
  on those pairs is to use the text.

**Verify:** task-swap probe; also a per-layout probe (same frame, both instructions → actions head to different bowls).
**Cost:** retrain or long fine-tune; plugin code changes for positional embeddings / aux head.

#### C. Remove the ambiguity from the data — single task (cheap, but stops testing language)

Train/fine-tune on `so101_bowl_blue_in_pink` only (90 episodes, idle-trimmed), exactly like ACT. One valid future per
scene → no task coin flip, far less mode averaging. **Best option for a fair TurboVLA vs ACT gate comparison** (same
data); the language comparison with SmolVLA then needs A/B separately.
**Verify:** episode-start probe, plan-to-plan consistency (below), gt error.

#### D. Smooth the hand-overs at inference — temporal ensembling (no retrain)

The plugin already supports `temporal_ensemble_coeff` (ACT-style): predict a chunk every tick and blend the
overlapping predictions. Damps plan-to-plan flips and chatter.
**Cost:** inference every tick (~119 ms on the Orin) → control loop ~8 Hz instead of ~24 Hz. Will not fix the wrong
task (it averages two tasks even harder) — only worth it combined with A/B/C.
**Alternative:** asynchronous inference (`--inference.type=rtc`) to hide latency; doesn't remove flips by itself.

#### E. Commit to a plan for longer — longer horizon (retrain)

`chunk_size` 50 (1.7 s, like SmolVLA): fewer re-plans per second → fewer chances to flip, smoother motion; also
removes the need for the idle trim. Init checkpoint resizer already exists
([`resize_turbovla_chunk.py`](scripts/resize_turbovla_chunk.py), query embeddings 12 → 50). Does not resolve the
task ambiguity on its own; makes each wrong decision last longer.

#### F. A head that can represent two answers (bigger change)

Replace the L1 regression head with a multimodal one — flow matching / diffusion (SmolVLA, π0), or a CVAE latent
like ACT — so two valid futures are sampled, not averaged. Addresses Problem 3 at its root even when ambiguity
remains (e.g. "descend now vs later"). Plugin architecture change + full retrain.

#### G. Housekeeping

- Fix the mislabelled **episode 108** of `so101_bowls_all` (= ep 18 of `so101_bowl_pink_in_blue`) — relabel to
  blue → pink, or delete it.
- Freeze the [success protocol](../../shared/gate/docs/GATE_SUCCESS_PROTOCOL.md) before the next counted trials.

### Recommended plan (as written before the afternoon runs — superseded by what was run, below and §7)

1. **A + G together** (relabel to "Pick up the blue/pink bowl.", fix ep 108), fine-tune from
   `turbovla_so101_bowls_trimmed` on HF Jobs (~1 h, 2 h timeout).
2. Offline gate must pass **three** checks before the arm: episode-start reach ≥ 10°, **task-swap ≥ 5°** (new
   threshold), **plan-to-plan consistency** (new probe: replay a demo, compare each chunk's last action with the
   next chunk's first — target close to the within-chunk step, i.e. < 1°).
3. If task-swap stays small → **B** (unfreeze BERT at low LR + text positional embeddings).
4. In parallel, for the gate table: **C** (single-task TurboVLA, same data as ACT) — gives a clean TurboVLA number now.
5. **F** is the long-term answer if the hesitation persists once the task is unambiguous.

### Artefacts

| What | Where |
|---|---|
| Exploratory tries (10 recordings, notes, CSV) | Jetson `~/gate/explore/` (`NOTES.md`, `trials_session3.csv`, 3 datasets) |
| Trimmed dataset | `ravediamond/so101_bowls_all_trimmed` (private) |
| Fine-tuned checkpoint | `ravediamond/turbovla_so101_bowls_trimmed` (HF Jobs `6abe23bffbc85ba682360e75`) |
| Gate / trim / resize scripts | [`scripts/`](scripts/) (`sensitivity_gate_v3.py`, `resize_turbovla_chunk.py`), [`trim_idle_start.py`](../../shared/tools/trim_idle_start.py) |
| Chunk-latency analysis | [`GATE_CHUNK_LATENCY.md`](../../shared/gate/docs/GATE_CHUNK_LATENCY.md) |

### Update — 2026-10-01 afternoon: what was tried, and results

Narrative version: §7. Arm data:
[`projects/turbovla/data/2026-10-01_orin_nx/`](data/2026-10-01_orin_nx/).

| Run | Solution(s) | Offline gate v3 | Arm (Orin NX) |
|---|---|---|---|
| C `turbovla_so101_bowls_blue_in_pink` | **C** single task (fine-tune from `…_trimmed`, 7k steps, blue → pink episodes only) | start 24.0° ✓ · tracks blue demos 1.07° vs pink 3.03° · task-swap/per-layout n/a | **1/4** gate trials (2 × went for pink, 1 × hesitated) |
| A `turbovla_so101_bowls_relabeled_paired` | **A + G + paired batches** (relabel "Pick up the blue/pink bowl.", ep 108 fixed, (blue, pink) frame pairs with the same layout and progress in every batch; 7k steps) | start 25.8° ✓ · **task-swap 0.25° ✗** · **per-layout 64% ✗** | 0/2 explore: grasped between the bowls |
| B `turbovla_so101_bowls_b_textpos` | **B** on top of A: text positional embeddings (plugin branch `text-pos-embed` @ `56e21fe`, zero-init, default off), BERT unfrozen at 1e-6 (`TURBOVLA_LANGUAGE_LR`), 6k steps | cancelled at 32% (1.9k / 6k steps) by decision, not evaluated; nothing pushed | — |

Reading so far:

- **Relabelling + paired batches did not make the instruction matter** (0.18° → 0.25°). The distinct token is there
  (the colour token now differs at 0.72 cosine instead of 0.94), but a model that has already learned to ignore the
  text path does not start using it in 7k fine-tuning steps: the trunk has no reason to re-open a path it has learned
  to route around. B (trainable BERT + order information, fine-tuned on top of A) was cancelled before evaluation, expected to
  hit the same wall; the text path likely needs to be trained from a fresh trunk (relabelled data from the start), or the fusion redesigned.
- **Fine-tuning on one task did not remove the other task (C).** Starting from the two-task model, 7k single-task
  steps left a strong pink → blue habit. The clean ACT comparison needs a single-task run from the libero warm start
  (ACT's protocol: trained on blue → pink only, from the start).
- A's "grasp between the bowls" is the averaging failure in its purest form, and it is exactly what the per-layout
  probe predicts offline (64% ≈ chance).

**Gate v3 additions** ([`projects/turbovla/scripts/sensitivity_gate_v3.py`](scripts/sensitivity_gate_v3.py)):
episode-start reach (≥ 10°), task-swap verdict (≥ 5°), per-layout probe (≥ 75%), `--task-map` for relabelled
policies on the untrimmed rest frames. Calibration: SmolVLA passes all three (37.5° / 7.4° / 86%). The offline
plan-to-plan probe is **informational only** — it rates ACT worst (16.4°) though ACT is the smoothest on the arm;
plan consistency must be measured on arm rollouts (plan hand-over jump vs within-plan step).

### Text-path diagnostic — the path is alive, the instruction is washed out in fusion

[`projects/turbovla/scripts/text_path_diagnostic.py`](scripts/text_path_diagnostic.py) on A and on `…_trimmed` (8 blue/pink frame pairs;
outputs in [`projects/turbovla/data/2026-10-01_gate/`](data/2026-10-01_gate/)):

| | trimmed | A |
|---|---|---|
| input: blue vs pink at the colour token (after `language_proj`) | 0.51 / 0.46 | 0.63 |
| fusion L0 / L2–5: instruction-specific share of the text→vision update | 26% / 1.1–1.7% | 53% / 2.6–3.7% |
| decoder attention mass on text (uniform 2–2.7%) | 8–21% | 7–21% |
| gradient per text token vs per patch | 24× | 33× |
| action chunk difference, blue vs pink | 0.36% | 0.48% |

**Correction to "Why it ignores the instruction" above**: the same-tokens explanation is not the bottleneck. The
instruction difference is present at the input (the colour token differs 46–63%) and in a live path (attention,
gradients), and it is the fusion that dilutes it to a few percent of a near-constant text update. Relabelling gave
the colour a distinct token and changed almost nothing for that reason. Next test (cheap): train on the relabelled
data from the libero warm start (fresh trunk) — `pick_cups`, with one-word-different instructions from the start,
reached 4.30° task-swap with the same fusion. If that also stays < 5°: change the conditioning (pooled-text FiLM on
the trunk, or an instruction token the decoder must route through) rather than the data.

### Fresh runs with an auxiliary task head — task encoded in the decoder, not used by the actions

§7.6, data and learning curve in
[`projects/turbovla/data/2026-10-01_fresh_runs/`](data/2026-10-01_fresh_runs/). Both from the libero warm start,
relabelled data, paired batches, stopped at ~22k of 35k steps:

| | run 1: + aux "which bowl" head (decoder states, pre-grasp, w = 0.2) | run 2: control |
|---|---|---|
| task-swap 5k / 10k / 15k / 20k | 1.91 / **2.29** / 2.02 / 1.38° | 0.56 / 0.47 / 0.50 / 0.67° |
| per-layout 5k / 10k / 15k / 20k | 39 / **56** / 53 / 53% | 33 / 36 / 56 / 53% |
| aux accuracy | ≥ 98% from step 1k | — |

- **Training order is not the cause** (run 2, fresh start: 0.5–0.7°). The "fresh trunk" option above is closed.
- **The task is encoded one layer before the actions and still unused**: run 1's aux head decodes "which bowl" at ≥ 98%
  from the very states `action_head` reads, while the action sensitivity peaks at 2.3° and decays. Same pattern as the
  text-path diagnostic, one level further down: the bottleneck is the readout from task-aware states to actions.
- Remaining options (outside the timebox): make the action computation task-conditional by construction (per-task
  action heads, or FiLM of the decoder states by the task representation), or a multimodal action head (flow /
  diffusion) that cannot average the two tasks. B (text positional embeddings + unfrozen BERT on top of A) was
  cancelled at 32% and not evaluated.

## 9. How to — train, gate, run rollouts

### Training on HF Jobs

#### Prerequisites

- HF account with billing set up for HF Jobs (pay-as-you-go, or included quota on Pro)
- `HF_TOKEN` with `write` scope (**never paste a real token into a file or chat —
  pass it as a job secret**, see below)

#### Launch

```bash
hf jobs run --flavor t4-small \
  --secrets HF_TOKEN \
  python:3.12 \
  bash -c "
    pip install 'lerobot[dataset,training]' lerobot-policy-turbovla-so101 &&
    lerobot-train \
      --dataset.repo_id=ravediamond/pick_cups_all \
      --policy.type=turbovla_so101 \
      --policy.device=cuda \
      --output_dir=outputs/train/turbovla_so101_pick_cups \
      --job_name=turbovla_so101_pick_cups \
      --steps=20000 \
      --batch_size=8 \
      --policy.push_to_hub=true \
      --policy.repo_id=ravediamond/turbovla_so101_pick_cups
  "
```

`--secrets HF_TOKEN` reads your local `hf auth login` token and injects it into the
job as `HF_TOKEN` — needed even when already logged in locally, since `hf jobs run`
does **not** forward your ambient auth into the job container on its own. Without it,
gated-repo access (`facebook/dinov3-vitb16-pretrain-lvd1689m`, the default vision
backbone) fails with a 401 even if your account has access — confirmed the hard way:
first run without `--secrets HF_TOKEN` failed here despite local `hf` auth already
having gated access approved.

#### Notes

- `push_to_hub`/`repo_id` are nested under `policy` in this CLI (`--policy.push_to_hub`,
  not a top-level `--push_to_hub`) — matches `PreTrainedConfig`'s own fields.
- torchcodec may fail to load in a fresh pip env (CUDA-version mismatch between the
  pulled `nvidia-cuda-runtime` wheel and the job image's driver) — harmless, it falls
  back to `pyav` automatically and training still runs.
- The plugin's `lerobot` dependency doesn't pull the `dataset` or `training` extras
  by default — install `lerobot[dataset,training]` explicitly (already in the command
  above), otherwise `lerobot-train` fails with `ImportError: 'datasets' is required...`
  (missing `dataset`) and then, once that's fixed, `ImportError: 'accelerate' is
  required...` (missing `training`) on the next run.
- `--flavor` picks the GPU tier — check current options/pricing with `hf jobs run --help`
  or the HF Jobs pricing page; `t4-small` is a reasonable, cheaper starting point —
  TurboVLA is lightweight (210M params, DINOv3+BERT, no full VLM backbone), so a T4's
  16GB is plenty at `batch_size=8`. Bump to `a10g-small` if the run feels too slow;
  T4 is roughly half an A10G's throughput per step.
- `--steps=20000`/`--batch_size=8` are starting points for a 180-episode dataset —
  adjust for GPU memory and time budget.
- `vision_backbone` defaults to `facebook/dinov3-vitb16-pretrain-lvd1689m` (gated —
  needs accepted license + `HF_TOKEN` with access). If access is still pending, add
  `--policy.vision_backbone=facebook/dinov2-base` to swap to the public model (same
  substitution used for the earlier dry-run smoke test) — but this diverges from the
  plugin's real published config, so resolve the gate first if possible.
- After training: real SO-101 rollout (N trials, success rate + latency), fill the
  `lerobot-policy-turbovla-so101` README `Results` section, post the numbers as a
  comment on `huggingface/lerobot#4400`.

#### 2026-10-01 — bowls runs (fine-tunes, HF Jobs t4-small)

All from the existing bowls checkpoints, each well under 2 h. Context: §8.

| Run | Data | From | Steps | Script |
|---|---|---|---|---|
| `turbovla_so101_bowls_trimmed` | `so101_bowls_all_trimmed` (idle starts cut) | `turbovla_so101_bowls` | 5k | `lerobot-train` |
| `turbovla_so101_bowls_blue_in_pink` (C) | same, `--dataset.episodes=[0,…,89]` (blue → pink only) | `…_trimmed` | 7k | `lerobot-train` |
| `turbovla_so101_bowls_relabeled_paired` (A) | `so101_bowls_all_relabeled` | `…_trimmed` | 7k | `projects/turbovla/scripts/train_turbovla_paired.py` |
| `turbovla_so101_bowls_b_textpos` (B) | `so101_bowls_all_relabeled` | A | 6k | same + `--policy.use_text_pos_embed=true --policy.freeze_language_backbone=false`, `-e TURBOVLA_LANGUAGE_LR=1e-6`, plugin `@text-pos-embed` |
| `turbovla_so101_bowls_fresh_aux` (run 1) | `so101_bowls_all_relabeled` | libero warm start | 35k planned, stopped at ~22k | same + `-e TURBOVLA_AUX_WEIGHT=0.2 -e TURBOVLA_UPLOAD_CHECKPOINTS_TO=<repo>` |
| `turbovla_so101_bowls_fresh` (run 2, control) | `so101_bowls_all_relabeled` | libero warm start | 35k planned, stopped at ~22k | same, `TURBOVLA_UPLOAD_CHECKPOINTS_TO` only |

```bash
# plain lerobot-train fine-tune (trimmed, C)
hf jobs run --detach --flavor t4-small --timeout 2h --secrets HF_TOKEN python:3.12 bash -c "
  pip install 'lerobot[dataset,training]' 'lerobot-policy-turbovla-so101 @ git+https://github.com/ravediamond/lerobot-policy-turbovla-so101.git@daaff77' &&
  lerobot-train --dataset.repo_id=ravediamond/so101_bowls_all_trimmed --policy.type=turbovla_so101 \
    --policy.pretrained_path=ravediamond/turbovla_so101_bowls --policy.freeze_vision_backbone=false \
    --policy.device=cuda --steps=5000 --save_freq=5000 --batch_size=8 --seed=1000 \
    --policy.push_to_hub=true --policy.repo_id=ravediamond/turbovla_so101_bowls_trimmed ..."

# paired-batch training (A, B): uploads the local script, lerobot pinned in its header, plugin passed with --with
hf jobs uv run --detach --flavor t4-small --timeout 2h --secrets HF_TOKEN [-e TURBOVLA_LANGUAGE_LR=1e-6] \
  --with "lerobot-policy-turbovla-so101 @ git+https://github.com/ravediamond/lerobot-policy-turbovla-so101.git@<ref>" \
  projects/turbovla/scripts/train_turbovla_paired.py --dataset.repo_id=ravediamond/so101_bowls_all_relabeled ...
```

Notes from these runs:

- **Always pass `--timeout`.** The 35k-step `turbovla_so101_bowls` run (no timeout given) trained for 5 h 58 m and
  pushed 13 s before the job's default 6 h limit killed it (`ERROR: Job timeout` after a successful push).
- **Dataset pushes**: `HF_HUB_DISABLE_XET=1` + `push_to_hub(..., upload_large_folder=True)` (the default Xet path
  stalls on these ~2 GB video datasets). Private repos are fine: the jobs read them with `--secrets HF_TOKEN`.
- Speed on t4-small: 1.55–1.63 steps/s (unfrozen DINOv3), 1.47 steps/s with BERT unfrozen as well.
- `--dataset.episodes=[...]` accepts a trailing comma (`[0,1,…,89,]`, what macOS `seq -s,` produces).
- `torchcodec ... cannot be loaded` / `libavutil.so.NN` tracebacks at startup are harmless: lerobot falls back to pyav.
- Data prep scripts: `shared/tools/trim_idle_start.py` (must run under `if __name__ == "__main__"` — the video encoder
  spawns workers), relabelling with `lerobot.datasets.dataset_tools.modify_tasks` (`task_replacements` +
  `episode_tasks`; in place, so run it on a copy).

##### Fresh runs (run 1 / run 2) — notes

- **Cost per 1k steps decides the flavour.** HF Jobs prices (`hf jobs hardware`): t4-small $0.40/h at ~1.57 steps/s
  ≈ **$0.07 per 1k steps** (35k ≈ 6.3 h ≈ $2.5). To match it per step: l4x1 ($0.80) must be > 2× faster, a10g-large
  ($1.50) > 3.75×, l40sx1 ($1.80) > 4.5×, a100-large ($2.50) > 6×. With ~$6.7 of credit the expensive benchmarks were
  cancelled; the l4x1 one never got hardware (stayed `SCHEDULING` ~20 min). Stayed on t4-small.
- `TURBOVLA_UPLOAD_CHECKPOINTS_TO` + `--save_freq=5000` → intermediate checkpoints under `checkpoints/<step>/` in
  the output repo, probed live with `projects/turbovla/scripts/probe_checkpoints.py` (learning curve + kill rule).
- `projects/turbovla/scripts/probe_checkpoints.py` must not swallow Hub errors: the first version treated any listing error as "repo not
  created yet" and silently missed the 10k checkpoints; it now prints and retries.
- A gated run that is clearly failing should be stopped on a rule written before launch (here: run 1's task-swap gap
  over the control at the same step) — saved ~$1.9 of the remaining credit.

### Gating a checkpoint offline

`projects/turbovla/scripts/sensitivity_gate_v3.py --ckpt <repo or dir> --dataset ravediamond/so101_bowls_all` (add `--task-map '{"Put the blue bowl in the pink bowl.": "Pick up the blue bowl.", "Put the pink bowl in the blue bowl.": "Pick up the pink bowl."}'` for relabelled checkpoints). Pass: episode-start reach ≥ 10°, task-swap ≥ 5°, per-layout ≥ 75% (the plan-to-plan jump is informational offline). `projects/turbovla/scripts/text_path_diagnostic.py` shows where an instruction signal is lost; `projects/turbovla/scripts/probe_checkpoints.py` gates intermediate checkpoints of a running job.

### Running a rollout

Use the gate harness ([`shared/gate/README.md`](../../shared/gate/README.md)), not raw `lerobot-rollout`: it records
every trial as a dataset episode, logs the gate-schema row, and handles the operator prompts.

```bash
# M1 (from the lerobot checkout, plugin installed in its venv)
uv run python ~/Workspace/git_repos/road-to-robotics/shared/gate/monitor_rollout.py --policy turbovla_c --trials 10 \
  --robot-profile ~/Workspace/git_repos/road-to-robotics/shared/gate/configs/robots/so-101-mac.json
# Orin NX (plugin installed by jetson_setup.sh into ~/gate/pydeps)
shared/gate/jetson_run.sh monitor_rollout.py --policy turbovla_c --trials 10
# relabelled checkpoints need the new wording:  --task "Pick up the blue bowl."
# exploratory tries (not gate results):          --out outputs/explore
```

Before any arm time: `projects/turbovla/scripts/sensitivity_gate_v3.py` must pass episode-start reach ≥ 10°, task swap ≥ 5° and the
per-layout probe ≥ 75% (multi-task checkpoints; SmolVLA passes all three).

#### Hardware facts (verified, supersede session 1's notes)

- Follower = serial `<serial>` (Mac `/dev/tty.usbmodem<serial>`), leader = `<serial>`; `--robot.id=so-101`,
  calibration in lerobot's default location (copies in [`shared/gate/configs/calibration/`](../../shared/gate/configs/calibration/)).
- **Cameras: Logitech C920 = `front_top_camera`, icSpring = `wrist_camera`.** On the Mac that is OpenCV index **0 =
  front, 1 = wrist** — session 1's notes had them the other way round, which is wrong. Re-verify after any replug
  (compare a live frame with a recorded one). Jetson: `/dev/v4l/by-id/…` paths in `shared/gate/configs/robots/so-101-jetson.json`.
- `observation.state` / `action`: 6-D, degrees. The `base` strategy drives the arm as soon as it connects (no `/start`
  gate).

---

### Checked and ruled out on the rig (both sessions)

Rollout code and inference path (normalisation, image scaling, colour order, task plumbing, camera order, temporal
ensembling off); dataset labels (1 mislabelled episode of 182, fixed in A); bowl layout ↔ task correlation (none);
camera mapping (verified against reference frames every session); calibration (motors match the files); latency
(stable ~119 ms per chunk on the Orin, ~150 ms on the M1); disk space (a full disk broke the first session's cameras
and one ACT save — check before every session).

## 10. Artifacts index

| Kind | Name | What |
|---|---|---|
| checkpoint | `ravediamond/turbovla_so101_pick_cups` | Exp. 1, frozen backbones — shortcut |
| checkpoint | `ravediamond/turbovla_so101_dinov3_libero_init` | libero-converted warm start (§5) |
| checkpoint | `ravediamond/turbovla_so101_pick_cups_v2` | Exp. 2, passes the pick_cups gate, never put on the arm |
| checkpoint | `ravediamond/turbovla_so101_bowls` | Exp. 3, 35k steps — never starts |
| checkpoint | `ravediamond/turbovla_so101_bowls_trimmed` | + idle-trimmed fine-tune — starts, ignores the instruction |
| checkpoint | `ravediamond/turbovla_so101_bowls_blue_in_pink` | C: single-task fine-tune |
| checkpoint | `ravediamond/turbovla_so101_bowls_relabeled_paired` | A: relabelled + paired batches |
| checkpoint | `ravediamond/turbovla_so101_bowls_fresh` / `_fresh_aux` | fresh control / fresh + aux head (private; `checkpoints/005000 … 020000/`) |
| checkpoint | `ravediamond/turbovla_so101_bowls_v2` / `_v2-ema` | fresh, paper optimizer recipe + EMA (plugin v0.2.0) — 0/3 on the arm (fact 8) |
| dataset | `ravediamond/so101_bowls_all` | 182 episodes, original wording |
| dataset | `ravediamond/so101_bowls_all_trimmed` | idle starts cut (private) |
| dataset | `ravediamond/so101_bowls_all_relabeled` | + "Pick up the blue/pink bowl.", ep 108 fixed (private) |
| plugin branch | `text-pos-embed` @ `56e21fe` | optional text positional embeddings (B, not evaluated) |
| data | `projects/turbovla/data/2026-10-01_orin_nx/` | arm trials on the Orin NX (gate + exploratory) |
| data | `projects/turbovla/data/2026-10-01_gate/` | gate v3 results, text-path diagnostic outputs |
| data | `projects/turbovla/data/2026-10-01_fresh_runs/` | probe curve, aux-head log |
| data | `projects/turbovla/data/2026-10-08_gate/` | gate v3 results for v2 / v2-ema |
| data | `projects/turbovla/data/2026-10-08_orin_nx/` | v2 arm trials (3), live no-motion probe, frame strips |
| figure | `projects/turbovla/figures/fresh_runs_curve.png` | learning curves of the fresh runs |
| scripts | `scripts/` | `sensitivity_gate_v3.py`, `text_path_diagnostic.py`, `train_turbovla_paired.py`, `trim_idle_start.py`, `resize_turbovla_chunk.py`, `probe_checkpoints.py`, `plot_fresh_runs.py`, `convert_turbovla_checkpoint.py`, `sensitivity_gate_v2.py`, `ground_truth_compare_v2.py`, `shared/gate/` |

## 11. Remaining deliverables and the #4400 status message

**Both done 2026-10-08.** Plugin README `Results` filled (plugin `96a3a32`); comment posted on
[#4400](https://github.com/huggingface/lerobot/issues/4400#issuecomment-6058427536). The draft below is the earlier
version (pre-v2), kept for the record.

1. ~~Fill the `Results` section of the plugin README with real numbers and plain caveats (sample size, one dataset, one environment).~~
2. ~~Post the promised comment on the closed [#4400](https://github.com/huggingface/lerobot/issues/4400).~~ Draft:

> Ported TurboVLA to LeRobot as a plugin and ran it on a real SO-101 (M1 and Jetson Orin NX). Three failure modes so
> far, all training-side, none in the rollout path: (1) frozen backbones + `action ≈ state` teleop data → a
> proprioceptive shortcut (fixed by a warm start and an unfrozen DINOv3); (2) idle starts in the demos + a 0.4 s action
> horizon → a fixed point at the rest pose (fixed by trimming idle starts); (3) on a two-task dataset whose
> two tasks share the same scenes, the model ignores the instruction (0.2° response) and its L1 head averages the
> two tasks — hovering between objects on the arm. The text path is alive (attention, gradients) but the fusion
> dilutes the instruction-specific signal to ~0.5% of the action. Relabelling, paired batches, a fresh start, and an
> auxiliary task head (which decodes the task from the decoder states at 98%) all failed to make the actions use it:
> the task is represented, the action readout ignores it.
> On the same task, ACT and SmolVLA reach 9–10/10. Will post success rates once a checkpoint passes the offline gate.

## Appendix — raw data

- Training log excerpt (loss trajectory):
  `step:200 loss:0.834` → `step:600 loss:0.633` → `step:7K loss:0.137` →
  `step:8K loss:0.132` → `step:11K loss:0.121` → `step:13K loss:0.109` →
  `step:17K loss:0.100` → `step:20K loss:0.098`.
- Rollout findings and current rollout commands: §9. The original live-robot diagnostic script (Exp. 1) is in git
  history: `git show fabfa47:docs/turbo-vla-findings.md`; the offline gate that replaces it is
  `projects/turbovla/scripts/sensitivity_gate_v3.py`.
- TurboVLA paper (arXiv:2607.27205), converted to markdown for reference:
  `2607.27205v1-turbo-vla.md` in this repo.
- Training launch commands, environment setup and troubleshooting history: §9.
- Checkpoint conversion script (source → SO-101-shaped warm-start), verified
  working: `convert_turbovla_checkpoint.py` in this repo. Output pushed to
  [`ravediamond/turbovla_so101_dinov3_libero_init`](https://huggingface.co/ravediamond/turbovla_so101_dinov3_libero_init).
- Experiment 2 trained checkpoint:
  [`ravediamond/turbovla_so101_pick_cups_v2`](https://huggingface.co/ravediamond/turbovla_so101_pick_cups_v2).
  No-robot sensitivity gate and ground-truth comparison scripts:
  `sensitivity_gate_v2.py` and `ground_truth_compare_v2.py` in this repo.
