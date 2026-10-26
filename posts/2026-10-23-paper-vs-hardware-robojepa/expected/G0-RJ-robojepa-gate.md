# G0-RJ: RoboJEPA gate (usefulness on our SO-101 arm, Orin latency)

Frozen 2026-10-13, before any RoboJEPA run. Repo convention (checked): protocols live next to the code in the project folder
(`botfails/BOTFAILS_PROTOCOL.md`, `wm_train/G*_PROTOCOL.md`), results in `results/`; there is no `docs/experiments/`, so this file
and `robojepa_gate/` are used instead of the default path. Anything added after seeing results is labelled **exploratory**.
The LeWM / W1 / W2 pipeline is not modified.

## Hypothesis

A pretrained RoboJEPA predictor (frozen V-JEPA 2.1-G encoder, 1664-dim patch features, action-conditioned predictor 22M-8B; Meta FAIR,
arXiv 2610.10515, code `facebookresearch/robo_jepa`, CC BY-NC-SA 4.0 repo licence, encoder weights from `dl.fbaipublicfiles.com`) predicts
the latent future of our SO-101 bowls data better than copy-last and uses the action, so it can replace or complement LeWM / W2 as a world
model and failure evaluator (H3). Its training mix includes LeRobot SO-101 data; leakage of our own dataset is checked and noted below.

## Gate A: usefulness (vast.ai, one 24-48 GB GPU)

- **Checkpoints.** 300M primary; 100M and 1B if budget allows. Prefer the K = 10 `cooldown_*` (long-horizon) files. Encoder licence / gating
  checked and recorded before download.
- **Data.** 20-50 held-out episodes of our SO-101 bowls dataset (episode ids logged; they are the G1 validation episodes of
  `wm_train/split.json` unless stated). Whether the dataset was public on HF before RoboJEPA's data collection is checked and noted either way.
- **Preprocessing (matching the paper).** Wrist + top views, 240x320, ImageNet normalization, deterministic center crop. Decimate 30 fps to the
  world-model rate RoboJEPA uses for LeRobot (read from their config, not guessed). SO-101 **joint-state** embodiment id; joints in radians
  (degrees converted); action = state delta s_{t+1} - s_t over the decimated stride; zero-padded to their padded width.
- **Camera ids.** (a) both views as the default / unmapped id used for LeRobot in training; (b) wrist = wrist id, top = top-down id. Both reported.
- **Eval protocol (paper's).** Random start timestep, 2 context frames, 9-step rollout with the true actions, per-token L1 against the encoder
  features of the real future frames, averaged over episodes; L1 per horizon 1..9.
- **Baselines on the same clips.** copy-last-frame (z_{t+k} = z_t); shuffled actions (actions from another episode).
- **Failure signal (only if failed rollouts exist).** Per-timestep surprise = L1 between predicted latent and the actual encoded frame, plotted
  over time with the failure moment marked. Search for failed rollouts first; if none, skip and flag.
- **Logging per run.** git commit hash, checkpoint name / size, encoder version, episode ids, config, hardware, cost / time. Results JSON / CSV
  and PNG in `results/`. Checkpoints, datasets and latent caches never committed.
- **Budget.** Stop and report if > $20 or > 6 h; destroy the instance when done. **Standing rules apply:** hard cap $15 per day across all work,
  the owner approves the dry-run and cost before the run starts, no HF write token on the instance.

## Gate B: latency on Jetson Orin NX

If the Orin is reachable over SSH: JetPack / PyTorch versions, load V-JEPA 2.1-G in fp16, time the encoder forward at 240x320 for 1 and 2 views
(10 warmup + 50 timed, p50 / p95 latency, peak memory), and one predictor step (100M and 300M) with a filled KV cache if it fits. TensorRT only
if eager fp16 misses the target by less than 2x. If the Orin is not reachable: the script is committed with the exact commands.

## Pre-registered go / no-go criteria (verbatim)

Gate A GO if all of:
1. 9-step L1 is at least 15% lower than copy-last-frame (average over held-out episodes, best camera-id setting);
2. true-action L1 is at least 5% lower than shuffled-action L1 (action conditioning works on our arm);
3. if failure episodes exist: surprise rises at or before the failure in the majority of failed episodes.

Gate B GO (on-device encoder) if 2-view encoder p50 <= 200 ms in fp16. Otherwise: encoder runs off-robot (Mac/cloud over LAN). Gate B failing
does NOT kill the RoboJEPA path.

## Deliverable

This file filled in (tables, plots, cost / time, deviations, GO / NO-GO per gate, next step: fine-tune RoboJEPA on bowls vs stay on LeWM), one
line in the architecture doc pointing here, branch pushed, PR opened, 5-line summary.

## Open items to resolve before running (not decided yet)

- LeRobot world-model rate, SO-101 embodiment ids, camera ids and padded action width: read from the robo_jepa configs (README does not state them).
- Encoder weight licence / gating; whether the pretrained checkpoints include the SO-101 episodes of our dataset.
- Whether failed rollouts exist (BotFails is other robots, so likely only our own deliberate failures once the Tuesday brick recording exists).

## Clarifications fixed before the first run (2026-10-13, after reading the robo_jepa code, before any number)

Facts read from `facebookresearch/robo_jepa` @ `f93106f5` (configs `high_compute_cooldown/*.yaml`, `docs/action_spaces/so101.md`,
`scripts/evals/eval_droid_holdout.py`):
- LeRobot SO-101 world-model rate is **5 Hz (raw 30 Hz, 6-frame stride, 0.2 s per step)**; id `so-101_joints` = 25; state = 5 joints + gripper
  (6 dims), action = their element-wise difference; both zero-padded to 64, nothing normalized; gripper = LeRobot value x pi/180.
- View ids from the training `view_map`: left 0, right 1, wrist 2, top 5, front 6. There is **no "default / unmapped" id** (the loader raises
  on a camera name that matches no key). So the two settings are: **(a)** our `front_top_camera` read as `front` (id 6) + wrist 2;
  **(b)** `front_top_camera` as `top` (id 5) + wrist 2. This replaces "(a) default/unmapped id" in the task text.
- Checkpoints: the bold files of the README (paper's real-robot checkpoints, K = 10 cooldown): 300M @ 300k (primary), 100M @ 150k, 1B @ 300k.
  The V-JEPA 2.1 ViT-G/384 file is 30.2 GB; the README gives no licence or gating for it (direct download), repo licence CC BY-NC-SA 4.0.
- The released rollout eval **feeds the true future joint states as inputs** (`states_suffix`), not only the actions. With true states the
  action-shuffle test is confounded (the state already carries the next action).

Definitions of the criteria, fixed now:
1. **Criterion 1 metric** = the paper's `rollout L1 mean over horizons 1..9` (per-token L1 on layer-normed encoder features, averaged over
   views, tokens, channels, all prefixes of the 16-frame clips, then clips); the step-9 value and the 2-context-frame-only (prefix of 2 frames)
   value are reported next to it. Prefix t has t+1 real context frames; horizon j = frame t+1+j. **Copy-last** = the last real context frame
   repeated, same targets (horizon j+1 frames from the last real frame, like the model). "Best camera-id setting" = the better of (a), (b).
2. **Criterion 2 metric** = same L1 with the actions of a clip of **another episode** (same length) **and the states re-integrated from
   them** (s_{k+1} = s_k + a_k, state-consistent, what a planner provides), versus true actions. The paper-literal arm (shuffled actions,
   true states) is reported too; criterion 2 is judged on the state-consistent arm. Shuffling uses a fixed seed.
3. Clips: 200 random 16-frame windows (5 Hz) from the 25 G1 validation episodes (8 per episode, `wm_train/split.json`, revision `4e5c6fda8dc5`),
   same windows for settings (a) and (b). Preprocessing: the 4:3 frames (480x640) are resized to 240x320 (aspect range fixed to 4/3, no random
   crop; the DROID eval's [1.333, 1.777] range would crop at random), ImageNet normalization.
4. **Criterion 3 cannot be evaluated now**: there are no failed rollouts in our data (the 182 bowls episodes are successful teleop; BotFails are
   other robots, not an SO-101 bowls task). It stays "not evaluated" until failure episodes exist (Tuesday brick recording); GO then rests on 1 and 2
   and is flagged as such.
5. **Leakage.** `ravediamond/so101_bowls_all_trimmed` was created on the Hub on 2026-10-01 (public, Apache-2.0), six days before the RoboJEPA
   paper (2026-10-07); RoboJEPA's LeRobot SO-101 training data is `lerobot/single_arm_labeled` (a pooled set). Training on our episodes is
   therefore very unlikely but **not verifiable** from the released material; the numbers are reported as "possible but unlikely leakage".
6. **Gate B.** The Orin is on the LAN (`192.168.1.149`) but `ssh` from the Mac is refused (`Permission denied (publickey,password)` with the
   Mac user name), so it was not run. `bench_orin.py` is committed with the commands below; it has not been executed on a device.
7. **Cost.** Standing rule: hard cap $15 per day across all work (stricter than the task's $20), and the owner approves the dry-run before the
   launch. Dry-run: one RTX 3090 class instance, 90 min + 45 min grace, 45 GB download, at most about $0.6 for the 300M run.

Commands for Gate B on the Orin (clone `facebookresearch/robo_jepa`, JetPack torch with CUDA, the encoder file and the checkpoints copied there):

    PYTHONPATH=$PWD python robojepa_gate/bench_orin.py --encoder vjepa2_1_vitG_384.pt --models-dir models --sizes 100m 300m --steps 150 300 --out results/gateB_orin.json

## Owner's decisions, 2026-10-13 (before the run)

- Go for all three predictors: 300M @ 300k (primary), 100M @ 150k, 1B @ 300k. V-JEPA 2.1-G encodes the clips once per camera setting; the
  features are cached on the instance and reused by the three predictors.
- **Primary camera setting = the one the released training config assigns.** The `Lerobot_joints` entry (`cameras: ['.*', '.*']`, two cameras
  chosen at random) maps camera **names** to ids through `view_map` with `re.match` in dict order: `front` 6, `front_1`/`front_2` 6, `left` 0,
  `right*` 1, `top*` 5, `wrist` 2. Our names `front_top_camera` and `wrist_camera` therefore get **front = 6 and wrist = 2** by the config's own
  rule, which is setting **(a)**. So **(a) is the primary (in-distribution) setting** and (b) (top = 5) is an ablation. (The exact name each
  pooled LeRobot dataset had in training is not in the release; the rule above is what the loader would apply.)
- Criterion 2: primary = shuffled actions with re-integrated states; the paper-literal arm (true states) is secondary. Approved.
- "9-step L1" = mean over horizons 1-9; step 9 and the 2-context-frame value are secondary. Approved.
- Leakage: the RoboJEPA training necessarily predates our 2026-10-01 upload; treated as no leakage.
- **The Gate A verdict is at most "conditional GO" until criterion 3 is evaluated on the Tuesday failure recordings.**
- Gate B: the owner sets up key auth on the Orin and gives user and host; `bench_orin.py` stays ready.
- Run time cap raised to 240 min (the 1B predictor is slow: about 32k rollout forwards per setting); cost still at most about $1.5 at
  $0.29/h, inside the $15/day cap.

## Gate B result (Orin NX, 2026-10-13) and architecture decision

Run on the Orin (`jetson@Orin NX.local`, JetPack R36.4.3, `lerobot-jetson:rc` image: torch 2.11.0, CUDA 12.6), power mode `MAXN_SUPER`,
`bench_orin.py` at `2d8ba28`+ (fp32 weights under fp16 autocast: pure `.half()` fails on the encoder's rotary q/k dtypes), 240x320, 10 warmup +
50 timed, `results/gateB_orin.json`. V-JEPA 2.1-G weights = the `target_encoder` of `vjepa2_1_vitG_384.pt` stored as fp16 (3.7 GB; the original
is 30 GB and does not fit the Orin's 14 GB free disk).

| Item | p50 ms | p95 ms | peak mem |
|---|---|---|---|
| Encoder, 1 view | 397.5 | 398.2 | 7.1 GB |
| Encoder, 2 views | 526.1 | 526.4 | 7.1 GB |
| Predictor step 100M @150k, KV cache | 252.1 | 252.6 | 1.3 GB |
| Predictor step 300M @300k, KV cache | 486.5 | 487.3 | 2.3 GB |

**Gate B: NO-GO on-device** (2-view encoder p50 526 ms vs the 200 ms target, 2.6x over). As pre-registered, TensorRT is only tried if eager fp16
misses by less than 2x, and the owner confirmed it is not tried. Caveat: fp32 weights under autocast, not pure fp16 weights.

**Architecture decision (owner, 2026-10-13):** the world model runs **off-board**: the robot streams frames and joint states over the LAN and the
monitor sits **outside the control loop**; the Orin stays dedicated to the VLA. Gate B failing does not kill the RoboJEPA path (pre-registered).

## Gate B' result (Mac, no GPU rental, 2026-10-13)

Same `bench_orin.py` (now `--device mps|cpu|cuda`), same settings, `results/gateB_mac.json`. Mac (Apple silicon, 24 GB, other apps open),
torch 2.7.1, `PYTORCH_ENABLE_MPS_FALLBACK=1`.

| Item | device | p50 ms | p95 ms | memory |
|---|---|---|---|---|
| Encoder, 1 view | MPS | 761.8 | 921.9 | 7.2 GB driver alloc (no peak counter on MPS) |
| Encoder, 2 views | MPS | 1025.1 | 1125.9 | 8.3 GB driver alloc |
| Predictor step 100M @150k, KV cache | **CPU** | 10353 | 12005 | not measured |
| Predictor step 300M @300k, KV cache | **CPU** | 23405 | 27642 | not measured |

Unsupported op: the predictor's `F.scaled_dot_product_attention` fails on MPS (`Placeholder storage has not been allocated on MPS device`),
so the predictor step ran on CPU with bf16 autocast (fp16 autocast not used on CPU). The encoder ran fully on MPS. **Target (2-view encoder p50
<= 200 ms): not met, 1025 ms (5x over); the Mac is about 2x slower than the Orin for the encoder.** A Mac host is therefore not enough for real-time use of the 1B-parameter encoder. **Owner's decision (2026-10-13): the off-board host must be CUDA.** The MPS SDPA fallback is not fixed.

## Gate A results (vast.ai, 2026-10-13)

Run: instance, NVIDIA RTX PRO 4000 Blackwell 24 GB (not a 3090), $0.2417/h, 14:57 to 19:36 UTC (4.65 h), about $1.2 including disk and
downloads. robo_jepa `f93106f5`; checkpoints `cooldown_{300m_300k, 100m_150k, 1b_300k}.pth.tar` (sha256 and sizes in each JSON), V-JEPA 2.1 ViT-G/384
`vjepa2_1_vitG_384.pt` (sha256 `7aae1a3c7a31...`, one encoding shared by the three predictors); validation episodes = the 25 episodes of
`wm_train/split.json` (ids in the JSON), 200 windows encoded, the first N used. Raw results `results/gateA/`, analysis `analyze.py`
(bootstrap over episodes, 2000 resamples), `results/gateA_summary.json`, plot `results/gateA_l1_per_horizon.png`. Rollout through the KV cache
(the released eval's option); checked against the uncached rollout on 2 clips: max |difference| 0.0005 on values of about 0.3 (bf16 noise).

Per-token L1, mean over horizons 1-9 (lower is better); reductions are 1 - model_true / baseline, 95% CI by bootstrap over episodes:

| run | clips | true actions | shuffled (states re-integrated) | shuffled (true states) | copy-last | **crit 1**: vs copy-last (>= 15%) | **crit 2**: vs shuffled (>= 5%) | vs shuffled, true states | step 9 vs copy-last | 2 context frames vs copy-last |
|---|---|---|---|---|---|---|---|---|---|---|
| 100m_150k / setting a | 60 | 0.3366 | 0.3712 | 0.3714 | 0.3759 | 10.5% [9.4, 11.4] | 9.3% [8.1, 10.6] | 9.4% [8.1, 10.7] | 12.9% [11.8, 13.9] | 10.9% [9.8, 12.0] |
| 100m_150k / setting b | 60 | 0.3369 | 0.3706 | 0.3705 | 0.3759 | 10.4% [9.3, 11.3] | 9.1% [7.9, 10.4] | 9.1% [7.8, 10.4] | 12.7% [11.7, 13.7] | 10.8% [9.7, 11.9] |
| 1b_300k / setting a | 30 | 0.3129 | 0.3493 | 0.3482 | 0.3714 | 15.8% [14.6, 17.0] | 10.4% [8.4, 12.5] | 10.2% [8.1, 12.2] | 17.2% [16.0, 18.6] | 15.4% [14.0, 16.9] |
| 300m_300k / setting a | 60 | 0.3314 | 0.3651 | 0.3660 | 0.3759 | 11.8% [10.8, 12.9] | 9.2% [7.9, 10.5] | 9.5% [8.1, 10.7] | 14.1% [13.0, 15.2] | 11.7% [10.4, 13.0] |
| 300m_300k / setting b | 60 | 0.3327 | 0.3652 | 0.3659 | 0.3759 | 11.5% [10.5, 12.5] | 8.9% [7.5, 10.1] | 9.1% [7.7, 10.3] | 13.8% [12.7, 14.9] | 11.4% [10.1, 12.7] |

Same 30 clips (setting a), to compare sizes: criterion 1 reduction 100M 9.7%, 300M 11.4%, 1B 15.8%; criterion 2 reduction 9.1%, 9.0%, 10.4%.

**Criterion 1 (>= 15% below copy-last, best setting):** the primary 300M gives **11.8%** (setting a; setting b 11.5%): **not met**. 100M: 10.5%
(not met). **1B: 15.8% [14.6, 17.0]** on 30 clips and setting a only: the point estimate meets the bar, the CI straddles 15%, and it is a
partial sample and not the pre-registered primary model. Reduction grows with size (9.7% to 11.4% to 15.8% on identical clips).
**Criterion 2 (>= 5% below shuffled actions, states re-integrated):** **met for every model and setting** (8.9% to 10.4%, every CI lower bound
above 7.4%). The paper-literal arm (true states) gives the same reductions, so on these clips the state input does not confound the test.
Action conditioning works on our arm. Camera setting (a) and (b) differ by 0.3 points or less: the view ids do not matter here.
**Criterion 3: not evaluated** (no failed rollouts; Tuesday's recordings).

**Gate A verdict: NO-GO for the pre-registered primary (300M @ 300k), criterion 1 fails (11.8% vs 15%) while criterion 2 passes. The 1B model
is borderline-GO on criterion 1 and GO on criterion 2 on a reduced sample (30 clips, setting a); that is not a verdict until it is
re-run on the full sample.** Per the owner's rule the best case would have been "conditional GO" pending criterion 3.

Caveat on criterion 1: copy-last is a strong baseline here (a 5 Hz step moves the scene little); the bar was pre-registered, not tuned.

## Deviations from the protocol

1. **Sample size.** Planned 200 windows per setting. The rollout took 29 s per clip for 300M even with the KV cache (54 s uncached), so the plan
   would not fit the 6 h limit. Used the first 60 windows (3 per episode, all 25 episodes) for 300M and 100M in both settings, and the first 30
   windows (setting a only) for 1B. The 1B run has no setting b. Fixed before any number was seen (the first 60-clip result came afterwards).
2. **Rollout path.** KV-cache rollout (released option) instead of the uncached one; verified equal to bf16 noise. Prefix set: a prefix counts only
   if >= 10 target frames follow it (t = 0..4), as in the released eval.
3. **Camera ids.** Primary (a) = ids assigned by the training config's name-matching rule (front 6, wrist 2); the task's "unmapped id" does not exist.
4. **GPU.** RTX PRO 4000 Blackwell, not a 3090; torch 2.7.1+cu128 (the first launch with cu126 had no sm_120 kernels). Launch failed twice
   (unset variable; wrong torch build) and one eval bug (list vs tensor) was fixed on the live instance; the runner scripts on the instance were
   copied by hand, final versions are committed.
5. **Gate B''** (latency on the rented GPU) was **skipped**: Gate A finished at 19:34 UTC, 9 minutes before the 19:43 deadline, below the 15-minute
   buffer. Gate A results were fetched (sha256 verified) and the instance destroyed.
6. **Leakage** treated as none (owner), noted above.

## Recommended next step

Not a fine-tune of the 300M model yet: the pre-registered gate fails for it. Options, in order: (1) re-run 1B on the full 200 windows in both
settings (about 51 s per clip, roughly 3 h on this GPU, about $1; the one run that could turn this GO) and then decide; (2) criterion 3 on
Tuesday's failure recordings for whichever size passes; (3) staying on W2 / LeWM for G7 (no GPU time spent) is the default until (1) says the
1B model clears 15% on the full sample. An off-board 1B world model needs a CUDA machine on the LAN; its latency on a desktop GPU is not measured
(Gate B'' skipped).


