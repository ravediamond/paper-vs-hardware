# VLA-Corrector (ZJU, 2607.01804) latent-vision monitor on our rollouts — protocol (WS-087, WS-081 A2; frozen 2026-11-01 before any score)

Not the register's #9 (VLA-Corrector 2609.06508, Song et al.); same name, different paper (Pan et al., ZJU). Owner,
2026-11-01 night: "WS-081 A2 (VLA-Corrector ZJU): frozen protocol, offline on the Orin, to its gate."

## Question

Does the paper's Latent-space Vision Monitor (LVM) — a residual MLP trained on successful demonstrations only to
predict the short-horizon change of frozen VLA visual latents from the executed action, read online as
E_t = 1 − cos(predicted change, observed change) through a robust event trigger — catch our failed rollouts at the
monitor gates' bar? It is the first monitor here trained like the policy, with no failure in its training data.

## Method (the paper's §3.1–3.3 and App. B–C; deviations from it marked ‡)

- **Encoder (frozen):** the vision path of our bowls SmolVLA (`ravediamond/smolvla_2026-09-24_14-34-40` @ `c1d76f6`):
  LeRobot's `prepare_images` then `vlm_with_expert.embed_image`, tokens mean-pooled per camera ‡ (the paper does not
  state its pooling), top and wrist concatenated → Z_t. Frames as in the 224×224 H.264 demo copy (area-scaled from
  640×480); rollout frames get the same transform so demos and rollouts share one input path ‡.
- **Corrector M_φ:** action → linear embedding, concatenated with Z_t, 4 residual MLP blocks of width 2048; target
  ΔZ*_{t+k} = Z_{t+k} − Z_t with **k = 10** frames (the paper's deployed "h1-k10"); a_t = the recorded action at t
  (6-d). Loss ‖ΔẐ − ΔZ*‖² + β(1 − cos), β = 1 ‡ (not stated). AdamW, lr 3·10⁻⁴, wd 10⁻⁴, cosine to 0.01·lr,
  batch 512, ≤ 30 epochs, early stopping patience 5 on the validation loss.
- **Training data: demonstrations only** — `ravediamond/so101_bowls_all_trimmed` 224 copy @ `4e5c6fda`, 157 train /
  25 val episodes (the world-model split `wm_train/split.json`). No rollout enters training (AGENTS.md §1 holds
  trivially: no failed action is imitated).
- **Online score:** E_t = 1 − cos(M_φ(Z_t, a_t), Z_{t+k} − Z_t) for every frame t with t + k ≤ T, a_t the recorded
  commanded action of the rollout.
- **Trigger (App. B.2):** window w = 15 of recent E, Me = median, MAD; T_on = Me + 3·MAD, T_off = Me + 2·MAD; counter
  +1 when E_t > T_on, reset when E_t < T_off; **interrupt when the counter reaches p = 5**. The first w frames only
  fill the window. An episode is **flagged** if at least one interrupt occurs.

## Test set

The 50-episode monitor set every monitor gate used (`g16b_episodes.csv`, `in_g16_set`): 42 successes, 8 failures, five
policies (act, smolvla, smolvla_vlacpp, smolvla_vlacpp_fa, vla_jepa), from `ravediamond/so101-bowls-policy-rollouts`
@ `bad1c8ea363789f5f55f9098e775f7361a0f019f`, operator labels.

## Bar (the monitor gates' bar, unchanged)

**Pass** if ≥ 5 / 8 failures are flagged **and** ≤ 2 / 42 successes are flagged (≤ 1 / 20 false interventions).
Otherwise fail. Wilson 95 % intervals on both counts.

## Secondary (descriptive, do not change the verdict)

- AUROC of the episode score max_t E_t (failure = positive), with a 2,000-resample bootstrap CI.
- Shuffled-action control: the same episodes with the action sequence replaced by another episode's of the same
  policy (seed 0); the flag counts and AUROC.
- Among flagged failures, whether the first interrupt comes before the failure onset
  (`shared/gate/data/2026-10-23_onsets_vlm/onsets.csv`).
- Interrupts per minute on successes (how often it would truncate a good chunk).
- Paper's own effect (OGG recovery, success rates) is not tested: offline only, no arm.

## Cost

$0: Orin GPU (feature extraction, training), no arm, no vast.

## Deviations

1. (2026-11-01, before any feature of a rollout or any E was computed; reason: compute) Latents at every 2nd frame
   (15 Hz) instead of every frame: the one-episode smoke test took ~6 min per demo on the Orin at 25 W (182 demos
   ≈ 18 h). k stays 10 frames (5 samples); the trigger's w = 15 and p = 5 count samples, so they span 1 s and 0.33 s
   instead of the paper's 0.5 s and 0.17 s at 30 Hz. Interrupt frames are converted back to 30 Hz frames for the onset
   comparison.
2. (2026-11-01, before any rollout feature or E; reason: compute) The vision encoder and connector run in bfloat16:
   fp32 took 29.5 s per 32 frame pairs on the Orin at 25 W, bf16 5.5 s, and the pooled latents of the two agree to
   cosine ≥ 0.99999 on a test batch. The two demo episodes extracted in fp32 were set aside and redone.
