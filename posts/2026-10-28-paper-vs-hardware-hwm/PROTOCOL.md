# WS-051 · HWM (Zhang et al., CoRL 2026, arXiv 2604.03208) — gate 0 protocol, frozen 2026-10-27 before training

Prompt: `journal/prompts/2026-10-27_hwm_hierarchical.md`, gate 0 (H1–H3). Owner `go vast WS-051` (cap $5).

**Compute.** The high level is trained on W1's *frozen* encoder latents (192-d per frame, precomputed once on the Orin), so the trainable part is an action encoder and a predictor over vectors: it is trained on the Orin, $0. vast.ai is used only if a training run on the Orin exceeds 2 h; then the same script runs there under the $5 cap.

**Data.** `so101_bowls_all_trimmed` @ `4e5c6fda` (local H.264 224 copy, 182 episodes / 61 462 frames), the G1b split (157 train / 25 val, seed 1234). Latents: W1 (`~/wm_data/weights_final.pt`, sha256 `cc9c5f0b…`), fused z_t, every frame. Actions: the 6 commanded joints, normalised with the train-split mean / std (as W1).

**Low level.** W1 unchanged: context = the 3 frames ending at anchor t (C = 3), autoregressive rollout with the executed actions, prediction read at t + h.

**High level (§2.1).** Per training sample: N = 3 waypoints t1 < t2 < t3 in one episode, span S = t3 − t1 ~ U{20, …, 120} frames (≤ 4 s), t2 ~ U(t1+1, t3−1). Action encoder: 2-layer transformer (width 128, 4 heads) over the primitive actions of a chunk (+ learned CLS, positions), CLS → linear → macro-action l ∈ R^32. Predictor F2: 4-layer causal transformer (width 256, 4 heads) over the interleaved tokens (l_t1, z_t1, l_t2, z_t2), predicting z_t2 and z_t3 (teacher forcing, ℓ1). AdamW lr 3e-4, wd 1e-4, batch 256, 20 000 steps, seed 1234. Model selection: last step (no val peeking). Differences from W1: W1 trains encoder + predictor end to end with SIGReg; here the encoder is frozen and only §2.1's high-level parts are trained (that is the paper's setup for the high level).

**Evaluation (val split).** Anchors every 10 frames with t − 2 ≥ 0 and t + 60 inside the episode. For h ∈ {22, 30, 38, 45, 60} frames (0.73, 1.0, 1.27, 1.5, 2.0 s): low-level error = mean |ẑ_{t+h} − z_{t+h}| over the 192 dims (W1 rollout); high-level error = same, with ẑ from one F2 step on (l(a_{t:t+h}), z_t). Mean over anchors, with SE.

**Bars.**
- **H1** a crossover: there is a horizon h* ∈ [1.0 s, 2.0 s] (30–60 frames) with high < low at every evaluated h ≥ h* and high ≥ low at every evaluated h < h* (on the evaluated grid; "high < low at 30 frames already" counts as a crossover ≤ 1.0 s and fails H1's interval).
- **H2** at 60 frames (2.0 s): high-level error ≤ 0.95 × low-level error.
- **H3** macro-actions (their Table 4b, operationalised here): three high-level models trained identically except the action input — (a) learned encoder, (b) concatenated primitive actions (zero-padded to 120 × 6, linear → 32), (c) handcrafted net joint displacement over the chunk (6-d, linear → 32). Metric: cosine between predicted and true latent displacement, cos(ẑ_{t+h} − z_t, z_{t+h} − z_t), averaged over val anchors and the five h, with SE. **Bar: (a) ≥ (c)**; (b) reported.
