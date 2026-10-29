# WS-047 · LeJEPA identifiability (Klindt, LeCun, Balestriero, arXiv 2605.26379) — gate 0 protocol, frozen 2026-10-27

Frozen before any latent is computed. Prompt: `journal/prompts/2026-10-27_lejepa_identifiability.md` (gate 0).

Models: W1, W2_best, W2_final from `ravediamond/lewm-w2-so101-bowls` @ `d0336db6` (public). Data: `ravediamond/so101_bowls_all_trimmed` @ `4e5c6fda`, G1b split (`paper-vs-hardware/.../leworldmodel/scripts/split.json`: 157 train / 25 val episodes, seed 1234).

Measurement:
1. Encoder latent h per frame (each model's fused encoder output z_t, the same latent G1b ranks with), every frame of the 25 val episodes; train-split frames subsampled every 3rd frame for fitting.
2. z = the 6 joint positions in `observation.state`. Ridge h → z, α by 5-fold CV grouped by episode on train only (grid 10^-3 … 10^3); fit on train; R² per joint on val; mean R²; anisotropy = max − min per-joint R².
3. ε = ‖Cov(h) − I‖_F / √n on val latents (after the model's own output normalisation, none added); δ = E‖h(x_{t+1}) − h(x_t)‖² − 2(1 − ρ̂)·tr(Cov(h)), ρ̂ = mean lag-1 autocorrelation of h's coordinates on val. ρ̂ is estimated from the data, not designed: indicative only.
4. G1b top-1 per model = best over d of `pair.<model>.d*.top1` in `expected/g1_investigation.json` (one protocol for all three models: W1 0.6117, W2_best 0.6145, W2_final 0.6529). Spearman(mean R², G1b top-1) over n = 3: a sighting, not a test.

Bars:
- **B1** W1 mean R²(h→z) < 0.80.
- **B2** W1 anisotropy (max − min per-joint R²) ≥ 0.20.
- **B3** the model with the higher mean R² has the higher G1b top-1 (direction only; with three models: Spearman > 0, and the top-R² model is the top-G1b model).

Not tested by gate 0: the theorem's world-side premises (independent, stationary, Gaussian, OU latents), which teleop data violates; gate 0 measures the predicted consequence only.

## Amendment 1 (2026-10-27, before any latent is computed)

The W2 / W2_best / W2_final checkpoints of the 10-13 investigation are not on the Hub (`lewm-w2-so101-bowls` holds only `w1/` and `w2prime/`) nor on the Orin, so they cannot be encoded. Models used instead:
- **W1** — local `~/wm_data/weights_final.pt`, sha256 `cc9c5f0b…` = Hub `w1/weights_final.pt` @ `d0336db6`. G1b top-1 = best cell of `leworldmodel/expected/g1b_range_w1.json` = **0.6046** (d30_H30), same G1 protocol as below.
- **W2′** — Hub `w2prime/weights_best.pt` @ `b5bd6036`. Its encoder is a **frozen DINOv2 ViT-S/14** (patch tokens; not LeJEPA, not trained on our data): it is a comparison point, not a second instance of the theorem's learner. h = mean-pooled patch tokens per view, the two views concatenated (768-d). G1b top-1 = best cell of `dino-wm/expected/g1_final_w2prime_best.json` = **0.6458** (d10_H10).
B3 therefore compares n = 2 models (direction only). The B3 values from the original protocol (0.6117 / 0.6145 / 0.6529, g1_investigation) are no longer used. B1 and B2 unchanged. For W2′, ε and δ are reported but not interpreted.
