# WS-050 · SURVEYOR (NeurIPS 2026 workshop poster) — gates 0–1 protocol, frozen 2026-10-27 before any number

Prompt: `journal/prompts/2026-10-27_surveyor_accept_rule.md`. Scope (README up front): FF-JEPA's diffusion subgoal drafter is not available; this tests (a) whether τ has an operating point on real-robot LeWM latents and (b) whether the accept rule's plan–execution discrepancy works as a failure monitor, with **W1's own imagination as the drafter stand-in**.

Encoder / predictor: W1 (`~/wm_data/weights_final.pt`, sha256 `cc9c5f0b…`), fused z_t; `rel(a, b) = ‖a − b‖₂ / ‖a‖₂`.

## Gate 0 — τ on real latents (69 published rollouts, `ravediamond/so101-bowls-policy-rollouts`, local copy)

- Judge frame per episode: G9c-3's frozen end rule (`judge/g9c3_test.py::judge_frame3`, `g9c2_rule.json`): first still moment after the last gripper release, else the last frame. Judge verdict: G9c-3's frozen colour judge (`g9c_judge.episode_score`, `g9c_config.json`, its τ) at that frame; episodes it marks invalid are dropped and counted.
- **Equivalent pairs:** all pairs (i, j), i ≠ j, of episodes the judge scores *success*: rel(z_i^judge, z_j^judge).
- **Non-equivalent pairs:** (a) success judge frame vs the mid-episode frame (⌊T/2⌋) of every other episode; (b) success judge frame vs the judge frame of every judge-*failure* episode. Reported separately and pooled.
- τ = median over equivalent pairs; report p10 / p90 and the distributions.
- **S1** τ < p10 of the pooled non-equivalent distribution. **S2** τ ≤ 0.40 (paper: 0.20).

## Gate 1 — the accept rule as a monitor (primary 50 episodes of the REAP / judge-τ set: 42 ok / 8 fail)

- Boundaries r every S frames from r = 2 (C = 3 context) while r + S < T; S = 10 (paper) and S = 30 (W1's horizon).
- ŵ_r = W1's imagined latent at r + S from context z_{r−2..r} and the executed actions (W1 rollout, K = S); δ_r = rel(z_{r+S}, ŵ_r).
- Fire at the first r with δ_r > τ (τ from gate 0). Episode score for AUROC = max_r δ_r.
- **S3** ≥ 5/8 failures caught at ≤ 1/20 false interventions (≤ 2/42 successes flagged), reported in `monitors_v2.json`'s shape (caught, fi, AUROC, time to warning vs the Gemini onsets).
- **S4** shuffled control: ŵ taken from a different episode (fixed derangement, seed 1234) at the same boundary index, truncated to common length. Bar: true AUROC ≥ shuffled AUROC + 0.05.
- If τ from gate 0 is undefined (S1 fails), gate 1 still runs with τ = the gate-0 median and the result is reported as "no operating point" for S3.
