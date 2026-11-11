# WS-167 · La-JEPA S on the bowls data: PROTOCOL (draft for owner confirmation, 2026-11-06)

Status: **not frozen.** Every bar in §5 was proposed in chat on 2026-11-06 (Claude draft) and is not a bar until the owner
confirms it (`go WS-167`). Frozen before any number: no number of this study exists yet. Source: `journal/prompts/2026-11-06_paper_scan_lajepa_gawm_so101bench.md`
Part A, paper `~/workspace/papers/World models/2611.08156v1-la-jepa.pdf`, memo `journal/papers/scan-2026-11-06/MEMO.md` §1.

## 1. Question
Does a compact JEPA world-action policy trained from scratch in the feature space of a frozen V-JEPA 2.1 ViT-L (La-JEPA S,
0.10 B trainable) match SmolVLA on our bowls data, does its future-prediction loss matter at this data size, and does it fit the
Orin NX latency budget?

Claims tested (paper numbers, memo §1): C1 capacity at 0.10 B trainable (Tables 1–2, p. 6), C2 latency (Table 15, p. 24),
C4 future loss matters though no future is predicted at inference (§4, Appendix A).

## 2. Data
- Dataset `ravediamond/so101_bowls_all_trimmed` @ `4e5c6fda8dc5ce2252214bc086181d3df8b594b0`: 182 episodes, 61,462 frames, 30 fps, robot type
  `so_follower` (`data/FACTS.json` keys `so101_bowls_all_trimmed.revision`, `.episodes`, `.frames`, `.fps`, `.robot_type`). Two cameras (top, wrist).
- Split (fixed, same for every policy): held-out = the 10 episodes with the highest index (172–181 of 0–181); train = the other 172.
- Seeds: three if the budget allows, otherwise one, stated in the result.
- Code: `https://github.com/mt-clemente/lajepa` was **empty on 2026-11-06** (clone has no commits). See §7 open item O1.

## 3. Metric
- Held-out first-action MAE: for every frame of the 10 held-out episodes, the policy predicts a chunk from the observation at that frame;
  take the first action; mean absolute error against the recorded action, in joint degrees (gripper in the dataset's units),
  averaged over the 6 action dimensions and over frames. Same normalisation for every policy.
- Aggregation: per-episode mean, then ratio of means over episodes.
- Bootstrap: 1,000 resamples (A1 wording), seed 0, unit of resampling = episode (10 episodes).
- Latency (A3): median of 100 calls after 20 warm-ups, batch 1, FP16, MAXN_SUPER + jetson_clocks; peak RAM.
- Arm (A4): successes of 20 trials per policy, Wilson 95 % intervals.

## 4. Baselines and controls
- SmolVLA on the same 10 held-out episodes, same normalisation (arm result 9/10, 238 ms per chunk in vla.cpp, EXPERIMENTS.md row 135).
  Its held-out first-action MAE is not in the repo yet: it is computed in Step 2 on the same frames (O2).
- Identity null: first action = current state (the recorded `observation.state` at that frame), same frames, same metric. Reported next to every ratio.
- Ablation (same code, same seed): La-JEPA S with future loss weight 0.
- Same-set rule: every comparison uses the same 10 episodes and the same frames.

## 5. Bars (verbatim from the prompt; each is pass / fail / not proven)
- A1 offline (vast, cap $15): La-JEPA S trained on bowls (train split) reaches held-out first-action MAE ≤ 1.10 × SmolVLA's held-out first-action MAE on the same 10 episodes,
  same normalisation (joint degrees, 30 fps), bootstrap CI over episodes (1,000 resamples) excluding 1.10. Three seeds if the budget allows, else one and say so. *(prompt Part A, Step 0)*
- A2 ablation (same run, $0 extra): La-JEPA S with future loss weight 0 (everything else identical) has held-out first-action MAE ≥ 1.05 × the full model's, CI excluding 1.00 —
  the paper's C4 on our data. If A2 fails, C4 does not transfer at this data size: that is the finding, not a bug. *(prompt Part A, Step 0)*
- A3 Orin latency (offline, $0): full policy call (two 256² cameras × 2 frames through V-JEPA 2.1 ViT-L, cached instruction embedding, S policy, 3 denoising steps, FP16,
  MAXN_SUPER + jetson_clocks, batch 1, median of 100 calls after 20 warm-ups, peak RAM) ≤ 238 ms (SmolVLA parity). Report the encoder alone as a separate line. *(prompt Part A, Step 0)*
- A4 arm (go arm): 20 trials on the bowls layouts of WS-163's protocol, alternated with 20 SmolVLA trials (ABAB), same lighting (curtains closed); successes ≥ SmolVLA − 2
  (non-inferiority); Wilson CIs. Judge at rest, sentinel safety (AGENTS.md §1). *(prompt Part A, Step 0)*

## 6. Checks before numbers
Two held-out episodes first: frame counts and action shapes (6 dims, 30 fps); one hand-computed first-action MAE on 5 frames; identity-null MAE of the same frames;
encoder feature shape for a 256² two-frame clip.

## 7. Deviations (empty) and open items
Deviations: none.
- O1 (owner): the authors' repo is empty, so Step 1 needs either the code (check again before `go WS-167`) or our own implementation from the paper (Appendix B/E, Tables 9 and 16).
  Training at about 20 k updates and batch 32 comes from the memo's feasibility estimate, not from a bar; it is not fixed here.
  **Answer (owner-delegated to agent recommendation 2026-11-07): implement from the paper** (Appendix B/E, Tables 9 and 16); the authors' repo is empty. Reason: waiting on an empty repo has no end date; an implementation from the paper is what the study measures, and any deviation from the paper's recipe is recorded in §7. Not a bar. The implementation is started by the main session, not by this edit.
- O2: SmolVLA's held-out first-action MAE needs a checkpoint evaluated offline on the 10 episodes; which SmolVLA checkpoint is the reference is for the owner to name.
  **Answer (owner-delegated to agent recommendation 2026-11-07): `smolvla-so101-bowls-020000`.** Reason: the checkpoint the arm trials of this project already use (WS-064 gate B; a GGUF of it is in vla-models/, `smolvla-so101-bowls-020000.gguf`, found 2026-11-07); a fixed checkpoint, evaluated offline on the same 10 held-out episodes. Open for Step 2: the offline MAE needs the PyTorch weights, not only the GGUF; if they are absent, that is reported then, not worked around. Not a bar. Bars A1-A4 stay as confirmed on 2026-11-07.
- O3: the prompt gives 1,000 bootstrap resamples; the frozen-protocol template says 2,000. 1,000 is used (prompt wording).

## 8. Exploratory
Anything chosen after data exists (LoRA on the last 8 encoder blocks, denoising-step count beyond 3, other sizes B/L) is labelled exploratory.
