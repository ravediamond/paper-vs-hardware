# Execution on a $200 arm: servo dead zone and the command–state gap — protocol (WS-088, WS-082 C1 + C2; frozen 2026-11-01 before any number)

Owner, 2026-11-01 night: "WS-082 C1 + C2: one WS, protocol citing 2609.37334 and 2609.33145, leader vs follower side
per metric, offline to the gate."

## Closest work (read in full, cards in `journal/papers/self-improvement-loop/RELATED_CARDS.md`)

- **2609.37334** (self-compensating VLA): on two AgileX Piper arms the commanded joint target and the next measured
  joint state differ by a mean normalised residual of 32.9 % (new arm) and 35.0 % (one year old); compensating it online
  adds > 30 points of success (Table 6). It measures the gap and fixes it; it does not ask how much of a policy's
  *failure* the gap explains, nor whether small command changes reach the joints at all.
- **2609.33145** (CSDW): command–state discrepancy in teleop demos carries the operator's intent under contact;
  supervising on commands (not states) and weighting by the discrepancy raises insertion success (Table I). It is
  about the demonstrations' side of the gap.
- Smooth MP chunking (2610.39451), RTC (2506.07339), AAC (2604.04161), PACT (2610.28019): smoothness scored on the
  commanded stream only.

What none of them measured, and why this rig can: whether the follower even moves for sub-degree command changes
(their arms track far below a chunk seam; STS3215 hobby servos have an encoder step of about 0.09° (4,096 per turn) and a
position dead band), and whether the gap differs between this policy's failures and successes.

## Sides (LeRobot SO-101 recordings)

- **Command side:** `action`. In the teleop demos it is the **leader** arm's position; in the rollouts it is the
  **policy's command**.
- **Follower side:** `observation.state`, the **follower** arm's measured position. Units: degrees for the 5 arm joints
  (gripper in its own range, reported separately).

## Data

- Demos: `ravediamond/so101_bowls_all_trimmed` @ `4e5c6fda` (the 224 copy's parquet, identical): 182 episodes, 30 fps.
- Rollouts: `ravediamond/so101-bowls-policy-rollouts` @ `bad1c8ea`: 69 episodes (44 success / 25 failure in
  `meta/rollout_labels.csv` at this revision), operator labels; the 50-episode monitor set is a subset and is reported beside.
- Smooth-chunking corrections: `outputs/smooth/chunks` (WS-064 gate A, 20 demo episodes, SmolVLA chunks).

## C1 — the servo dead zone (command side → follower side)

- **Lag L per joint**, fixed first on the demos: the shift 0–10 frames maximising the correlation of Δaction(t) and
  Δstate(t + L).
- **Hold windows:** spans where the command moves ≤ 0.05° per frame on that joint for ≥ 15 consecutive frames.
  **Dead zone d_j** = median |state(end + L) − action(end)| over the hold windows of joint j (follower minus command
  where the command is still: the band the follower settles within). Demos and rollouts separately; bootstrap CI over
  episodes (2,000, seed 0).
- **Sub-step response (descriptive):** for command steps after a hold, the share where the follower moves less than
  half the command step within L + 5 frames, binned by step size (0–0.1, 0.1–0.2, 0.2–0.4, 0.4–0.8, > 0.8°).
- **Correction share:** on the cached chunks at s = 50, the per-step, per-joint change of the projected stream S2 and
  the conditioned stream S3 against raw S0 (degrees, executed rows); share of those changes smaller than d_j.
- **Reading (seed bar of WS-082 C1, adopted here):** **sub-servo** if d_j ≥ 0.3° (CI lower ≥ 0.3°) on ≥ 3 of the 5 arm
  joints in the rollouts; then the correction share says how much of the smoothing never reaches the joints.

## C2 — choosing or tracking (command side vs follower side, failures vs successes)

- **Gap g_t** = |action(t) − state(t + L)| per joint, the follower's distance from what was commanded (rollouts).
- **Grasp window:** from 15 frames before to 30 frames after the first crossing of the commanded gripper below
  (5th + 95th percentile)/2 of that episode's gripper command (the command closing); episodes with no crossing are
  excluded from the window metric and counted.
- **Per-episode statistic:** median g over the window, for the gripper and for wrist_flex + wrist_roll (mean of the two).
- **Primary:** ratio of the failures' median to the successes' median (gripper), bootstrap CI stratified by outcome;
  Mann–Whitney p. **Reading (seed bar of WS-082 C2):** **execution matters** if the ratio is ≥ 2 with CI lower > 1;
  **not seen** if the CI contains 1; otherwise **partial**.
- **Order (descriptive):** among failures with a VLM onset (`2026-10-23_onsets_vlm/onsets.csv`), the gap in the 30
  frames before the onset vs the window statistic of successes.
- **2609.37334's number (descriptive):** per rollout, Σ_t ‖state(t + 1) − action(t)‖₂ / Σ_t ‖action(t) − state(chunk
  start)‖₂ over the 5 arm joints (our reading of their ‖η‖/‖a‖; joint degrees), mean over rollouts, per policy, beside
  their 32.9 % / 35.0 %. The same on the demos (leader vs follower) for CSDW's demo-side gap.

## Cost

$0, Orin CPU, no arm.

## Deviations

(none yet)
