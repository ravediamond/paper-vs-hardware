# RACE diagnostic on SmolVLA — protocol (WS-086, WS-081 A1; frozen 2026-11-01 before any error is computed)

Paper: RACE, "When to Switch: Reliable Action-Chunk Extension for VLA Models" (2610.05719): within long chunks, action
error concentrates around transitions between manipulation sub-skills and grows sharply with chunk length.
Owner, 2026-11-01 night: "WS-081 A1 (RACE): frozen protocol, offline on the Orin, to its gate."

## Question

On this arm's SmolVLA, is the chunk's error against the demonstrated actions concentrated near sub-skill transitions,
and does that concentration grow with the position in the chunk (the chunk length that is executed)?

## Data (reused, nothing regenerated)

- Chunks: `outputs/smooth/chunks/ep*.npz` from WS-064 gate A (`scripts/smooth_chunking.py chunks`): SmolVLA
  `ravediamond/smolvla_2026-09-24_14-34-40` @ `c1d76f6`, 10 Euler steps, seeded, one H = 50 chunk at every frame
  f = 0, 25, 50, … (f + 50 ≤ T) of the 20 G13b episodes of `ravediamond/so101_bowls_all` (teleop demos), in degrees,
  with the episode's recorded actions.
- Error e(f, i) = mean over the 5 arm joints of |chunk_f[i] − recorded[f + i]| in degrees, i = 0 … 49.

## Sub-skill transitions (label-free, fixed here)

The gripper channel of the recorded actions (leader command): threshold θ = (5th + 95th percentile) / 2 per episode;
a transition is every frame where the command crosses θ (closing = grasp onset, opening = release), crossings closer
than 15 frames merged to the first. Distance δ(t) = frames to the nearest transition. **Near** = δ ≤ 5, **far** = δ > 15
(5 < δ ≤ 15 not used). Episodes with no crossing are excluded and counted.

## Statistic

For each executed length H ∈ {10, 25, 50}: R(H) = mean e over (f, i < H) with f + i near / mean e over (f, i < H) far,
pooled over episodes. 95 % CI: episode bootstrap, 2,000 resamples, seed 0. Rise: R(50) − R(10), same bootstrap.

Control (no model): the hold predictor chunk_f[i] = recorded[f] gives R_hold(H) on the same cells — how much of the
concentration is the demonstration's own motion. Reported beside, and the excess R(H) − R_hold(H) with its CI.
Secondary (descriptive): R by chunk position i in 5 bins of 10; the 6th joint (gripper) separately.

## Decision rule (the card's seed bar, adopted here before any number)

**Concentrated as RACE says** if R(50) ≥ 2 with CI lower bound > 1, **and** R rises with H: R(10) < R(25) < R(50)
(point estimates) with the CI of R(50) − R(10) above 0. **Partly** if only the first holds. **Not seen** otherwise.
The hold control does not change the verdict; it is reported so the reader can see whether the policy or the motion
produces the concentration.

## Cost

$0, Orin CPU, minutes. No arm.
