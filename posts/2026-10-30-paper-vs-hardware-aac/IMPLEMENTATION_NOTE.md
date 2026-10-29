# WS-049 · implementation choices the protocol leaves open (2026-10-29, committed before any gate number)

`PROTOCOL.md` and its bars are unchanged. These fill gaps in it; none was chosen after seeing data.

- **Episodes, policy, seeds:** G13b's (`run_g13b_rtc_offline.py`: `choose_episodes(root, 20, seed 0)`, policy
  `c1d76f6`, noise seed `seed·1 000 003 + e·10 007 + t`). Sample 0 at decision p = t − 25 uses G13b's `noise_prev`
  for boundary t, so it is exactly G13b's executed previous chunk. Samples 1–19: seed `base·31 + 7919·k`.
- **Batched sampling:** the 20 samples are one batched `predict_action_chunk` call (the observation repeated 20 ×).
  The preflight checks sample 0 batched against the single-sample call.
- **Prefix overwrite:** G13b's `overwrite` processor at β = 5 (G13b ran β 5 and 10 with the same verdict).
- **Release** (A1's transport window ends there): the first frame after the grasp where the recorded gripper command
  goes back above 20, the W1-C1 arming level. Grasp is W1-C1's `first_close` (falls below 10 having been above 20).
  Gripper "closed" in a sample: command < `g_open` = 10 (`g9c2_rule.json`), as the protocol says.
- **Entropy:** natural log; 1e-6·I added to each 5×5 covariance (as the protocol); the gripper term is 0 when all
  20 samples agree.
- **AAC's new chunk** at p + s uses G13b's `noise` for boundary t (the same noise as the fixed arm's new chunk), so the
  two arms differ only in where they switch.
- **A2 / A3 rows:** an AAC row exists only where frame p + s exists. Decisions with no grasp in the episode are
  labelled `no_grasp` and stay in A2 / A3, not in A1.
