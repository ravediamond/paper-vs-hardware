# 2026-10-23: failure onsets of the 25 failed published rollouts, VLM-marked (protocol deviation)

Owner's instruction: Gemini instead of owner-marked onsets, logged as a deviation in `wm_train/G7_PROTOCOL.md` and
`jetson/REAP_PROTOCOL.md` before labelling. Script `scripts/onset_vlm_label.py` (G15's coarse-to-fine contact-sheet
search; the model is told the attempt failed and the operator's failure note), `gemini-3.8-flash`, temperature 0,
52 calls. Labels: [`onsets.csv`](onsets.csv) (25 rows, all with an onset). Sheets and answers kept in
`outputs/onset_vlm_sheets/` on the Orin (not committed: video frames). The API key lives in a 600 file outside the repo.

## Spot check (5 episodes, Claude Code on contact sheets to the frame): [`spot_check.csv`](spot_check.csv)

| ep | policy | failure | Gemini | Claude | offset |
|---|---|---|---|---|---|
| 2 | act | missed grasp, retried | 180 | 174 | +6 |
| 18 | smolvla | missed grasp | 349 | 239 | +110 |
| 30 | smolvla_vlacpp_fa | wrong object (released beside) | 201 | 200 | +1 |
| 36 | smolvla_vlacpp_fa | missed grasp | 418 | 129 | +289 |
| 66 | turbovla_v2 | wrong object (went to pink) | 7 | 10 | −3 |

**Median offset +6 frames (0.2 s)**, but not a uniform error: on the two missed-grasp episodes with repeated
attempts Gemini marked a later retry, not the first missed closure (+3.7 s, +9.6 s). Where the failure is one event
(release beside, wrong bowl from the start, a single miss) it is within 6 frames.
