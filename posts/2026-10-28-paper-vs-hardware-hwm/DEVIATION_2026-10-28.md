# WS-051 · deviation note, 2026-10-28 (the protocol and its bars are unchanged)

**First gate-0 run is invalid; re-run on fixed latents.** `scripts/hwm_extract.py` read each frame as `it[cam][0]`.
With one timestamp, LeRobotDataset returns `(3, H, W)`, so `[0]` took the red plane and broadcast it to three
channels after normalisation: W1 saw red-only images. Found before scoring by a consistency check against G0
(`expected/action_sensitivity.json`, re-run today on the same `weights_final.pt`: K = 30 open-loop MSE 0.316, copy-last
0.899, latent std 0.98). On the red-only latents the copy-last MSE at 30 frames was 0.10 (std 0.36 per dim) and W1's
rollout error was 4× copy-last, which no W1 measurement has ever shown. Pixel check: the training loader's frames
match LeRobotDataset index for index on channel 0 and differ on channels 1–2. probe.py's `selfcheck` never saw the
bug because it asks for four timestamps (`(4, 3, H, W)`, where `[0]` is a frame).

The same bug was in WS-047's `gate0_extract.py` (corrected 2026-10-28, B1–B3 still met).

**What the first run showed, kept for the record, not scored:** `results_invalid_redchannel/` (vast 58020852, 20 000
steps × 3 models). H1 no crossover (high level below W1 from 22 frames), H2 ratio 0.137, H3 learned 0.738 ≥
displacement 0.655. Those numbers compare a high level trained on red-only latents with W1 rolled out off its training
distribution, so none of them is a result.

**Re-run.** Same scripts (fix: `one_frame`), same seed, split, steps and bars; fresh latents in `out_v2/`; the high
level retrained on vast under the same `go vast WS-051` (cap $5, first run ≈ $0.35). Before training, the latents
must reproduce G0's scale: copy-last MSE at 30 frames ≈ 0.9 and W1 open-loop error below it.
