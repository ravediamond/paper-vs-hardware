# G0.5 on an Orin NX: 8.7 s per chunk on this image, cause not found yet

Post: https://shissorobotics.com/blog/2026/11/10/g05-claims-vs-so101/

Release claim: not tested. On this rig: unclear.

What is here:
- `scripts/s2_latency.py`: 5 warm-up and 50 timed `predict_action_chunk` calls per system (System 2, then System 1) on frames cycled from two earlier SO-101 rollouts; writes the median, p90, min, max and peak torch memory.
- `scripts/s2_swap_probe.py`: its `frames` subcommand builds the frames file (`outputs/s2/frames.npz`).
- `expected/a2_latency.json`: the saved result (System 2 median 8,727.9 ms, System 1 median 6,222.6 ms, MAXN_SUPER, 50 calls).

The script keeps only the summary numbers, not the 55 individual call times per system, so none are stored.

Run (container `lerobot-jetson:0.7.0-jp6.2`, LeRobot `main` at `d490e472a5d1a207cfe0255caaaf74b6d66e4656` on `PYTHONPATH`):

    python scripts/s2_latency.py --policy lerobot/g05_so101 \
      --revision 59119185170fa3e08a43a508d8255a7b6d96ecc3 --frames outputs/s2/frames.npz \
      --out expected/a2_latency.json --power "$(nvpmodel -q | head -1)" --stream-bf16

The image has neither `causal_conv1d` nor `flash-linear-attention`; whether that explains the time is not shown.
