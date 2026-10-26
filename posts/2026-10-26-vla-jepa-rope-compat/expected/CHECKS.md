# VLA-JEPA rotary compatibility: steps 1–3 (checks, then the cost rule)

BRIEF.md steps 1–3. No fixed-rotary loss on real data was computed or printed here: in check 5 the fixed arm ran
only for timing, and its output was discarded. Numbers: [`checks.json`](checks.json). Script:
[`eval_rope_compat.py`](eval_rope_compat.py).

## Step 1: environment

- **How it runs:** the `lerobot-jetson:rc` image, with the fresh checkout `~/contrib/lerobot-rope` @ **`d3793b7`**
  mounted first on `PYTHONPATH`. The image's own tree, `~/gate/lerobot` and `~/venvs/wm` are untouched. The pre-fix
  parent **`2e88044`** is a second worktree (`~/contrib/lerobot-rope-prefix`), used only for the parity check.
- **Versions:** torch 2.11.0, transformers 5.5.4, lerobot 0.6.2 (the `d3793b7` tree). Orin NX, 25W.
- **M1 load:** `ravediamond/vla_jepa_so101_bowls_trimmed` @ `c65c98b`, `enable_world_model=True`.
  - Qwen3-VL and V-JEPA 2 are built from config; the checkpoint fills every tensor.
  - Strict check by name **and value**: 0 missing / 0 unexpected world-model keys (predictor 152 tensors, encoder
    587), and every world-model tensor equals the checkpoint's.
  - The only model key absent from the checkpoint is Qwen's `embed_tokens`, tied to the checkpoint's `lm_head`
    (verified in WS-004).
  - `causal_world_model_context` = False (the saved config). Window = frames t … t+28 at stride 4.
- **M2 (`lerobot/VLA-JEPA-LIBERO` + `HuggingFaceVLA/libero`): skipped.**
  - The checkpoint's root files are 6.16 GB and the LIBERO dataset is 34.9 GB (images in parquet).
  - The Orin had 2.4 GB free, so even the checkpoint does not fit (rule 1.4).
- **Memory:** 5.9 GB peak torch allocation for one window at batch 1. No bf16
  fallback was needed.

## Step 2: measurement checks

| # | Check | Result |
|---|---|---|
| 1 | Parity: legacy patch on `d3793b7` vs the unpatched `2e88044` predictor, same weights, same 6 windows, fp32 | max abs diff of predicted states **0.0** (≤ 1e-6 required) |
| 2 | Determinism: legacy twice on the same 6 windows | identical losses: **True** |
| 3 | Smoke: the M1 predictor under legacy vs a random-init predictor (same config, seed 0, legacy) vs copy-last, same 6 windows | legacy 1.14–1.26; random-init 1.97–2.05; copy-last 2.20–2.28. "Meaningless" (random init) sits near copy-last; the trained predictor under its training-time rotary is clearly below both |
| 4 | Rotary sanity on random tensors (no checkpoint, no data) | legacy vs fixed differ (max abs 7.59). The fixed version rotates each adjacent pair by one angle: pair norms preserved (err 8.9e-16), and the angle equals pos × ω_i (err 1.3e-15). The legacy version does **not** preserve pair norms (err 2.19): the two coordinates of a pair are scaled by different frequencies' cos / sin, so it is not a rotation per pair |
| 5 | Cost: seconds per window on the Orin (encoders + Qwen once, legacy + fixed predictor, copy-last) | median **2.86 s** (max 2.87, n = 4), at 25W |

In plain words:
- The patched legacy path reproduces the pre-fix code exactly.
- Runs are deterministic.
- The instrument separates a trained predictor from an untrained one.
- The fixed rotary really is a per-pair rotation, while the legacy one is not.

## Step 3: PROTOCOL.md's cost rule

M1: 2.86 s/window × 200 windows (25 episodes × 8) = **9.5 min ≤ 2 h → 8 windows per episode**,
as the protocol's default (start frames from `numpy.random.default_rng([0, episode])`, read as "rng(0) per episode
index", uniform over the valid range). M2: not run (step 1).
