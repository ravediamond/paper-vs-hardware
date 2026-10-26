# VLA-JEPA rotary fix (#4835): does a legacy-trained world predictor still work? (step 4)

Protocol: [`PROTOCOL.md`](PROTOCOL.md) (frozen 2026-10-25, not edited). Checks: [`CHECKS.md`](CHECKS.md). Script:
[`eval_rope_compat.py`](eval_rope_compat.py) `compare`. Rows and summary: [`results.json`](results.json);
exploratory breakdowns: [`exploratory.json`](exploratory.json).

## Setup

- **Model and data:** M1 = `ravediamond/vla_jepa_so101_bowls_trimmed` @ `c65c98b`, trained with the world model on,
  legacy rotary, lerobot `8c920c427`. Data: the 25 val episodes of `split.json` (`so101_bowls_all_trimmed`).
- **Sample:** 8 windows per episode, start frames from `numpy.random.default_rng([0, episode])` uniform over the
  valid range: **200 windows**.
- **Arms:**
  - Each window runs the encoders and Qwen once.
  - The predictor then runs under `legacy` (the pre-#4835 expansion, patched in; parity with `2e88044` 0.0) and
    `fixed` (lerobot `d3793b7` as merged).
  - `copy_last` predicts each next temporal position with the current one.
  - Loss = `_world_model_loss`'s L1 (mean over tokens).
- **Environment:** lerobot-jetson:rc + the `d3793b7` checkout mounted, torch 2.11.0, transformers 5.5.4, Orin NX
  at 25W.
- **M2** (`lerobot/VLA-JEPA-LIBERO`): not run; it does not fit on the Orin's disk (`CHECKS.md` step 1).
- **Scope:** the windows come from the checkpoint's own training distribution, so this is a **compatibility test**
  (does the predictor still do what it learned), not a generalisation test.

## Results (M1, 25 episodes, 200 windows; paired bootstrap over episodes, 2 000 resamples, seed 0)

| | mean L1 (mean over episodes) |
|---|---|
| `legacy` | **1.175** |
| `fixed` | **1.731** |
| `copy_last` | **2.235** |

| Statistic | value | 95 % CI |
|---|---|---|
| **R = fixed / legacy** | **1.473** | [1.462, 1.484] |
| fixed − legacy | +0.556 | [0.547, 0.564] |
| fixed − copy_last | −0.505 | [−0.508, −0.501] |
| legacy − copy_last | −1.060 | [−1.069, −1.051] |

## Bar, applied as written

1. **Instrument:** the 95 % CI of `legacy − copy_last` is [−1.069, −1.051], entirely below 0, so **it holds**.
2. **COLLAPSES:** the 95 % CI of `fixed − copy_last` is [−0.508, −0.501], entirely below 0, so **no**: the fixed
   rotary still beats copy-last.
3. **DEGRADES:** the instrument holds, it does not collapse, R = 1.473 ≥ 1.10, and the 95 % CI of `fixed − legacy`
   ([0.547, 0.564]) is entirely above 0, so **yes**.

**Verdict on M1: DEGRADES.**

Per the protocol's decision: future fine-tunes from a legacy checkpoint pin the legacy rotary (a patch, or a lerobot
commit before `f9e9d73`) and say so in the model card. `UPSTREAM_DRAFT.md` proposes an opt-in
`world_model_rope_legacy: bool` (default False).

## Exploratory (decided in the protocol, labelled exploratory)

- **Episodes where `fixed` is worse than `legacy`:** 25 / 25.
- **By predicted temporal position** (3 positions after the context):

  | | position 1 | 2 | 3 |
  |---|---|---|---|
  | legacy | 1.203 | 1.183 | 1.139 |
  | fixed | 1.733 | 1.708 | 1.752 |
  | copy_last | 2.256 | 2.216 | 2.234 |

  The gap does not grow with distance from the context: the fixed rotary costs about 0.53–0.61 at every position.
- **By window position in the episode** (start / middle / end thirds):

  | | start | middle | end |
  |---|---|---|---|
  | legacy | 1.19 | 1.21 | 1.11 |
  | fixed | 1.73 | 1.74 | 1.72 |
  | copy_last | 2.24 | 2.24 | 2.23 |

**In plain words.** Run with the rotary it was not trained with, the predictor's loss rises by 47 % on its own
training distribution. It still beats copying the last frame, so it has not forgotten everything. For fine-tuning
from a legacy checkpoint, that means the world-model loss restarts about half way to copy-last on the first step,
and the pretrained predictor is no longer a pretrained predictor in the sense its training gave it.
