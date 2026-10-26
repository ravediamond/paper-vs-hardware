# 2026-10-22: W1-R, D-JEPA's ranking operator on frozen W1 (G1b regret)

Protocol: [`../../D_JEPA_PROTOCOL.md`](../../D_JEPA_PROTOCOL.md) amendment 2 (`10debe4`, committed before any number).
Script: [`../../djepa_operator.py`](../../djepa_operator.py). Numbers: [`w1r.json`](w1r.json).
Run: Orin NX GPU, `lerobot-jetson-wm:rc` + transformers 5.19.0 overlay, 2026-10-22, 998 s, free.
W1 = `weights_final.pt` sha256 `cc9c5f0b…d063` (Hub `runs/20261010T1941_train_55270119`); `scripts/check_strict_load.py`
passed first. Data `ravediamond/so101-bowls-h264-224`, `split.json`.

## Verdict: **fail**

Regret of record (mean over d and 5 seeds): **W1-R 0.2631**, W1 0.2630, action-only −0.0075. Bar: ≤ 0.10 (no) and
≤ action-only + 0.05 = 0.0425 (no). The operator does not change W1's selection.

**Labels are proxy** (executed chunk = success, G1b decoys = failure), so this does not test D-JEPA's claim, which
needs executed outcomes of every candidate from the same start.

## Step 0: reproduction

On 703 val anchors, W1's native regret on the Orin is 0.2635 / 0.2767 / 0.2756 / 0.2577 / 0.2415 (d = 5…30), against
the stored 0.2635 / 0.2767 / 0.2755 / 0.2577 / 0.2415 (`2026-10-12_g1_diag_exploratory.json`). Action-only is identical
to the stored values. **Reproduced.**

## Per distance (H = d)

| d | W1 regret | W1-R regret (mean of 5) | action-only | W1 top-1 (val) | operator top-1 ungated (val) | W1 top-1 on calibration episodes | gate = "never" | exploratory: rank-2 swap oracle regret |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.2635 | 0.2639 | −0.0146 | 0.454 | 0.474 | 0.670 | 4/5 | 0.1542 |
| 10 | 0.2767 | 0.2767 | −0.0079 | 0.508 | 0.516 | 0.837 | 5/5 | 0.1370 |
| 15 | 0.2756 | 0.2756 | −0.0049 | 0.573 | 0.572 | 0.885 | 5/5 | 0.1370 |
| 20 | 0.2577 | 0.2577 | −0.0050 | 0.603 | 0.609 | 0.904 | 5/5 | 0.1216 |
| 30 | 0.2415 | 0.2415 | −0.0050 | 0.606 | 0.617 | 0.911 | 5/5 | 0.1149 |

## Reading

1. **The gate never fires (24 of 25 operators).** τ_R is calibrated on held-out *train* episodes, and W1 already
   ranks those well: top-1 0.67–0.91 against 0.45–0.61 on the val episodes. W1 was trained on those episodes, so the
   operator learns and is calibrated where W1 is near its best, and calibration finds nothing to correct.
2. **Ungated, the operator adds little:** val top-1 +0.00 to +0.03 over W1.
3. **Exploratory (labelled; computed after the protocol, not part of the bar):** with K = 8 the base ranks are 1/7 apart
   and D-JEPA's correction is bounded by ε = 0.2, so the operator can only swap W1's winner with its rank-2
   candidate. Even a perfect rank-2 swap (oracle) gives regret 0.115–0.154, mean 0.133, above the 0.10 bar. The bar
   is out of reach for this operator on this candidate set, whatever it learns.

What would make a fairer test, all left for an amendment before any run: operator training and calibration data the
world model did not train on (e.g. the G7 rollouts, which also bring executed outcomes), and a larger candidate set so
the bounded correction has room.
