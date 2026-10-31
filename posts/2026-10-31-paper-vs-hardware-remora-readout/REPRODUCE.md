# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python3 l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Premature / missed rates, flips, the R3 / R1 flip ratio, regression
recall, false regressions and MAE are recomputed from `expected/readouts.jsonl` (the four readouts per query on the 34
test episodes), `expected/target.jsonl` (the error-aware target), the G15 transition labels and the G14 windows; R0's
event rates (G15's own stage machine) and the bootstrap CIs are read from `expected/summary.json`.

## L1: rerun the analysis

Rollouts `ravediamond/so101-bowls-policy-rollouts` @ `bad1c8ea` (public); CPU, ~1 h on an Orin NX (feature extraction
decodes the top camera). `scripts/remora_readout.py features` (G15's evidence, a query every 3 frames), `fit` (dev half
only), `target`, `checks`, `test`. The scripts read G15's split and labels and G14's windows from the research repo's
paths (`shared/gate/data/…`); the same files are in `expected/` (`g15_split.json`, `g15_transition_labels.csv`,
`g14_episodes.csv`). Status: not re-verified from this folder.

## L2

Not applicable: offline readout of recorded rollouts; nothing ran on the arm.
