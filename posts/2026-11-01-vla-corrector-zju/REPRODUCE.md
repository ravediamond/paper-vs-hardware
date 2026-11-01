# Reproduce

## L0
```bash
python3 l0_check.py
```
Recounts flags and the AUROC from `expected/lvm_episodes.json` and applies the bar.

## L1 (Orin NX GPU, ~3 h)
`scripts/lvm.py features` on the 224 px demo copy and on the rollouts, `train`, `score` (`scripts/PROTOCOL.md`,
deviations 1–2). Status: not re-verified from this folder.

## L2
Not tested (the paper's truncate-and-replan use needs the arm).
