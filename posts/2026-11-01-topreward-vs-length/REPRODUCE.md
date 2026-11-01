# Reproduce

## L0 (seconds, CPU)
```bash
python3 l0_check.py
```
Recomputes the success score from the per-episode log-probabilities (`expected/ab.jsonl`, `expected/own.jsonl`), the
pooled and within-policy AUROCs of TOPReward and of episode length, and the reading; bootstrap CIs are read from
`expected/topreward.json`.

## L1 (a ≥ 24 GB GPU, ~2.7 h)
`scripts/remote_topreward.sh` (pinned model and data, anonymous Hub), then
`python scripts/topreward_analysis.py --ab ab.jsonl --own own.jsonl --own-labels own_rollout_labels.csv` beside
`scripts/length_analysis.py`. Status: not re-verified from this folder.

## L2
Not applicable: offline scoring of recorded rollouts.
