# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python3 l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Counts and the verdict branch are recomputed from `expected/labels.csv`
(owner labels at rest, placement residuals; local paths removed).

## L1 / L2

`expected/LAYOUTS.csv` is the sealed layout list (sha256 `49e4d364…`); `scripts/run_s29_session.py` places the bowls
from it and records each rollout; `scripts/s29_score.py` is the frozen scorer (not run: no failures).
