# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Tolerance: the rounding the post uses (given per line).

## L1: rerun the analysis

Pins: lerobot-jetson container for evaluation (torch 2.11.0, JetPack 6.2) or the x86 env of `wm_train/README.md`; transformers 5.19.0 (the saving version: W1 loads strictly only with it); dataset `ravediamond/so101_bowls_all_trimmed` @ `4e5c6fda`; weights `ravediamond/lewm-w2-so101-bowls` @ `d0336db6`, folder `w1/` (**private until gate 2**).

```bash
hf download ravediamond/lewm-w2-so101-bowls --revision d0336db6 --include 'w1/*' --local-dir weights
hf download ravediamond/so101_bowls_all_trimmed --repo-type dataset --revision 4e5c6fda --local-dir data
python scripts/eval_action_sensitivity.py --help   # K = 1..30 open loop vs copy-last
python scripts/g1b_range.py --help                 # G1b top-1 of 8 candidates
```
Status: **pending visibility** (W1 weights private). Training (~$3.30 on vast) is documented in the source README, not rerun.

## L2: rerun on hardware

The robot-side protocol (episodes, bars, controls) is in the post. Not re-verified by this repo.
