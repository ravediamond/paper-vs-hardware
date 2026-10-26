# Reproduce

## L0: check the numbers (seconds, CPU)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r env/requirements.txt
python l0_check.py
```
Expected: every line `ok`, last line `L0 PASS`. Tolerance: the rounding the post uses (given per line).

## L1: rerun the analysis

Pins: upstream TurboVLA @ `a3365e8`; transformers 4.56.2; path-A checkpoint `ravediamond/turbovla_so101_bowls_faithful_patha` @ `68ecb75d` (weights-only, sha256 `807219eb…`; public since 2026-10-26, DINOv3 License); data `ravediamond/so101_bowls_all_trimmed` @ `4e5c6fda`.

```bash
hf download ravediamond/turbovla_so101_bowls_faithful_patha --revision 68ecb75d --local-dir ckpt
python scripts/gate_upstream.py --upstream <TurboVLA@a3365e8> --ckpt ckpt/turbovla_bowls_faithful_35000.pth --cache <val cache>
```
Status: inputs public since 2026-10-26 (token-free download of the pinned revision checked); the rerun needs the upstream code, a validation cache and a CUDA GPU and is **not yet re-verified** here. Training path A (~$3 on vast) is documented, not rerun.

## L2: rerun on hardware

The robot-side protocol (episodes, bars, controls) is in the post. Not re-verified by this repo.
