# Pins

- Data: `ravediamond/so101_bowls_all_trimmed` @ `4e5c6fda`, local H.264 224 copy (182 episodes, 61 462 frames), G1b split (157 train / 25 val, seed 1234).
- Low level: W1 `ravediamond/lewm-w2-so101-bowls` `w1/weights_final.pt` @ `d0336db6` (sha256 `cc9c5f0b…`).
- Latents: `latents.npz` sha256 `c1e86e72…`, `lowlevel.npz` `d2f636b8…` (not published; rebuilt by `scripts/hwm_extract.py`).
- High level: `scripts/hwm_train.py`, seed 1234, 20 000 steps, AdamW 3e-4; vast.ai RTX 3090 (instance 58030359).
