# 2026-10-18 — W1 on the Jetson, no arm: strict checkpoint load (noise floor and timings not run)

Planned (V1.0 review §7 item 3, "Jetson only, nothing moves"): W1's error on a still scene by chunk step against the
G11-B thresholds (2.2 at k ≤ 5, 10.0 at k ≥ 15), W1 ms per check alone and next to vla.cpp, and the G13a no-arm timing
run. **Only the checkpoint load could be tested: no camera (and no arm) was attached to the Jetson** — `lsusb` showed
only the two hubs and the Bluetooth radio; `/dev/video*`, `/dev/v4l/by-id/` and `/dev/serial/by-id/` did not exist.
Every no-arm script opens both cameras before its first trial, so steps 3–5 stop there, and vla-server was not
started.

- `w1_strict_load.txt` — the strict-load result under the image's transformers and under 5.19.0.
- `strict_load_log.py` — the script that printed it (calls `W1LiveMonitor.from_checkpoint`, unchanged).

## Setup

| | |
|---|---|
| Date / host | 2026-10-18, `Orin NX` (Jetson Orin NX 16 GB, L4T R36.4.3, `MAXN_SUPER`) |
| Image | `lerobot-jetson-wm:rc` (`sha256:036f5165d88d…`): lerobot 0.6.2 with OpenCV cameras, hydra 1.3.7, omegaconf 2.3.1, stable-worldmodel @ `21446f1`, torch 2.11.0 (CUDA), transformers 5.5.4; `~/gate/pydeps` mounted for the `vla_cpp` client. The other images lack stable-worldmodel (`lerobot-jetson:rc`) or also hydra/omegaconf (`lerobot-jetson:public`) |
| Extra packages | `transformers==5.19.0` and `tokenizers==0.23.3` (needed by 5.19.0), `pip install --no-deps --target ~/gate/pydeps-w1`, first on `PYTHONPATH`; no image rebuilt. With them the 57 tests still pass, and `vla_cpp.client`, lerobot's OpenCV camera and `AutoTokenizer` import |
| Mounts | `jetson_run.sh` does not mount `~/wm_data`; runs used the same `docker run` flags plus `-v ~/wm_data:~/wm_data:ro` |
| Code | commit `c138251`, unchanged |
| Checkpoint | `<W1 weights_final.pt>` (43.5 MB, not committed) |

Command (step 3 as specified; it stops at the first camera):

```bash
docker run --rm -i --runtime nvidia --privileged --network host --ipc host -v /dev:/dev \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $HOME/gate/pydeps:/opt/gate-pydeps:ro \
  -v $HOME/gate/pydeps-w1:/opt/pydeps-w1:ro -v ~/wm_data:~/wm_data:ro \
  -v $PWD:/work -w /work -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/opt/pydeps-w1:/work/shared/gate:/opt/gate-pydeps lerobot-jetson-wm:rc \
  python scripts/run_g11b_live.py --no-arm --w1 <W1 weights_final.pt> \
    --hold-pose 0,-45,60,0,0,50 --trials 5 --out shared/gate/data/2026-10-18_w1_noise_floor
```

`--hold-pose` is the default rest pose: the arm's present pose cannot be read without connecting the follower, and
the latest rollout state is days old. It did not matter here (the run never reached a trial).

## Results

| check | result |
|---|---|
| W1 strict load, transformers 5.5.4 (image) | **fails**, with and without the `_KEY_REMAP` retry: 192 missing `encoder.encoder.layer.{0..11}.*`, 192 unexpected `encoder.layer.{0..11}.*` (after remap) |
| W1 strict load, transformers 5.19.0 (overlay) | **loads strictly on the first attempt, no remap**: all 279 checkpoint keys match the model verbatim; CUDA, FP16 |
| W1 still-scene error by chunk step (max, p95) vs 2.2 / 10.0 | **not run** (no cameras) |
| W1 ms per check, alone (step 3) and next to vla.cpp (step 4) | **not run** (no cameras) |
| G13a no-arm timing, sync vs async (step 5) | **not run** (no cameras) |

The run died at `OpenCVCamera(/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0).connect()`
(`ConnectionError: Failed to open`), after W1 loaded. Nothing was written to this folder by the script.

## Why the load fails under 5.5.4

The checkpoint was saved with transformers 5.19.0 (as `projects/vla-jepa-world-model/jetson/bench_w1_surprise_orin.py`
notes), whose ViT stores layers as `encoder.layers.N.attention.q_proj.*` (`encoder.` = W1's ViTModel). Transformers
5.5.4 names the same tensor `encoder.encoder.layer.N.attention.attention.query.*`. `w1_live._KEY_REMAP` renames
`layers`→`layer` and `q_proj`→`attention.query`, but does not add the inner `encoder.` level, so the remapped keys
still miss. The code under test was not changed: with the transformers version that saved the checkpoint, the remap
is not needed. (`tests/test_w1_live.py` only exercises the `layers`→`layer` rename on a stand-in model, which is why
the suite passes.)

**Consequence for G11-A.** `bench_w1_surprise_orin.py` (G11-A, 12.78 ms per check) and the retired
`run_g11_phase_b_live.py` used the same incomplete remap with `strict=False`: all 192 ViT encoder tensors stayed at
random initialisation. G11-A's timing still holds (same architecture, same compute), but no latent or surprise value
from those scripts came from the trained encoder.

## To finish steps 3–5

Plug in the C920 (`front_top_camera`) and the icSpring (`wrist_camera`), check that the two `/dev/v4l/by-id/` paths in
`shared/gate/configs/robots/so-101-jetson.json` exist, then rerun step 3 with the command above (into this folder),
and steps 4–5 with the same flags and the `~/gate/pydeps-w1` overlay after starting vla-server on the host
(`~/vla.cpp/build-bench/vla-server ~/vla-models/smolvla-so101-bowls-020000.gguf --bind tcp://127.0.0.1:5555
--flash-attn --mm-prec default`). The arm does not need to be connected; keep the scene
still for step 3.
