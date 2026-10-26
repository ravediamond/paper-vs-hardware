"""Gate B: latency of the RoboJEPA encoder and predictor step on a Jetson Orin NX (G0-RJ-robojepa-gate.md). Written, not yet run on the device.

On the Orin (JetPack PyTorch with CUDA), from a clone of facebookresearch/robo_jepa (PYTHONPATH=$PWD):

    python bench_orin.py --encoder /path/vjepa2_1_vitG_384.pt --out results/gateB_orin.json
    python bench_orin.py --encoder /path/vjepa2_1_vitG_384.pt --models-dir /path/robojepa --sizes 100m 300m --steps 150 300 --out ...

Encoder: V-JEPA 2.1 ViT-G/384, fp16, random input 240x320, 1 view and 2 views (one image per view, T = 1), 10 warmup + 50 timed forward
passes with cuda synchronize, p50 / p95 ms, peak GPU memory. Predictor: one step with a filled KV cache (3 context frames of 2 views
prefilled, then one new frame), same warmup / timing, if the checkpoint fits (an out-of-memory is recorded, not fatal).
TensorRT is only tried if eager fp16 misses the 200 ms target by less than 2x (not in this script).
"""

import argparse
import json
import os
import platform
import subprocess
import time

import torch

DEV = "cuda"
PURE_FP16 = False
WARM, ITERS = 10, 50
H, W, DIM = 240, 320, 1664


def stats(ms):
    ms = sorted(ms)
    return {"p50_ms": round(ms[len(ms) // 2], 2), "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 2), "mean_ms": round(sum(ms) / len(ms), 2),
            "min_ms": round(ms[0], 2), "n": len(ms)}


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


def mem_mb():
    """peak allocated on CUDA; on MPS only the current driver allocation after the timed loop (no peak counter)."""
    if DEV == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if DEV == "mps":
        return round(torch.mps.driver_allocated_memory() / 2**20, 1)
    return None


def timed(fn):
    for _ in range(WARM):
        fn()
    sync()
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ms = []
    for _ in range(ITERS):
        t = time.perf_counter()
        fn()
        sync()
        ms.append((time.perf_counter() - t) * 1000)
    return {**stats(ms), "peak_mem_mb" if DEV == "cuda" else "driver_alloc_mb_after": mem_mb()}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def bench_encoder(path):
    from app.robo_jepa.utils import build_encoder

    enc = build_encoder(version="v21", device=DEV, filename=path).eval()   # fp32 weights, fp16 autocast (rotary q/k stay float)
    out = {}
    if PURE_FP16:
        try:                                                             # fp16 weights and inputs, no autocast; fails on the rotary q/k dtype on some builds
            enc = enc.half()
            with torch.no_grad():
                enc(torch.randn(1, 3, 1, H, W, device=DEV, dtype=torch.half))
            out["dtype"] = "fp16 weights, no autocast"
        except RuntimeError as e:
            out["dtype"] = f"fp16 autocast (fp32 weights): pure fp16 failed: {str(e)[:100]}"
            enc = build_encoder(version="v21", device=DEV, filename=path).eval()
    else:
        out["dtype"] = "fp16 autocast (fp32 weights)"
    for views in (1, 2):
        pure = out["dtype"].startswith("fp16 weights")
        x = torch.randn(views, 3, 1, H, W, device=DEV, dtype=torch.half if pure else torch.float32)

        def fwd():
            with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16 if DEV != "cpu" else torch.bfloat16, enabled=not pure):
                return enc(x)
        try:
            out[f"{views}_view"] = timed(fwd)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            out[f"{views}_view"] = {"error": f"OOM {str(e)[:120]}"}
        print(f"encoder {views} view:", out[f"{views}_view"], flush=True)
    del enc
    (torch.cuda.empty_cache() if DEV == 'cuda' else None)
    return out


def bench_predictor(models_dir, size, step):
    from app.robo_jepa.world_model import WorldModel
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    cfg = json.load(open(os.path.join(models_dir, f"wm-{size}", "config.json")))
    cfg["attn_impl"] = "sdpa"
    json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"))
    os.symlink(os.path.abspath(os.path.join(models_dir, f"wm-{size}", f"cooldown_{size}_{step}k.pth.tar")), os.path.join(tmp, "world_model.pth.tar"))
    try:
        wm = WorldModel.load_pretrained(tmp, load_decoder=False, load_encoder=False, device=DEV)
    finally:
        shutil.rmtree(tmp)
    wm.eval()
    emb = torch.tensor([list(wm.model.embodiment_list).index("so-101_joints")], device=DEV)
    V, C = 2, 3
    view_map = torch.tensor([[6, 2]], device=DEV)
    feats = torch.randn(1, C, V, 15, 20, DIM, device=DEV, dtype=torch.float32)
    act = torch.randn(1, C, 64, device=DEV, dtype=torch.float32)
    st = torch.randn(1, C, 64, device=DEV, dtype=torch.float32)
    dt = torch.full((1, C), 0.2, device=DEV, dtype=torch.float32)
    new_f = torch.randn(1, 1, V, 15, 20, DIM, device=DEV, dtype=torch.float32)

    def step_fn():
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16 if DEV != "cpu" else torch.bfloat16):
            _, _, _, cache = wm.forward(feats, act, states=st, camera_extrinsics=None, view_map=view_map, embodiment_ids=emb, dt=dt,
                                        use_cache=True, cache=None, cache_size=0, input_position=0)
            return cache
    # timing the cached step alone: prefill once outside the timed region, then feed one frame
    cache = step_fn()

    def one():
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16 if DEV != "cpu" else torch.bfloat16):
            wm.forward(new_f, act[:, :1], states=st[:, :1], camera_extrinsics=None, view_map=view_map, embodiment_ids=emb, dt=dt[:, :1],
                       use_cache=True, cache=cache, cache_size=min(cache.pointer[0], cache.cache_len), input_position=C)
    res = timed(one)
    del wm, cache
    (torch.cuda.empty_cache() if DEV == 'cuda' else None)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--models-dir")
    ap.add_argument("--sizes", nargs="*", default=[])
    ap.add_argument("--steps", nargs="*", type=int, default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--predictor-device", choices=["cuda", "mps", "cpu"], help="default: --device (MPS fails in F.scaled_dot_product_attention of the predictor: use cpu)")
    ap.add_argument("--skip-encoder", action="store_true")
    ap.add_argument("--pure-fp16", action="store_true", help="try fp16 weights without autocast for the encoder (falls back to autocast, recorded)")
    a = ap.parse_args()
    global DEV, PURE_FP16
    DEV = a.device
    PURE_FP16 = a.pure_fp16
    res = {"host": platform.node(), "machine": platform.machine(), "jetpack": sh("cat /etc/nv_tegra_release | head -1"),
           "torch": torch.__version__, "cuda": torch.version.cuda, "device": DEV, "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else platform.processor() or platform.machine(),
           "power_mode": sh("nvpmodel -q 2>/dev/null | tail -2 | tr '\\n' ' '"), "warmup": WARM, "iters": ITERS,
           "input": f"{H}x{W}, fp16 (encoder dtype in encoder.dtype; predictor: fp16 autocast)", "encoder_weights": os.path.basename(a.encoder), "robo_jepa_commit": sh("git rev-parse HEAD"),
           "encoder": {} if a.skip_encoder else bench_encoder(a.encoder), "predictor_step": {}, "predictor_device": a.predictor_device or DEV}
    DEV = a.predictor_device or DEV
    for size, step in zip(a.sizes, a.steps):
        try:
            res["predictor_step"][f"{size}_{step}k"] = bench_predictor(a.models_dir, size, step)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            res["predictor_step"][f"{size}_{step}k"] = {"error": f"OOM {str(e)[:120]}"}
        print("predictor", size, res["predictor_step"][f"{size}_{step}k"], flush=True)
    res["gate_b_encoder_2view_p50_le_200ms"] = bool(res["encoder"].get("2_view", {}).get("p50_ms", 1e9) <= 200)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
