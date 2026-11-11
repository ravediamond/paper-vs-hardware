"""WS-167 A3: Orin latency of the full La-JEPA S policy call (PROTOCOL section 5, A3).

  scripts/memcap.sh -m 11G python3 projects/lajepa/bench_orin.py --encoder vjepa21 [--ckpt RUN/model.pt] --out projects/lajepa/results/orin_latency.json

One call = two cameras x two 256x256 frames (uint8) -> normalise -> frozen V-JEPA 2.1 ViT-L (2-frame tube clip per camera, batch of 2 clips) -> 2x2 mean pool ->
channel normalisation -> cached instruction embedding + proprioception -> La-JEPA S, 3 Euler steps (observation K/V once) -> first-action chunk.
FP16, batch 1, median / p95 of 100 calls after 20 warm-ups (cuda synchronised), peak RAM (process RSS high-water and torch CUDA allocator peak).
The encoder alone is timed as a separate line. Latency does not depend on the weight values: without --ckpt the policy is random-init and the
result file says so. Run only on MAXN_SUPER + jetson_clocks (quiet profile otherwise: the file records nvpmodel output), never next to another GPU job.
--encoder stub is a CPU smoke path (random patch embedding), its numbers are meaningless.
"""
import argparse, json, os, resource, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lajepa import Cfg, LaJEPA, n_params
import features as FT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vjepa21", choices=["vjepa21", "stub", "hf"])
    ap.add_argument("--ckpt", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3); ap.add_argument("--compile", action="store_true")
    ap.add_argument("--random-encoder", action="store_true", help="vjepa21: skip the 5 GB weight file (timing only)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    if a.encoder == "stub":
        enc = FT.StubEncoder(dev); enc_name = "stub"
    elif a.encoder == "hf":      # transformers Vjepa2 ViT-L from the cached V-JEPA 2 config, random weights: same ViT-L shapes, timing only
        from transformers import VJEPA2Config, VJEPA2Model
        cfg = VJEPA2Config.from_pretrained("facebook/vjepa2-vitl-fpc64-256"); hf = VJEPA2Model(cfg).to(dev, dt).eval()
        enc = lambda x: hf(pixel_values_videos=x.transpose(1, 2).to(dt), skip_predictor=True).last_hidden_state; enc_name = "hf-vjepa2-vitl-random"
    else:
        if a.random_encoder:
            sys.path.insert(0, os.environ.get("VJEPA2_REPO", "/root/g/vjepa2"))
            from app.vjepa_2_1.models import vision_transformer as vt
            enc = vt.vit_large(patch_size=16, img_size=(256, 256), num_frames=64, tubelet_size=2, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
                               uniform_power=False, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True, n_output_distillation=1,
                               return_all_tokens=True).to(dev, dt).eval(); enc_name = "vjepa2.1-vitl-random"
        else:
            enc = FT.load_vjepa21(dev); enc_name = "vjepa2.1-vitl-pretrained"
    c = Cfg()
    pol = LaJEPA(c).to(dev).eval()
    weights = "random"
    mu = torch.zeros(1, 1, 1024, device=dev); sd = torch.ones(1, 1, 1024, device=dev)
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=dev); pol.load_state_dict(ck["ema"]); weights = a.ckpt
        nz = np.load(Path(a.ckpt).parent / "norm.npz") if (Path(a.ckpt).parent / "norm.npz").exists() else None
        if nz is not None:
            mu = torch.from_numpy(nz["obs_mean"]).to(dev).view(1, 1, -1); sd = torch.from_numpy(nz["obs_std"]).to(dev).view(1, 1, -1)
    if dt == torch.float16: pol = pol.half(); mu, sd = mu.half(), sd.half()
    text = torch.randn(1, c.text_len, c.text_dim, device=dev, dtype=dt); tmask = torch.ones(1, c.text_len, dtype=torch.bool, device=dev)
    state = torch.randn(1, c.state_dim, device=dev, dtype=dt)
    frames = np.random.default_rng(0).integers(0, 255, (2, 2, 256, 256, 3), dtype=np.uint8)   # (camera, frame, H, W, C)
    noise = torch.randn(1, c.horizon, c.act_dim, device=dev, dtype=dt)
    sample = torch.compile(pol.sample) if a.compile else pol.sample
    mean_t = torch.tensor(FT.MEAN, device=dev).view(1, 3, 1, 1, 1); std_t = torch.tensor(FT.STD, device=dev).view(1, 3, 1, 1, 1)

    def encode_only():
        x = torch.from_numpy(frames).to(dev).permute(0, 4, 1, 2, 3).float() / 255.0
        x = ((x - mean_t) / std_t).to(dt)
        with torch.no_grad():
            y = enc(x)
        return y

    def call():
        y = encode_only()
        y = y.view(2, 1, 16, 16, 1024).view(2, 8, 2, 8, 2, 1024).mean(dim=(2, 4)).reshape(1, 2, 64, 1024)
        obs = (y - mu.unsqueeze(1)) / sd.unsqueeze(1)
        with torch.no_grad():
            return sample(obs, state, text, tmask, steps=a.steps, noise=noise)

    def timeit(f):
        for _ in range(a.warmup): f()
        if dev.type == "cuda": torch.cuda.synchronize()
        ts = []
        for _ in range(a.n):
            t = time.perf_counter(); f()
            if dev.type == "cuda": torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        return {"median_ms": statistics.median(ts), "p95_ms": float(np.percentile(ts, 95)), "min_ms": min(ts), "n": a.n, "warmup": a.warmup}

    res = {"encoder_alone": timeit(encode_only), "full_call": timeit(call), "encoder": enc_name, "policy_params": n_params(pol), "policy_weights": weights,
           "dtype": str(dt), "denoise_steps": a.steps, "compiled": a.compile, "device": str(dev),
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           "cuda_peak_alloc_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else None,
           "bar_A3_ms": 238, "power_mode": subprocess.run(["bash", "-c", "nvpmodel -q 2>/dev/null | tail -2 | tr '\\n' ' '"], capture_output=True, text=True).stdout}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
