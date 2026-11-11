"""B2 (D1_PROTOCOL O6): Orin encode latency of the D1 student vs W1; bar: student <= 2 x W1.
D1 student = W1 architecture (arch_check.json: 279 tensors, same keys and shapes as W1), so both are the same graph: ViT-Tiny
(HF ViTModel, 192-d, 12 layers, 3 heads, patch 16, 224 px; stable_pretraining.vit_hf config) on both views, BatchNorm projector
(192-2048-192), mean over views, fuse Linear(192,192) = d1_train.encode_stages. The LeWM package is not installed on the Orin, so the
encode path is rebuilt from that config (random init; latency does not depend on weights). Two independent instances are timed:
'W1' and 'D1_student'. Input: 1 frame x 2 views."""
import json, statistics, sys, time
from pathlib import Path
import torch, torch.nn as nn
from transformers import ViTConfig, ViTModel

def mk():
    cfg = ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3, intermediate_size=768, image_size=224, patch_size=16)
    m = nn.Module()
    m.encoder = ViTModel(cfg, add_pooling_layer=False, use_mask_token=False)
    m.projector = nn.Sequential(nn.Linear(192, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Linear(2048, 192))
    m.fuse = nn.Linear(192, 192)
    return m.cuda().eval()

def encode(model, x):  # d1_train.encode_stages
    b, v = x.shape[:2]
    cls = model.encoder(x.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state[:, 0]
    proj = model.projector(cls).view(b, v, -1)
    return model.fuse(proj.mean(1))

torch.manual_seed(0)
models = {"W1": mk(), "D1_student": mk()}
x = torch.randn(1, 2, 3, 224, 224, device="cuda")
res = {}
for dtype in ("fp32", "fp16"):
    for name, m in models.items():
        med = []
        for rep in range(5):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dtype == "fp16"):
                for _ in range(50): encode(m, x)
                torch.cuda.synchronize(); ts = []
                for _ in range(200):
                    t = time.perf_counter(); encode(m, x); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
            med.append(statistics.median(ts))
        res[f"{name}_{dtype}_ms"] = statistics.median(med); res[f"{name}_{dtype}_reps"] = med
for dtype in ("fp32", "fp16"):
    res[f"ratio_{dtype}"] = res[f"D1_student_{dtype}_ms"] / res[f"W1_{dtype}_ms"]
res["bar"] = "student <= 2 x W1"
res["verdict"] = "met" if max(res["ratio_fp32"], res["ratio_fp16"]) <= 2 else "not met"
res["params_encode_path"] = sum(p.numel() for p in models["W1"].parameters())
Path(sys.argv[1]).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
