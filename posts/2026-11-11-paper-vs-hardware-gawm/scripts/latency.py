"""WS-168 B3: chunk-query latency on the Orin, batch 1, median of 200. Trainable core has random weights (latency does not depend on them)."""
import sys, json, numpy as np, torch
sys.path.insert(0, "/r/projects/gawm")
import gawm_so101 as g
data = g.Data("/s/smoke"); eps = list(data.by_orig)
ns = g.norm_stats(data, eps)
ex = g.DS(data, data.frames_of(eps), ns)[50]
model = g.build_model("gawm", data.tasks).cuda().eval()
def sync(): torch.cuda.synchronize()
def med(f, n=200, warm=30):
    import time
    for _ in range(warm): f()
    t = []
    for _ in range(n):
        sync(); a = time.perf_counter(); f(); sync(); t.append((time.perf_counter() - a) * 1e3)
    return float(np.median(t)), float(np.percentile(t, 90)), float(min(t))
def enc():
    cur, _ = model._pad_inference_views(ex["image"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.backbone.encode_patch_frames([[cur]])
with torch.no_grad():
    full = med(lambda: model.predict_action([ex]))
    e = med(enc)
r = {"full_ms_median_p90_min": full, "encoder_ms_median_p90_min": e, "rest_ms_median": full[0] - e[0], "n": 200, "batch": 1,
     "precision": "as trained: frozen encoder bf16 autocast, trainable core fp32 (not FP16)"}
print(json.dumps(r)); json.dump(r, open("/r/runs/WS-168/latency.json", "w"), indent=1)
