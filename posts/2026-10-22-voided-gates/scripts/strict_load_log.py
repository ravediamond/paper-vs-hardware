import sys, transformers, torch
from so101_harness.w1_live import W1LiveMonitor
W1 = sys.argv[1]  # path to W1 weights_final.pt (was a placeholder string; REPRODUCE passes it)
print(f"transformers {transformers.__version__}, torch {torch.__version__}")
try:
    m = W1LiveMonitor.from_checkpoint(W1)
    ck = torch.load(W1, map_location="cpu", weights_only=False)["state_dict"]
    print(f"STRICT LOAD OK on {m.device} ({m.dtype}); checkpoint keys used verbatim: {set(m.model.state_dict()) == set(ck)} ({len(ck)} tensors)")
except RuntimeError as e:
    msg = str(e)
    print("STRICT LOAD FAILED (with and without the _KEY_REMAP retry)")
    import re
    for label in ("Missing key(s)", "Unexpected key(s)"):
        i = msg.find(label)
        if i < 0: continue
        seg = msg[i:].split("\n")[0]
        keys = re.findall(r'"([^"]+)"', seg)
        print(f"{label}: {len(keys)}; first 3: {keys[:3]}; last: {keys[-1]}")
