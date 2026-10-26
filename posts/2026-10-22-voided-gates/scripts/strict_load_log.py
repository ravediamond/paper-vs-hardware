import sys, transformers, torch
from so101_harness.w1_live import W1LiveMonitor
print(f"transformers {transformers.__version__}, torch {torch.__version__}")
try:
    m = W1LiveMonitor.from_checkpoint("<W1 weights_final.pt>")
    ck = torch.load("<W1 weights_final.pt>", map_location="cpu", weights_only=False)["state_dict"]
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
