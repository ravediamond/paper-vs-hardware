"""WS-167 CPU self-test of the attention mask and the inference path (tiny config; run: python3 selftest.py). Prints PASS lines; exits 1 on failure."""
import os, sys, torch
from lajepa import Cfg, LaJEPA, n_params

torch.manual_seed(0)
c = Cfg(width=96, depth=2, heads=2)
m = LaJEPA(c).eval()
for p in m.parameters():            # break the zero init so every path carries signal
    if p.ndim >= 1 and float(p.abs().sum()) == 0.0: torch.nn.init.normal_(p, std=0.05)
B = 2
obs = torch.randn(B, 2, c.n_obs, 1024); fut1 = torch.randn(B, 2, c.n_fut, 1024); fut2 = torch.randn_like(fut1)
st = torch.randn(B, 6); tx = torch.randn(B, 32, 1024); tm = torch.ones(B, 32, dtype=torch.bool); no = torch.zeros(B, dtype=torch.bool)
ok = True

def world(fut):
    cond, cm = m.embed_cond(tx, tm, st, no, no)
    o = m.embed_obs(obs); n = o.shape[1]
    x = torch.cat([o, m.embed_fut(fut)], 1) if fut is not None else o
    keyok = torch.ones(B, x.shape[1], dtype=torch.bool)
    mods = m.world_mod(B, n, x.shape[1] - n, torch.full((B,), 0.7), x.device)
    return m.run_world(x, cond, cm, keyok, mods, m.rope_world(fut is not None), n), cond, cm, n

with torch.no_grad():
    (xa, kva), cond, cm, n = world(fut1)
    (xb, kvb), _, _, _ = world(fut2)
    (xc, kvc), _, _, _ = world(None)
    same = all(torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5) for a, b in zip(kva, kvb))
    print("PASS" if same else "FAIL", "obs K/V do not depend on the future tokens (mask obs->obs only)"); ok &= same
    same = all(torch.allclose(a[0], b[0], atol=1e-5) for a, b in zip(kva, kvc))
    print("PASS" if same else "FAIL", "obs K/V identical with and without future tokens (inference drops them, K/V cacheable)"); ok &= same
    diff = not torch.allclose(xa[:, n:], xb[:, n:], atol=1e-4)
    print("PASS" if diff else "FAIL", "future tokens do depend on their own input"); ok &= diff
    a = torch.randn(B, c.horizon, 6); tau = torch.full((B,), 0.3); okobs = torch.ones(B, n, dtype=torch.bool)
    va = m.run_action(m.in_act(a), tau, kva, okobs, cond, cm); vb = m.run_action(m.in_act(a), tau, kvb, okobs, cond, cm)
    same = torch.allclose(va, vb, atol=1e-5)
    print("PASS" if same else "FAIL", "action velocity is independent of the future input"); ok &= same
    o2 = obs.clone(); o2[:, 0] += 1.0
    (_, kvd), _, _, _ = (lambda: (m.run_world(m.embed_obs(o2), cond, cm, torch.ones(B, n, dtype=torch.bool), m.world_mod(B, n, 0, None, o2.device), m.rope_world(False), n), cond, cm, n))()
    vd = m.run_action(m.in_act(a), tau, kvd, okobs, cond, cm)
    diff = not torch.allclose(va, vd, atol=1e-4)
    print("PASS" if diff else "FAIL", "action velocity does depend on the observation"); ok &= diff
    s1 = m.sample(obs, st, tx, tm, steps=3, noise=torch.zeros(B, c.horizon, 6)); s2 = m.sample(obs, st, tx, tm, steps=3, noise=torch.zeros(B, c.horizon, 6))
    print("PASS" if torch.allclose(s1, s2) and s1.shape == (B, c.horizon, 6) else "FAIL", "sample shape and determinism given noise", tuple(s1.shape))
m.train()
b = {"obs": obs, "fut": fut1, "fut_ok": torch.ones(B, 2, c.n_fut, dtype=torch.bool), "action": torch.randn(B, c.horizon, 6), "act_ok": torch.ones(B, c.horizon, dtype=torch.bool),
     "state": st, "text": tx, "text_mask": tm}
o = m(b); o["loss"].backward()
g = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
print("PASS" if torch.isfinite(o["loss"]) and g > 0 else "FAIL", "training forward/backward finite", float(o["loss"]))
m0 = LaJEPA(Cfg(width=96, depth=2, heads=2, fut_weight=0.0)); m0.train()
o0 = m0(b); print("PASS" if float(o0["future_loss"]) == 0.0 else "FAIL", "fut_weight=0: no future loss")
print("params S:", n_params(LaJEPA(Cfg())))
# encoder dtype path (WS-167 launch 1): half weights + fp32 RoPE-like q/k + half v must run through features.encode (autocast), and must fail without it
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as ft
class MixedEnc(nn.Module):
    def __init__(s):
        super().__init__(); s.qkv = nn.Linear(1024, 3 * 1024).half()
    def forward(s, x):
        B, C, T, H, W = x.shape
        t = x.new_zeros(B, (T // 2) * 256, 1024) + x.mean()
        q, k, v = s.qkv(t).chunk(3, -1)
        q, k, v = (a.view(B, -1, 8, 128).transpose(1, 2) for a in (q, k, v))
        pos = torch.arange(q.shape[2], dtype=torch.float32) * 0.5
        q = q * torch.cos(pos)[None, None, :, None]; k = k * torch.sin(pos)[None, None, :, None]    # fp32 q, k (promotion), fp16 v
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, -1, 1024)
me = MixedEnc().eval(); xx = torch.randn(1, 3, 2, 256, 256).half()
try:
    with torch.no_grad(): me(xx)
    raised = False
except RuntimeError as e:
    raised = "same dtype" in str(e)
print("PASS" if raised else "FAIL", "mixed fp32 q/k + fp16 v reproduces the vast error without autocast"); ok &= raised
yy = ft.encode(me, xx)
print("PASS" if yy.shape == (1, 1, 16, 16, 1024) and torch.isfinite(yy).all() and yy.dtype == torch.float32 else "FAIL", "features.encode runs the mixed-dtype encoder under autocast"); ok &= bool(torch.isfinite(yy).all())
sys.exit(0 if ok else 1)
