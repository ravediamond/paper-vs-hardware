"""WS-167: La-JEPA (arXiv 2611.08156) policy, size S, implemented from the paper (the authors' repo is empty).

Two diffusion transformers trained from scratch on frozen V-JEPA 2.1 features: a World DiT over observation + future-feature
tokens, an Action DiT over the action chunk. At every layer the action queries read the World DiT's observation keys/values next
to their own. Mask (paper section 3.1): obs -> obs; future -> obs + future; action -> obs + action. Joint flow matching,
L = MSE(v_f, z_fut - eps_f) + MSE(v_a, A - eps_a). Inference drops the future tokens, tau_ctx = 0, Euler steps.

Choices the paper does not fix are listed in journal/workstack/notes/WS-167.md (section D). Nothing here sets a bar.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    width: int = 384            # Table 9, size S
    depth: int = 12
    heads: int = 6
    mlp: int = 4
    feat_dim: int = 1024        # V-JEPA 2.1 ViT-L
    act_dim: int = 6
    state_dim: int = 6
    horizon: int = 32
    n_cam: int = 2
    obs_grid: tuple = (1, 8, 8)   # (t, y, x) after 2x2 mean pooling of the 16x16 patch grid of a 256^2 frame
    obs_pool: int = 2
    fut_grid: tuple = (8, 4, 4)   # 8 tube-motion grids, 4x4 mean pooled
    fut_pool: int = 4
    text_dim: int = 1024        # FLAN-T5-large
    text_len: int = 32
    n_prop_tokens: int = 4
    p_text_drop: float = 0.1    # Table 9
    p_prop_drop: float = 0.1
    p_cam_drop: float = 0.1
    fut_weight: float = 1.0     # A2 ablation: 0.0

    @property
    def n_obs(self):
        return self.obs_grid[0] * self.obs_grid[1] * self.obs_grid[2]

    @property
    def n_fut(self):
        return self.fut_grid[0] * self.fut_grid[1] * self.fut_grid[2]


# ---------- rotary positions (axial: t, y, x) ----------
def rope_cos_sin(pos, head_dim, base=10000.0):
    """pos: (L, 3) float coordinates (t, y, x). Head dim split into (t,y,x) parts, each even. Returns cos, sin of shape (L, head_dim/2)."""
    d_t = (head_dim // 3) // 2 * 2
    d_y = (head_dim - d_t) // 2 // 2 * 2
    d_x = head_dim - d_t - d_y
    parts = []
    for ax, d in enumerate((d_t, d_y, d_x)):
        inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        parts.append(pos[:, ax:ax + 1].float() * inv[None])
    ang = torch.cat(parts, dim=1)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, h, L, hd); cos/sin: (L, hd/2). Rotate consecutive pairs."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


def grid_positions(grid, pool, t_coord):
    """Token coordinates in units of the original 16x16 patch grid (centre of each pooled cell). t_coord: list over temporal grids."""
    T, Y, X = grid
    ys = (torch.arange(Y).float() + 0.5) * pool - 0.5
    xs = (torch.arange(X).float() + 0.5) * pool - 0.5
    t = torch.tensor(t_coord, dtype=torch.float32)
    tt, yy, xx = torch.meshgrid(t, ys, xs, indexing="ij")
    return torch.stack([tt, yy, xx], dim=-1).reshape(-1, 3)


def timestep_embedding(t, dim, dtype=torch.float32):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    a = t[:, None].float() * 1000.0 * f[None]
    return torch.cat([a.cos(), a.sin()], dim=-1).to(dtype)


class Mlp(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.f1, self.f2 = nn.Linear(d, d * r), nn.Linear(d * r, d)

    def forward(self, x):
        return self.f2(F.gelu(self.f1(x), approximate="tanh"))


def heads_split(x, h):
    B, L, D = x.shape
    return x.view(B, L, h, D // h).transpose(1, 2)


def heads_merge(x):
    B, h, L, hd = x.shape
    return x.transpose(1, 2).reshape(B, L, h * hd)


class Stream(nn.Module):
    """Per-layer weights of one DiT (World or Action): own attention projections, cross-attention to the conditioning, MLP, adaLN."""

    def __init__(self, c: Cfg):
        super().__init__()
        d = c.width
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.normc = nn.LayerNorm(d, eps=1e-6)
        self.cq, self.ckv, self.cproj = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(d, c.mlp)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def cross(self, x, cond, cmask, h):
        q = heads_split(self.cq(self.normc(x)), h)
        k, v = self.ckv(cond).chunk(2, dim=-1)
        o = F.scaled_dot_product_attention(q, heads_split(k, h), heads_split(v, h), attn_mask=cmask[:, None, None, :])
        return self.cproj(heads_merge(o))


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LaJEPA(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        d = c.width
        self.in_obs = nn.Linear(c.feat_dim, d)
        self.in_fut = nn.Linear(c.feat_dim, d)
        self.in_act = nn.Linear(c.act_dim, d)
        self.act_pos = nn.Parameter(torch.zeros(1, c.horizon, d))
        self.cam_emb = nn.Parameter(torch.zeros(c.n_cam, d))
        self.type_emb = nn.Parameter(torch.zeros(2, d))   # 0 obs, 1 future
        self.text_proj = nn.Linear(c.text_dim, d)
        self.prop_proj = nn.Linear(c.state_dim, c.n_prop_tokens * d)
        self.bias_tok = nn.Parameter(torch.zeros(1, 1, d))  # always-visible conditioning token (keeps the key set non-empty)
        self.t_mlp_w = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.t_mlp_a = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.world = nn.ModuleList([Stream(c) for _ in range(c.depth)])
        self.action = nn.ModuleList([Stream(c) for _ in range(c.depth)])
        self.fut_head = nn.Sequential(nn.LayerNorm(d, eps=1e-6), nn.Linear(d, c.feat_dim))
        self.act_head = nn.Sequential(nn.LayerNorm(d, eps=1e-6), nn.Linear(d, c.act_dim))
        for p in (self.act_pos, self.cam_emb, self.type_emb, self.bias_tok):
            nn.init.normal_(p, std=0.02)
        nn.init.zeros_(self.fut_head[1].weight); nn.init.zeros_(self.act_head[1].weight)
        nn.init.zeros_(self.fut_head[1].bias); nn.init.zeros_(self.act_head[1].bias)
        hd = d // c.heads
        obs_pos = grid_positions(c.obs_grid, c.obs_pool, [0.0])
        fut_pos = grid_positions(c.fut_grid, c.fut_pool, [(3 + 4 * j) / 4.0 for j in range(c.fut_grid[0])])
        self.register_buffer("rope_obs", torch.stack(rope_cos_sin(obs_pos, hd)), persistent=False)   # (2, n_obs, hd/2)
        self.register_buffer("rope_fut", torch.stack(rope_cos_sin(fut_pos, hd)), persistent=False)

    # ----- helpers -----
    def rope_world(self, with_fut):
        cos = [self.rope_obs[0]] * self.c.n_cam
        sin = [self.rope_obs[1]] * self.c.n_cam
        if with_fut:
            cos += [self.rope_fut[0]] * self.c.n_cam
            sin += [self.rope_fut[1]] * self.c.n_cam
        return torch.cat(cos, 0), torch.cat(sin, 0)

    def embed_cond(self, text, text_mask, state, drop_text, drop_prop):
        """text (B, Lt, text_dim), text_mask (B, Lt) bool, state (B, state_dim); drop_* (B,) bool. Returns cond tokens and key mask."""
        B = state.shape[0]
        d = self.c.width
        t = self.text_proj(text)
        p = self.prop_proj(state).view(B, self.c.n_prop_tokens, d)
        b = self.bias_tok.expand(B, -1, -1)
        cond = torch.cat([b, t, p], dim=1)
        m = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=state.device),
                       text_mask & ~drop_text[:, None],
                       (~drop_prop)[:, None].expand(-1, self.c.n_prop_tokens)], dim=1)
        return cond, m

    def embed_obs(self, obs):
        """obs (B, n_cam, n_obs, feat_dim) -> (B, n_cam*n_obs, d)"""
        x = self.in_obs(obs) + self.cam_emb[None, :, None, :] + self.type_emb[0]
        return x.flatten(1, 2)

    def embed_fut(self, fut):
        x = self.in_fut(fut) + self.cam_emb[None, :, None, :] + self.type_emb[1]
        return x.flatten(1, 2)

    def world_mod(self, B, n_obs_tok, n_fut_tok, tau_f, device):
        d = self.c.width
        e0 = self.t_mlp_w(timestep_embedding(torch.zeros(1, device=device), d, self.in_act.weight.dtype))      # tau_ctx = 0
        mods_obs = [s.ada(e0)[:, None, :].expand(B, n_obs_tok, -1) for s in self.world]
        if n_fut_tok == 0:
            return mods_obs
        ef = self.t_mlp_w(timestep_embedding(tau_f, d, self.in_act.weight.dtype))
        return [torch.cat([mo, s.ada(ef)[:, None, :].expand(B, n_fut_tok, -1)], dim=1) for mo, s in zip(mods_obs, self.world)]

    # ----- world stream -----
    def run_world(self, x, cond, cmask, key_ok, mods, rope, n_obs_tok):
        """x (B, Lw, d) = [obs tokens ; (future tokens)]. key_ok (B, Lw) bool = camera kept. Returns per-layer obs (K, V) for the action stream."""
        h = self.c.heads
        B, L, _ = x.shape
        cos, sin = rope
        isobs = torch.arange(L, device=x.device) < n_obs_tok
        # obs query -> obs keys; future query -> everything (obs + future); both restricted to kept cameras
        allow = torch.where(isobs[:, None], isobs[None, :], torch.ones(L, L, dtype=torch.bool, device=x.device))
        mask = allow[None] & key_ok[:, None, :]
        mask = mask | torch.eye(L, dtype=torch.bool, device=x.device)[None]  # a dropped token still sees itself (avoids empty rows)
        kv = []
        for s, mod in zip(self.world, mods):
            sh1, sc1, g1, sh2, sc2, g2 = mod.chunk(6, dim=-1)
            q, k, v = s.qkv(modulate(s.norm1(x), sh1, sc1)).chunk(3, dim=-1)
            q, k, v = heads_split(q, h), heads_split(k, h), heads_split(v, h)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
            x = x + g1 * s.proj(heads_merge(o))
            x = x + s.cross(x, cond, cmask, h)
            x = x + g2 * s.mlp(modulate(s.norm2(x), sh2, sc2))
            kv.append((k[:, :, :n_obs_tok], v[:, :, :n_obs_tok]))
        return x, kv

    # ----- action stream -----
    def run_action(self, a_tok, tau_a, kv, obs_ok, cond, cmask):
        h = self.c.heads
        d = self.c.width
        B, H, _ = a_tok.shape
        ea = self.t_mlp_a(timestep_embedding(tau_a, d, self.in_act.weight.dtype))
        x = a_tok + self.act_pos
        n_obs = kv[0][0].shape[2]
        mask = torch.cat([obs_ok, torch.ones(B, H, dtype=torch.bool, device=x.device)], dim=1)[:, None, None, :]
        for s, (ko, vo) in zip(self.action, kv):
            sh1, sc1, g1, sh2, sc2, g2 = s.ada(ea)[:, None, :].chunk(6, dim=-1)
            q, k, v = s.qkv(modulate(s.norm1(x), sh1, sc1)).chunk(3, dim=-1)
            q, k, v = heads_split(q, h), heads_split(k, h), heads_split(v, h)
            o = F.scaled_dot_product_attention(q, torch.cat([ko, k], dim=2), torch.cat([vo, v], dim=2), attn_mask=mask)
            x = x + g1 * s.proj(heads_merge(o))
            x = x + s.cross(x, cond, cmask, h)
            x = x + g2 * s.mlp(modulate(s.norm2(x), sh2, sc2))
        return self.act_head(x)

    # ----- training -----
    def forward(self, b, generator=None):
        """b: dict of tensors on device.
        obs (B, n_cam, n_obs, F) normalised; fut (B, n_cam, n_fut, F) normalised, fut_ok (B, n_cam, n_fut) bool (grid valid);
        action (B, H, A) normalised, act_ok (B, H) bool; state (B, S) normalised; text (B, Lt, T), text_mask (B, Lt).
        Returns dict of losses."""
        c = self.c
        B = b["obs"].shape[0]
        dev = b["obs"].device
        rnd = lambda *s: torch.rand(*s, device=dev, generator=generator)
        # condition dropout (text / proprio 0.1 each; each camera 0.1, at least one view kept)
        drop_text, drop_prop = rnd(B) < c.p_text_drop, rnd(B) < c.p_prop_drop
        cam_drop = rnd(B, c.n_cam) < c.p_cam_drop
        all_dropped = cam_drop.all(dim=1)
        keep_one = (rnd(B) * c.n_cam).long().clamp(max=c.n_cam - 1)
        cam_drop[all_dropped, keep_one[all_dropped]] = False
        cam_keep = ~cam_drop                                                    # (B, n_cam)
        tau_f, tau_a = rnd(B), rnd(B)
        eps_f = torch.randn(b["fut"].shape, device=dev, generator=generator)
        eps_a = torch.randn(b["action"].shape, device=dev, generator=generator)

        cond, cmask = self.embed_cond(b["text"], b["text_mask"], b["state"], drop_text, drop_prop)
        obs_tok = self.embed_obs(b["obs"])
        n_o = obs_tok.shape[1]
        obs_ok = cam_keep[:, :, None].expand(-1, -1, c.n_obs).flatten(1)
        use_fut = c.fut_weight != 0.0
        if use_fut:
            zf = (1 - tau_f)[:, None, None, None] * eps_f + tau_f[:, None, None, None] * b["fut"]
            x = torch.cat([obs_tok, self.embed_fut(zf)], dim=1)
            fut_ok = cam_keep[:, :, None].expand(-1, -1, c.n_fut).flatten(1)
            key_ok = torch.cat([obs_ok, fut_ok], dim=1)
        else:
            x, key_ok = obs_tok, obs_ok   # identical action pathway: futures never reach the actions (mask), they only change the shared obs weights through the loss
        mods = self.world_mod(B, n_o, x.shape[1] - n_o, tau_f, dev)
        x, kv = self.run_world(x, cond, cmask, key_ok, mods, self.rope_world(use_fut), n_o)

        A = b["action"]
        a_noisy = (1 - tau_a)[:, None, None] * eps_a + tau_a[:, None, None] * A
        v_a = self.run_action(self.in_act(a_noisy), tau_a, kv, obs_ok, cond, cmask)
        w = b["act_ok"][:, :, None].float()
        loss_a = (((v_a - (A - eps_a)) ** 2) * w).sum() / (w.sum() * c.act_dim).clamp(min=1)
        out = {"action_loss": loss_a}
        if use_fut:
            v_f = self.fut_head(x[:, n_o:]).view(B, c.n_cam, c.n_fut, c.feat_dim)
            m = (b["fut_ok"] & cam_keep[:, :, None]).float()[..., None]
            loss_f = (((v_f - (b["fut"] - eps_f)) ** 2) * m).sum() / (m.sum() * c.feat_dim).clamp(min=1)
            out["future_loss"] = loss_f
            out["loss"] = loss_a + c.fut_weight * loss_f
        else:
            out["future_loss"] = torch.zeros((), device=dev)
            out["loss"] = loss_a
        return out

    # ----- inference -----
    @torch.no_grad()
    def sample(self, obs, state, text, text_mask, steps=3, generator=None, noise=None):
        """Action chunk (B, H, A), normalised. Future tokens dropped, tau_ctx = 0, Euler from tau_a = 0 to 1."""
        c = self.c
        B, dev = obs.shape[0], obs.device
        no = torch.zeros(B, dtype=torch.bool, device=dev)
        cond, cmask = self.embed_cond(text, text_mask, state, no, no)
        obs_tok = self.embed_obs(obs)
        n_o = obs_tok.shape[1]
        ok = torch.ones(B, n_o, dtype=torch.bool, device=dev)
        mods = self.world_mod(B, n_o, 0, None, dev)
        _, kv = self.run_world(obs_tok, cond, cmask, ok, mods, self.rope_world(False), n_o)   # computed once, cached over the Euler steps
        a = noise if noise is not None else torch.randn(B, c.horizon, c.act_dim, device=dev, generator=generator)
        for i in range(steps):
            tau = torch.full((B,), i / steps, device=dev)
            a = a + (1.0 / steps) * self.run_action(self.in_act(a), tau, kv, ok, cond, cmask)
        return a


def n_params(m):
    return sum(p.numel() for p in m.parameters())
