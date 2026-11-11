"""WS-196 H1/H2: AdaJEPA / OpTTA (own implementation) on the public LeWM PushT checkpoint.

Spec source: OpTTA arXiv 2610.35636 Alg. 1, Eq. 6-8, App. A.1 Table 3 (see PROTOCOL.md section 7 for deviations).
Run: scripts/memcap.sh -m 11G ~/venvs/optta/bin/python projects/optta/optta_pusht.py --method optta --seed 0
"""
import argparse, json, os, sys, time, math
import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor", "lewm"))
sys.path.insert(0, os.path.join(HERE, "vendor", "swm_min"))
from jepa import JEPA  # noqa: E402
from module import ARPredictor, Embedder, MLP  # noqa: E402
from transformers import ViTConfig, ViTModel  # noqa: E402

SCR = os.path.expanduser("~/scratch/WS-196")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BLOCK, HOR, NACT = 5, 5, 10  # frameskip, planning horizon (latent steps), action dim per latent step


def build_model():
    enc = ViTModel(ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3, intermediate_size=768,
                             patch_size=14, image_size=224, qkv_bias=True), add_pooling_layer=False, use_mask_token=False)
    pred = ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=6, heads=16,
                       mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0)
    m = JEPA(enc, pred, Embedder(input_dim=10, emb_dim=192),
             projector=MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d), pred_proj=MLP(192, 2048, 192, norm_fn=nn.BatchNorm1d))
    sd = torch.load(os.path.join(SCR, "ckpt", "weights.pt"), map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    return m.to(DEV).eval().requires_grad_(False)


def prep(img, sev):
    """uint8 HxWx3 -> normalised (1,3,224,224). Darkening: pixels x (1 - sev), before ImageNet normalisation."""
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0)[None]
    x = x * (1.0 - sev)
    return ((x - MEAN) / STD).to(DEV)


@torch.no_grad()
def encode(m, x):
    out = m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    return m.projector(out)  # (1,D)


def rollout_cost(m, z0, zg, acts):
    """Mirror of JEPA.rollout + criterion with 1 history frame. z0,zg (1,D); acts (S,H,10) -> (S,) cost."""
    S, H = acts.shape[:2]
    emb = z0[None].expand(S, 1, -1)
    act = acts[:, :1]
    for t in range(H - 1):
        ae = m.action_encoder(act)
        p = m.predict(emb[:, -3:], ae[:, -3:])[:, -1:]
        emb = torch.cat([emb, p], 1)
        act = torch.cat([act, acts[:, t + 1:t + 2]], 1)
    ae = m.action_encoder(act)
    p = m.predict(emb[:, -3:], ae[:, -3:])[:, -1:]
    last = p[:, -1]
    return ((last - zg) ** 2).sum(-1)


def cem(m, z0, zg, gen, init_mean, n_cand=200, n_iter=10, topk=30):
    mean = torch.zeros(HOR, NACT, device=DEV)
    if init_mean is not None:
        mean[: init_mean.shape[0]] = init_mean
    std = torch.ones(HOR, NACT, device=DEV)
    for _ in range(n_iter):
        cand = torch.randn(n_cand, HOR, NACT, generator=gen, device=DEV) * std + mean
        cand[0] = mean
        with torch.no_grad():
            cost = rollout_cost(m, z0, zg, cand)
        idx = torch.topk(cost, topk, largest=False).indices
        el = cand[idx]
        mean, std = el.mean(0), el.std(0, correction=0)
    return mean, cand, cost  # final-iteration candidates and their costs


def adapt_params(m):
    return (list(m.predictor.transformer.layers[-1].parameters()) + list(m.predictor.transformer.norm.parameters())
            + list(m.pred_proj.parameters()))


def gnorm(gs):
    return math.sqrt(sum(float((g.float() ** 2).sum()) for g in gs if g is not None))


def adapt_step(m, params, buf, z0, zg, cand, cost, k, kmax, method, rho, lr, elites):
    """One Adam step (fresh Adam). AdaJEPA: L_pred only. OpTTA: L_pred + alpha_k L_plan (Eq. 6-8)."""
    z_i = torch.cat([b[0] for b in buf]); a_i = torch.stack([b[1] for b in buf]); z_n = torch.cat([b[2] for b in buf])
    for q in params:
        q.requires_grad_(True)
    p = m.predict(z_i[:, None], m.action_encoder(a_i[:, None]))[:, -1]
    l_pred = ((p - z_n) ** 2).mean()
    g_pred = torch.autograd.grad(l_pred, params)
    grads = list(g_pred)
    alpha = 0.0
    if method in ("optta", "optta30", "pess"):
        c = cand
        if method == "optta30":
            c = cand[torch.topk(cost, 30, largest=False).indices]
        l_plan = rollout_cost(m, z0, zg, c).mean()
        g_plan = torch.autograd.grad(l_plan, params)
        alpha = rho * math.sqrt(k / kmax) * gnorm(g_pred) / (gnorm(g_plan) + 1e-12)
        grads = [gp + alpha * gq for gp, gq in zip(g_pred, g_plan)]
    opt = torch.optim.Adam(params, lr=lr)
    for q, g in zip(params, grads):
        q.grad = g
    opt.step()
    for q in params:
        q.requires_grad_(False); q.grad = None
    return float(l_pred), alpha


def make_env():
    from stable_worldmodel.envs.pusht.env import PushT
    return PushT(render_mode="rgb_array", resolution=224, relative=True)


def episodes(seed, n):
    d = np.load(os.path.join(SCR, "pusht_head.npz"))
    ep, st, s = d["ep"], d["step"], d["state"]
    rows = []
    for e in np.unique(ep):
        idx = np.nonzero(ep == e)[0]
        L = len(idx)
        for j in range(0, L - 26):  # goal = state 25 steps later (goal_offset_steps 25 of LeWM eval config)
            rows.append((idx[j], idx[j + 25]))
    g = np.random.default_rng(seed)
    pick = g.choice(len(rows), size=n, replace=False)
    return [(s[rows[i][0]], s[rows[i][1]]) for i in sorted(pick)]


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def run_episode(m, env, start, goal, args, ep_i, init_sd, params, std_a, mean_a):
    m.load_state_dict(init_sd, strict=True)  # restore pretrained params (task-local adaptation)
    gen = torch.Generator(device=DEV).manual_seed(args.seed * 100000 + ep_i)
    obs, info = env.reset(seed=args.seed * 100000 + ep_i, options={"state": start, "goal_state": goal})
    zg = encode(m, prep(info["goal"], args.sev))
    buf, warm, succ, t_plan, t_adapt, alphas = [], None, False, [], [], []
    img = env.render()
    x = prep(img, args.sev); z0 = encode(m, x)
    for k in range(args.kmax):
        sync(); t0 = time.perf_counter()
        mean, cand, cost = cem(m, z0, zg, gen, warm)
        sync(); t1 = time.perf_counter()
        a_blk = mean[0]  # first latent step, normalised (10,)
        warm = mean[1:].clone()
        acts = (a_blk.view(BLOCK, 2).cpu().numpy() * std_a + mean_a).clip(-1, 1).astype(np.float32)
        for a in acts:
            _, _, term, _, info = env.step(a)
            if term:
                succ = True; break
        img = env.render(); z1 = encode(m, prep(img, args.sev))
        buf.append((z0, a_blk.detach(), z1)); buf = buf[-5:]
        if succ:
            t_plan.append((t1 - t0) * 1e3); break
        if args.method != "frozen":
            t2 = time.perf_counter()
            _, al = adapt_step(m, params, buf, z0, zg, cand, cost, k + 1, args.kmax, args.method, args.rho, args.lr, 30)
            sync(); t3 = time.perf_counter()
            t_adapt.append((t3 - t2) * 1e3); alphas.append(al)
        t_plan.append((t1 - t0) * 1e3)
        z0 = z1
    return dict(success=succ, steps=k + 1, plan_ms=t_plan, adapt_ms=t_adapt, alpha=alphas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["frozen", "adajepa", "optta", "optta30", "pess"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sev", type=float, default=0.85)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.rho is None:
        args.rho = -0.1 if args.method == "pess" else 1.0
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False  # FP32, TF32 off
    m = build_model()
    init_sd = {k: v.clone() for k, v in m.state_dict().items()}
    params = adapt_params(m)
    d = np.load(os.path.join(SCR, "pusht_head.npz")); a = d["action"]
    mean_a, std_a = a.mean(0), a.std(0)  # StandardScaler on the readable head of pusht_expert_train
    env = make_env()
    eps = episodes(args.seed, args.n)
    res = []
    T0 = time.time()
    for i, (s, g) in enumerate(eps):
        r = run_episode(m, env, s, g, args, i, init_sd, params, std_a, mean_a)
        res.append(r)
        print(f"{args.method} seed{args.seed} ep{i} succ={r['success']} steps={r['steps']} "
              f"plan_ms={np.median(r['plan_ms']):.0f} adapt_ms={np.median(r['adapt_ms']) if r['adapt_ms'] else 0:.0f} "
              f"wall={time.time()-T0:.0f}s", flush=True)
        json.dump(dict(args=vars(args), episodes=res, wall_s=time.time() - T0), open(args.out, "w"))


if __name__ == "__main__":
    main()
