"""W1-R (D_JEPA_PROTOCOL.md, amendment 2): D-JEPA's relational operator (arXiv 2609.24749 §3-4, App. A.2) trained on top
of the frozen W1, scored with G1b regret against the action-only baseline.

Candidates are G1b's (g1b_range.build_chunks): val episodes seed 0 for evaluation, train episodes seed 1 for the
operator (80 % train / 20 % calibration episodes, seed 0). Labels are proxy: the executed chunk is the only success.
Step 0 reproduces W1's and action-only's stored G1b regret before any W1-R number.

    WM_SPLIT=split.json python djepa_operator.py --w1 <dir with weights_final.pt> --data <so101-bowls-h264-224> \\
        --cache <latents.pt> --out results/2026-10-22_djepa_operator/w1r.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_diag as D  # noqa: E402
import g1b_range as G  # noqa: E402
import probe as P  # noqa: E402

DS, K, HMAX = G.DS, 8, G.HMAX
STORED = {"W1": {5: 0.2635, 10: 0.2767, 15: 0.2755, 20: 0.2577, 30: 0.2415},          # 2026-10-12_g1_diag_exploratory
          "action_only": {5: -0.0146, 10: -0.0079, 15: -0.0049, 20: -0.005, 30: -0.005}}
HP = dict(T=0.05, mu=0.02, w_local=0.25, w_trust=0.1, k_local=16, eps=0.2, rank=8, hidden=128, layers=2, heads=4,
          updates=1000, lr=3e-4, wd=1e-4, clip=1.0, batch=64)
TAUS = [round(0.05 * i, 2) for i in range(21)] + [None]                                # None = never replace


@torch.no_grad()
def end_features(cand, Z, anchors, chunks, batch=16):
    """Per anchor, candidate and d (H = d): imagined end latent minus the goal latent z_{t+d}. -> (N, K, len(DS), D)."""
    out = []
    for i in range(0, len(anchors), batch):
        an = anchors[i:i + batch]
        ctx = torch.stack([Z[ep][t - G.C + 1:t + 1] for ep, t in an]).float()
        a = torch.as_tensor(chunks[i:i + batch]).flatten(0, 1)
        rolled = cand.rollout(ctx.repeat_interleave(K, 0), a).view(len(an), K, HMAX, -1)
        fut = torch.stack([Z[ep][t + 1:t + 1 + HMAX] for ep, t in an]).float()
        out.append(torch.stack([rolled[:, :, d - 1] - fut[:, None, d - 1] for d in DS], 2).half())
    return torch.cat(out)


class Operator(nn.Module):
    """Shared token encoder + permutation-equivariant Transformer layers + bounded rank-8 correction (zero-init)."""

    def __init__(self, dim, hp):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.inp = nn.Sequential(nn.Linear(dim + 1, hp["hidden"]), nn.GELU(), nn.Linear(hp["hidden"], hp["hidden"]))
        layer = nn.TransformerEncoderLayer(hp["hidden"], hp["heads"], 2 * hp["hidden"], dropout=0.0, batch_first=True)
        self.set = nn.TransformerEncoder(layer, hp["layers"])                          # no positional input
        self.down, self.up = nn.Linear(hp["hidden"], hp["rank"]), nn.Linear(hp["rank"], 1)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        self.eps = hp["eps"]

    def forward(self, diff, r):
        h = self.set(self.inp(torch.cat([self.ln(diff), r[..., None]], -1)))
        delta = self.eps * torch.tanh(self.up(torch.tanh(self.down(h)))).squeeze(-1)
        return r + delta, delta


def ranks(cost):
    return cost.argsort(1).argsort(1).float() / (K - 1)


def loss_fn(s, delta, hp):
    """L_R with one success (index 0): success mass + local success/failure pairs + trust region."""
    y = torch.zeros_like(s, dtype=torch.bool); y[:, 0] = True
    l_succ = -torch.logsumexp(-s / hp["T"] + torch.where(y, 0.0, float("-inf")), 1) + torch.logsumexp(-s / hp["T"], 1)
    k = min(hp["k_local"], s.shape[1])
    low = torch.zeros_like(y).scatter(1, s.topk(k, 1, largest=False).indices, True)
    pair = (y & low)[:, :, None] & (~y & low)[:, None, :]                                # (B, i success, j failure)
    sp = nn.functional.softplus((hp["mu"] + s[:, :, None] - s[:, None, :]) / hp["T"])
    has = pair.flatten(1).any(1)
    full = y[:, :, None] & ~y[:, None, :]                                               # one class in the subset: all pairs
    pair = torch.where(has[:, None, None], pair, full)
    l_local = (sp * pair).flatten(1).sum(1) / pair.flatten(1).sum(1)
    return (l_succ + hp["w_local"] * l_local + hp["w_trust"] * delta.pow(2).mean(1)).mean()


def train_operator(diff, cost, seed, hp, dev):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    op = Operator(diff.shape[-1], hp).to(dev)
    opt = torch.optim.AdamW(op.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    r = ranks(cost)
    for _ in range(hp["updates"]):
        idx = torch.randint(len(diff), (hp["batch"],), generator=g)
        s, delta = op(diff[idx].float().to(dev), r[idx].to(dev))
        loss = loss_fn(s, delta, hp)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(op.parameters(), hp["clip"]); opt.step()
    return op.eval()


@torch.no_grad()
def scores(op, diff, cost, dev):
    s, _ = op(diff.float().to(dev), ranks(cost).to(dev))
    return s.cpu()


def compose(cost, s, tau):
    """Index chosen per anchor: the base (W1) winner unless b_base - s_operator > tau."""
    b = ranks(cost)
    ib, i_s = b.argmin(1), s.argmin(1)
    if tau is None:
        return ib, torch.zeros(len(ib), dtype=torch.bool)
    swap = (b[torch.arange(len(b)), ib] - s[torch.arange(len(s)), i_s] > tau) & (i_s != ib)
    return torch.where(swap, i_s, ib), swap


def onehot_score(sel):
    """A score whose argmin is sel (for g1_diag.regret)."""
    sc = torch.ones(len(sel), K); sc[torch.arange(len(sel)), sel] = 0.0
    return sc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w1", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="W1 latents of every episode (written if missing)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0)
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    tr_eps, va_eps = split["train_episodes"], split["val_episodes"]
    cand = G.W1(args.w1)
    tab = P.frame_table(args.data)
    a_mean, a_std = P.action_stats(args.data, tr_eps)
    if args.cache.exists():
        Z = torch.load(args.cache)
    else:
        Z = {ep: z.half() for ep, z in G.embed_episodes(cand, args.data, tab, tr_eps + va_eps, args.workers).items()}
        torch.save(Z, args.cache)
    norm = lambda x: ((x - a_mean) / a_std).astype(np.float32)  # noqa: E731
    sets = {}
    for name, eps, seed in (("val", va_eps, 0), ("train", tr_eps, 1)):
        an, ch = G.build_chunks({ep: norm(tab[ep][1]) for ep in eps}, eps, np.random.default_rng(seed))
        sets[name] = {"anchors": an, "chunks": ch, "diff": end_features(cand, Z, an, ch),
                      "cost": D.end_state_dist(ch, an, {ep: norm(tab[ep][2]) for ep in eps})}
        print(f"{name}: {len(an)} anchors, features {tuple(sets[name]['diff'].shape)} ({time.time() - t0:.0f} s)", flush=True)
    va = sets["val"]
    res = {"protocol": "wm_train/D_JEPA_PROTOCOL.md amendment 2", "revision": split["revision"], "hp": HP,
           "anchors": {"val": len(va["anchors"]), "train": len(sets["train"]["anchors"])}, "step0": {}, "per_d": {}}

    # step 0: reproduce W1 native and action-only regret (stored 2026-10-12)
    ok = len(va["anchors"]) == 703
    for di, d in enumerate(DS):
        w1 = D.regret(va["diff"][:, :, di].float().norm(dim=-1), va["cost"][d], va["anchors"])
        ao = D.regret(va["cost"][d], va["cost"][d], va["anchors"])
        res["step0"][f"d{d}"] = {"W1": w1, "action_only": ao}
        ok &= abs(w1["regret"] - STORED["W1"][d]) <= 0.01 and abs(ao["regret"] - STORED["action_only"][d]) <= 0.002
    res["step0"]["reproduced"] = bool(ok)
    print("step 0:", {k: (v["W1"]["regret"], v["action_only"]["regret"]) for k, v in res["step0"].items() if k != "reproduced"},
          "reproduced" if ok else "NOT REPRODUCED", flush=True)
    if not ok:
        args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(res, indent=1))
        raise SystemExit("step 0 failed: stop (protocol)")

    # operator-training / calibration episodes (seed 0), one operator per d and seed
    rng = np.random.default_rng(0)
    perm = list(rng.permutation(tr_eps)); cal_eps = set(perm[:round(0.2 * len(perm))])
    tr = sets["train"]
    is_cal = torch.tensor([ep in cal_eps for ep, _ in tr["anchors"]])
    dev = P.DEV
    for di, d in enumerate(DS):
        cost_tr = tr["diff"][:, :, di].float().norm(dim=-1)                                # W1's native cost
        cost_va = va["diff"][:, :, di].float().norm(dim=-1)
        row = {"seeds": []}
        for seed in range(args.seeds):
            op = train_operator(tr["diff"][~is_cal, :, di], cost_tr[~is_cal], seed, HP, dev)
            s_cal, s_va = scores(op, tr["diff"][is_cal, :, di], cost_tr[is_cal], dev), scores(op, va["diff"][:, :, di], cost_va, dev)
            top1 = {str(t): float((compose(cost_tr[is_cal], s_cal, t)[0] == 0).float().mean()) for t in TAUS}
            tau = max(TAUS, key=lambda t: (top1[str(t)], t is None, t or 0))                  # ties: prefer never / larger
            sel, swap = compose(cost_va, s_va, tau)
            reg = D.regret(onehot_score(sel), va["cost"][d], va["anchors"])
            row["seeds"].append({"seed": seed, "tau_R": tau, "calibration_top1": top1[str(tau)],
                                 "calibration_top1_never": top1["None"], "val_regret": reg,
                                 "val_top1": float((sel == 0).float().mean()), "val_swap_rate": float(swap.float().mean()),
                                 "val_top1_operator_ungated": float((s_va.argmin(1) == 0).float().mean())})
        rg = [x["val_regret"]["regret"] for x in row["seeds"]]
        row["regret_mean_over_seeds"] = float(np.mean(rg))
        # exploratory (labelled): the bound of a rank-local swap = take the rank-2 candidate exactly when it is the true chunk
        b = ranks(cost_va); i1, i2 = b.argmin(1), (b == 1 / (K - 1)).float().argmax(1)
        oracle = torch.where(i2 == 0, i2, i1)
        row["exploratory_rank2_swap_oracle_regret"] = D.regret(onehot_score(oracle), va["cost"][d], va["anchors"])
        row["W1_top1"] = float((cost_va.argmin(1) == 0).float().mean())
        res["per_d"][f"d{d}"] = row
        print(f"d={d}: W1-R regret {rg} (W1 {res['step0'][f'd{d}']['W1']['regret']}), taus "
              f"{[x['tau_R'] for x in row['seeds']]}, swap {[round(x['val_swap_rate'], 3) for x in row['seeds']]} "
              f"({time.time() - t0:.0f} s)", flush=True)

    rec = float(np.mean([res["per_d"][f"d{d}"]["regret_mean_over_seeds"] for d in DS]))
    ao_rec = float(np.mean([res["step0"][f"d{d}"]["action_only"]["regret"] for d in DS]))
    w1_rec = float(np.mean([res["step0"][f"d{d}"]["W1"]["regret"] for d in DS]))
    res["record"] = {"W1R_regret": rec, "W1_regret": w1_rec, "action_only_regret": ao_rec,
                     "bar_1_le_0.10": rec <= 0.10, "bar_2_le_action_only_plus_0.05": rec <= ao_rec + 0.05,
                     "verdict": "pass" if rec <= 0.10 and rec <= ao_rec + 0.05 else "fail"}
    res["seconds"] = round(time.time() - t0, 1)
    print("record:", res["record"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
