"""Offline action-sensitivity test for a trained multi-view LeWM (train_lewm_mv.py), on held-out episodes.

Autoregressive rollouts from the same C = 3 true context frames for K = 1..30, with the true actions and with corrupted
ones: cross-episode (actions of a window from another episode), zero (normalized 0 = the mean action), constant (the
first action held), time-shifted by +15 / -15 steps (same episode, wrong timing). Per K: error vs the true future
latents and the ratio to the true-action error; at K = 30: ||rollout(a) - rollout(a')|| / latent std (raw
sensitivity). Also: AdaLN-zero gate magnitudes per predictor block, action normalization statistics, the
action-frame pairing used by the adapter and the model.

    WM_DATA=<transcoded snapshot> WM_SPLIT=projects/vla-jepa-world-model/wm_train/split.json python projects/vla-jepa-world-model/wm_train/eval_action_sensitivity.py <run dir>/out/multiview \\
        [--windows 192] [--device mps|cpu|cuda]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_lewm_mv as T  # noqa: E402

SHIFT = 15


def load(run_dir: Path, device: str):
    import hydra
    from torch import nn

    ckpt = torch.load(run_dir / "weights_final.pt", map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = hydra.utils.instantiate(cfg.model)
    model.fuse = nn.Linear(cfg.embed_dim, cfg.embed_dim)
    if cfg.loss.idm.weight > 0:
        h = cfg.loss.idm.hidden_dim
        model.idm = nn.Sequential(nn.Linear(2 * cfg.embed_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                  nn.Linear(h, cfg.data.action_dim))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def rollout(model, z_ctx, actions, c, k):
    ctx, preds = z_ctx, []
    for i in range(k):
        nxt = model.predict(ctx[:, -c:], actions[:, i:i + c])[:, -1:]
        preds.append(nxt)
        ctx = torch.cat([ctx, nxt], 1)
    return torch.cat(preds, 1)


def gates(model):
    out = []
    for i, blk in enumerate(model.predictor.transformer.layers):
        lin = blk.adaLN_modulation[-1]
        w = lin.weight.detach().float().view(6, -1, lin.weight.shape[1])
        b = lin.bias.detach().float().view(6, -1)
        names = ["shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"]
        out.append({n: round(float(w[j].norm() + b[j].norm()), 4) for j, n in enumerate(names)} | {"block": i})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--windows", type=int, default=192)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(0)
    model, cfg = load(args.run_dir, args.device)
    c, k = cfg.wm.context, cfg.checks.horizon_k
    split = json.loads(Path(cfg.split_file if "WM_SPLIT" not in __import__("os").environ else __import__("os").environ["WM_SPLIT"]).read_text())
    import os
    cfg.dataset_root = os.environ["WM_DATA"]

    # Windows of SHIFT + C + K + SHIFT frames: the true actions are the middle slice, the time-shifted ones the outer.
    span = SHIFT + c + k + SHIFT
    train_set = T.make_dataset(cfg, split["train_episodes"], cfg.wm.context + cfg.wm.rollout)
    val = T.make_dataset(cfg, split["val_episodes"], span)
    tfs = [T.img_tf(v, cfg.img_size) for v in cfg.views]
    norm = {col: T.get_column_normalizer(train_set, col, col) for col in cfg.data.keys_to_load if col not in cfg.views}
    tf = T.spt.data.transforms.Compose(*tfs, *norm.values())
    val.transform = tf
    rng = np.random.default_rng(0)
    idx = rng.choice(len(val), size=min(args.windows, len(val)), replace=False)

    # action normalization + pairing facts
    raw_actions = np.asarray(train_set.get_col_data("action"))
    a_norm = norm["action"].transform.transform(torch.as_tensor(raw_actions, dtype=torch.float32)) \
        if hasattr(norm["action"], "transform") and hasattr(norm["action"].transform, "transform") else None
    facts = {"action_raw_mean": raw_actions.mean(0).round(2).tolist(), "action_raw_std": raw_actions.std(0).round(2).tolist()}
    if a_norm is not None:
        facts |= {"action_norm_mean": a_norm.mean(0).numpy().round(3).tolist(), "action_norm_std": a_norm.std(0).numpy().round(3).tolist()}
    facts["pairing"] = ("adapter: frame t and action t (the command recorded at frame t, which leads to frame t+1); model: "
                        "predict(z[j:j+C], a[j:j+C])[-1] -> z[j+C], so the last action used is a[j+C-1], the one leaving the "
                        "last context frame")

    zs, acts, eps = [], [], []
    for n, i in enumerate(idx):
        item = val[int(i)]
        batch = {kk: v.unsqueeze(0).to(args.device) for kk, v in item.items() if torch.is_tensor(v)}
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        mid = slice(SHIFT, SHIFT + c + k)
        sub = {v: batch[v][:, mid] for v in cfg.views} | {"action": batch["action"][:, mid]}
        with torch.no_grad():
            z, _ = T.encode(model, sub, cfg.views)
        zs.append(z.float().cpu())
        acts.append(batch["action"].float().cpu())
        eps.append(int(item["ep_idx"][0]) if "ep_idx" in item else -1)
        if (n + 1) % 32 == 0:
            print(f"encoded {n + 1}/{len(idx)} windows", flush=True)
    z = torch.cat(zs).to(args.device)                      # (N, C+K, D)
    a_all = torch.cat(acts).to(args.device)                # (N, span, A)
    eps = np.array(eps)
    true_a = a_all[:, SHIFT:SHIFT + c + k]
    # cross-episode: for each window, the actions of a window from another episode
    perm = np.arange(len(eps))
    for _ in range(100):
        rng.shuffle(perm)
        if (eps[perm] != eps).all():
            break
    corrupt = {
        "cross_episode": true_a[torch.as_tensor(perm, device=args.device)],
        "zero": torch.zeros_like(true_a),
        "constant_first": true_a[:, :1].expand_as(true_a).contiguous(),
        "shift_+15": a_all[:, 2 * SHIFT:2 * SHIFT + c + k],
        "shift_-15": a_all[:, 0:c + k],
    }
    with torch.no_grad():
        enc_a = {"true": model.action_encoder(true_a)} | {n: model.action_encoder(x) for n, x in corrupt.items()}
        roll = {n: rollout(model, z[:, :c], e, c, k).float() for n, e in enc_a.items()}
    target = z[:, c:c + k].float()
    lat_std = float(z.float().flatten(0, 1).std(0).mean())
    err = {n: (r - target).pow(2).mean(dim=(0, 2)).cpu().numpy() for n, r in roll.items()}
    copy = (z[:, c - 1:c].float() - target).pow(2).mean(dim=(0, 2)).cpu().numpy()
    res = {"windows": len(idx), "latent_std": round(lat_std, 4), "gates": gates(model), "facts": facts, "by_k": {}}
    for kk in (1, 5, 10, 15, 20, 30):
        res["by_k"][kk] = {"true_mse": round(float(err["true"][kk - 1]), 5), "copy_last_mse": round(float(copy[kk - 1]), 5),
                           **{f"{n}_ratio": round(float(err[n][kk - 1] / err["true"][kk - 1]), 3) for n in corrupt}}
    res["sensitivity_k30"] = {n: round(float((roll[n][:, -1] - roll["true"][:, -1]).norm(dim=-1).mean() / (lat_std * np.sqrt(z.shape[-1]))), 4)
                              for n in corrupt}
    best = min(res["by_k"][30][f"{n}_ratio"] for n in corrupt)
    res["verdict_k30_min_ratio"] = best
    res["decision"] = "action path alive (>= 1.3 at K=30 for every corruption)" if best >= 1.3 else \
        "below 1.3 at K=30 for at least one corruption"
    out = args.run_dir / "action_sensitivity.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
