"""Multi-view LeWM on the SO-101 bowls demos: the ContactGuard recipe (arXiv 2608.13438; spec: latent-stack architecture
v0.2 §4.2 B). One shared ViT-Tiny (patch 16) encodes every camera; per-view CLS -> BatchNorm projector -> mean over views
-> learned linear projection = the latent z_t (192). Predictor: 4 AdaLN-zero blocks, causal, context C = 3, trained
teacher-forced on L = 5 targets (each predicted from the C true frames before it) at 30 Hz, one joint action per step.
Loss: per-step MSE + SIGReg (0.09, 512 projections, 17 knots); optional inverse dynamics (off by default); optional
decision term (W1-D, D_JEPA_PROTOCOL.md: softmax over the true chunk and 7 G1b-type decoys, off by default).

Checks on held-out episodes (each epoch and at the end): latent std / effective rank (collapse), teacher-forced error
vs "copy the last latent", action shuffle (error with actions permuted across the batch / with the true actions),
open-loop autoregressive error for K = 1..30 with the true actions (should grow smoothly).

    WM_DATA=<snapshot> WM_SPLIT=projects/vla-jepa-world-model/wm_train/split.json WM_OUT=<out> python projects/vla-jepa-world-model/wm_train/train_lewm_mv.py [hydra overrides]
"""

import json
import math
import os
import time
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt
from stable_worldmodel.data import LeRobotAdapter
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from torch import nn

ALIASES = {"observation.images.wrist_camera": "wrist"}
PRIMARY = "observation.images.front_top_camera"  # -> "pixels"


def img_tf(key: str, size: int):
    return dt.transforms.Compose(
        dt.transforms.ToImage(**dt.dataset_stats.ImageNet, source=key, target=key),
        dt.transforms.Resize((size, size), source=key, target=key),
    )


def make_dataset(cfg, episodes, num_steps):
    # camera names default to the bowls data; the brick fine-tune overrides data.primary_camera_key / data.aliases
    return LeRobotAdapter(cfg.data.repo_id, root=cfg.dataset_root, episodes=episodes, frameskip=cfg.data.frameskip,
                          num_steps=num_steps, keys_to_load=list(cfg.data.keys_to_load),
                          keys_to_cache=list(cfg.data.keys_to_cache),
                          primary_camera_key=cfg.data.get("primary_camera_key", PRIMARY),
                          key_aliases=dict(cfg.data.get("aliases", ALIASES)))


class DecoyWindows(torch.utils.data.Dataset):
    """Train windows + what the decision term needs beyond them (D_JEPA_PROTOCOL.md): the episode index, and the same
    episode's actions shifted by -shift / +shift steps (G1b's time-shifted decoys), read from the cached action column
    and normalised by the same fitted scaler as `action`. A shift that leaves the episode is marked invalid."""

    def __init__(self, base, scaler, shift: int):
        self.base, self.scaler, self.shift = base, scaler, shift

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        ep, start = self.base.clip_indices[i]
        n, acts = self.base.span, self.base._cache["action"]
        shifted, valid = [], []
        for s in (-self.shift, self.shift):
            ok = 0 <= start + s and start + s + n <= self.base.lengths[ep]
            g = int(self.base.offsets[ep] + (start + s if ok else start))
            shifted.append(self.scaler(torch.as_tensor(np.asarray(acts[g:g + n]))).reshape(item["action"].shape))
            valid.append(ok)
        item["dec_ep"] = torch.tensor(ep)
        item["dec_shifted"] = torch.stack(shifted)
        item["dec_valid"] = torch.tensor(valid)
        return item


def decision_loss(model, z, batch, cfg):
    """W1-D term: roll out h steps from the C real context latents with 8 candidate chunks (true, 4 cross-episode from
    the batch, shifted -/+ shift, first action held, as g1b_range.build_chunks) and take the softmax cross-entropy of
    -score / T with the true chunk as the class. Latents are detached: the term trains the predictor and action encoder."""
    d, c = cfg.loss.decision, cfg.wm.context
    n = c + d.horizon - 1                                                   # actions a_0 .. a_{c+h-2}
    act, ep, b = batch["action"][:, :n], batch["dec_ep"], batch["action"].shape[0]
    cands, valid = [act], [torch.ones(b, dtype=torch.bool, device=act.device)]
    for k in range(1, d.n_cross + 1):
        j = torch.roll(torch.arange(b, device=act.device), -k)
        cands.append(act[j]); valid.append(ep[j] != ep)
    shifted = torch.nan_to_num(batch["dec_shifted"][:, :, :n], 0.0)
    cands += [shifted[:, 0], shifted[:, 1]]; valid += [batch["dec_valid"][:, 0], batch["dec_valid"][:, 1]]
    cands.append(act[:, :1].expand(-1, n, -1)); valid.append(valid[0])
    a = model.action_encoder(torch.stack(cands, 1).flatten(0, 1))           # (b K, n, D)
    kk = len(cands)
    ctx = z[:, :c].detach().repeat_interleave(kk, 0)
    for i in range(d.horizon):                                              # eval_action_sensitivity.rollout indexing
        ctx = torch.cat([ctx, model.predict(ctx[:, -c:], a[:, i:i + c])[:, -1:]], 1)
    end, tgt = ctx[:, -1].view(b, kk, -1), z[:, c + d.horizon - 1].detach()[:, None]
    if d.kind == "l2":
        score = (end - tgt).norm(dim=-1)
    elif d.kind == "cosine":
        score = 1 - nn.functional.cosine_similarity(end, tgt, dim=-1)
    else:
        raise ValueError(f"loss.decision.kind must be l2 or cosine, not {d.kind}")
    logits = (-score / d.temperature).masked_fill(~torch.stack(valid, 1), float("-inf"))
    return nn.functional.cross_entropy(logits.float(), torch.zeros(b, dtype=torch.long, device=act.device))


def decision_on(cfg) -> bool:
    return cfg.loss.get("decision") is not None and cfg.loss.decision.weight > 0


def encode(model, batch, views):
    """Shared encoder over views -> per-view projector -> mean -> linear fusion. Returns z (B, T, D), action emb.
    A spatial model (spatial_lewm.SpatialLeWM) encodes itself: z (B, T, tokens, D)."""
    if hasattr(model, "encode_views"):
        return model.encode_views(batch, views)
    x = torch.stack([batch[v] for v in views], dim=2)  # (B, T, V, 3, H, W)
    b, t, v = x.shape[:3]
    cls = model.encoder(x.flatten(0, 2).to(next(model.encoder.parameters()).dtype),
                        interpolate_pos_encoding=True).last_hidden_state[:, 0]
    z = model.fuse(model.projector(cls).view(b, t, v, -1).mean(dim=2))
    return z, model.action_encoder(batch["action"])


def teacher_forced(model, z, a, c: int, n: int):
    """Predict z[:, c + j] from z[:, j:j+c] and a[:, j:j+c] for j < n (one window per target). -> pred, target (B, n, D)."""
    win = torch.stack([z[:, j:j + c] for j in range(n)], dim=1).flatten(0, 1)
    act = torch.stack([a[:, j:j + c] for j in range(n)], dim=1).flatten(0, 1)
    pred = model.predict(win, act)[:, -1].view(z.shape[0], n, *z.shape[2:])
    return pred, z[:, c:c + n]


def forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    z, a = encode(self.model, batch, cfg.views)
    pred, tgt = teacher_forced(self.model, z, a, cfg.wm.context, cfg.wm.rollout)
    out = {"pred_loss": (pred - tgt).pow(2).mean(), "sigreg_loss": self.sigreg(z.transpose(0, 1) if z.dim() == 3 else z.permute(1, 0, 2, 3).flatten(1, 2))}
    out["loss"] = out["pred_loss"] + cfg.loss.sigreg.weight * out["sigreg_loss"]
    if cfg.loss.idm.weight > 0:
        idm = self.model.idm(torch.cat([z[:, :-1], z[:, 1:]], -1))
        out["idm_loss"] = (idm - batch["action"][:, :-1]).pow(2).mean()
        out["loss"] = out["loss"] + cfg.loss.idm.weight * out["idm_loss"]
    if decision_on(cfg) and stage == "fit":
        out["decision_loss"] = decision_loss(self.model, z, batch, cfg)
        out["loss"] = out["loss"] + cfg.loss.decision.weight * out["decision_loss"]
    self.log_dict({f"{stage}/{k}": v.detach() for k, v in out.items()}, on_step=True, sync_dist=True)
    return out | {"emb": z}


class Recorder(Callback):
    def __init__(self, cfg, out: Path, val_loader, horizon_loader):
        self.cfg, self.out, self.val, self.horizon = cfg, out, val_loader, horizon_loader
        self.t0, self.t_last, self.t_fetch, self.steps, self.waits = time.time(), None, None, [], []
        self.best, self.best_step, self.stop_reason = float('inf'), 0, None   # best val pred_over_copy (cfg.early_stop)

    def write(self, rec):
        with open(self.out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.time()
        if self.t_last is not None:
            self.waits.append(now - self.t_last)
        self.t_fetch = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        self.steps.append(now - self.t_fetch)
        self.t_last = now
        if trainer.global_step % self.cfg.trainer.log_every_n_steps == 0 and self.steps:
            s, w = sum(self.steps) / len(self.steps), sum(self.waits) / max(1, len(self.waits))
            m = {k: float(v) for k, v in trainer.callback_metrics.items() if "loss" in k}
            self.write({"kind": "train", "step": trainer.global_step, "epoch": trainer.current_epoch,
                        "elapsed_s": round(now - self.t0, 1), "gpu_s_per_step": round(s, 4),
                        "data_wait_s_per_step": round(w, 4),
                        "samples_per_s": round(self.cfg.loader.batch_size / max(s + w, 1e-9), 1),
                        "lr": trainer.optimizers[0].param_groups[0]["lr"], **m})
            self.steps, self.waits = [], []

    @torch.no_grad()
    def checks(self, pl_module, tag):
        cfg, model, dev = self.cfg, pl_module.model, pl_module.device
        was = model.training
        model.eval()
        ac = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
        zs, err, copy, shuf = [], [], [], []
        for i, batch in enumerate(self.val):
            if i >= cfg.checks.batches:
                break
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            with ac:
                z, a = encode(model, batch, cfg.views)
                p, t = teacher_forced(model, z, a, cfg.wm.context, cfg.wm.rollout)
                ps, _ = teacher_forced(model, z, a[torch.randperm(a.shape[0], device=dev)], cfg.wm.context, cfg.wm.rollout)
            z, p, t, ps = z.float(), p.float(), t.float(), ps.float()
            zs.append(z.flatten(0, -2))      # (samples, D): frames, or frames x tokens for a spatial model
            err.append((p - t).pow(2).mean().item())
            shuf.append((ps - t).pow(2).mean().item())
            prev = z[:, cfg.wm.context - 1:cfg.wm.context - 1 + cfg.wm.rollout]
            copy.append((prev - t).pow(2).mean().item())
        zz = torch.cat(zs)
        std = zz.std(0)
        eig = torch.linalg.eigvalsh(torch.cov(zz.T)).clamp(min=0)
        pr = eig / eig.sum()
        rank = float(torch.exp(-(pr[pr > 0] * pr[pr > 0].log()).sum()))
        # open-loop: autoregressive rollout with the true actions for K steps
        k_err, k_copy, c = None, None, cfg.wm.context
        for i, batch in enumerate(self.horizon):
            if i >= cfg.checks.horizon_batches:
                break
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            with ac:
                z, a = encode(model, batch, cfg.views)
                ctx = z[:, :c]
                preds = []
                for k in range(cfg.checks.horizon_k):
                    nxt = model.predict(ctx[:, -c:], a[:, k:k + c])[:, -1:]
                    preds.append(nxt)
                    ctx = torch.cat([ctx, nxt], 1)
            pk = torch.cat(preds, 1).float()
            tk = z[:, c:c + cfg.checks.horizon_k].float()
            e = (pk - tk).pow(2).flatten(2).mean(dim=(0, 2))
            cp = (z[:, c - 1:c].float() - tk).pow(2).flatten(2).mean(dim=(0, 2))
            k_err = e if k_err is None else k_err + e
            k_copy = cp if k_copy is None else k_copy + cp
        n = max(1, min(cfg.checks.horizon_batches, len(self.horizon)))
        k_err, k_copy = (k_err / n).tolist(), (k_copy / n).tolist()
        mono = sum(k_err[i + 1] >= k_err[i] * 0.98 for i in range(len(k_err) - 1)) / max(1, len(k_err) - 1)
        res = {"emb_std_mean": float(std.mean()), "eff_rank": rank, "eff_rank_frac": rank / zz.shape[1],
               "tf_mse": sum(err) / len(err), "copy_last_mse": sum(copy) / len(copy),
               "shuffled_action_mse": sum(shuf) / len(shuf), "openloop_mse_by_k": [round(x, 5) for x in k_err],
               "openloop_copy_by_k": [round(x, 5) for x in k_copy], "openloop_monotone_frac": round(mono, 3)}
        res["pred_over_copy"] = res["tf_mse"] / max(res["copy_last_mse"], 1e-9)
        res["shuffle_ratio"] = res["shuffled_action_mse"] / max(res["tf_mse"], 1e-9)
        ck = cfg.checks
        res["pass"] = {"no_collapse_std": res["emb_std_mean"] >= ck.min_emb_std,
                       "no_collapse_rank": res["eff_rank_frac"] >= ck.min_eff_rank_frac,
                       "beats_copy_last": res["pred_over_copy"] <= ck.max_pred_over_copy,
                       "action_shuffle_hurts": res["shuffle_ratio"] >= ck.min_shuffle_ratio,
                       "openloop_grows_smoothly": res["openloop_monotone_frac"] >= 0.8,
                       "finite": all(math.isfinite(x) for x in [res["tf_mse"], res["emb_std_mean"]])}
        res["all_pass"] = all(res["pass"].values())
        model.train(was)
        if self.cfg.get("early_stop") and self.update_best(pl_module.global_step, res, tag):
            self.save(pl_module, "weights_best.pt")
            res["is_best"] = True
        if self.cfg.get("early_stop"):
            res["best_pred_over_copy"], res["best_step"] = self.best, self.best_step
        rec = {"kind": "checks", "tag": tag, "step": pl_module.global_step, **res}
        self.write(rec)
        return rec

    def update_best(self, step: int, res: dict, tag: str) -> bool:
        """Best-val bookkeeping (cfg.early_stop): True when this check is the best so far by val pred_over_copy; sets the stop request
        when the best is patience_steps old (never on the final check)."""
        improved = res["pred_over_copy"] < self.best
        if improved:
            self.best, self.best_step = res["pred_over_copy"], step
        if tag != "final" and step - self.best_step >= self.cfg.early_stop.patience_steps:
            self.stop_reason = (f"no improvement of pred_over_copy for {self.cfg.early_stop.patience_steps} steps "
                                f"(best {self.best:.4f} at step {self.best_step})")
            self.trainer_ref.should_stop = True
        return improved

    def save(self, pl_module, name):
        torch.save({"state_dict": pl_module.model.state_dict(), "config": OmegaConf.to_container(self.cfg, resolve=True)},
                   self.out / name)

    def on_train_epoch_end(self, trainer, pl_module):
        self.trainer_ref = trainer
        epoch = trainer.current_epoch + 1
        if epoch % self.cfg.checks.every_epochs == 0 or epoch == 1:  # ~2 min each on a 3090 Ti
            self.checks(pl_module, f"epoch_{epoch}")
        self.save(pl_module, "weights_last.pt")
        if epoch % self.cfg.save_every_epochs == 0:  # kept (and pushed to the Hub with the rest of out/)
            self.save(pl_module, f"weights_epoch_{epoch:03d}.pt")

    def on_train_end(self, trainer, pl_module):
        rec = self.checks(pl_module, "final")
        (self.out / "collapse_checks.json").write_text(json.dumps(rec, indent=1))
        self.save(pl_module, "weights_final.pt")
        self.write({"kind": "end", "step": trainer.global_step, "elapsed_s": round(time.time() - self.t0, 1),
                    "stop_reason": self.stop_reason, "best_step": self.best_step if self.cfg.get("early_stop") else None})


@hydra.main(version_base=None, config_path="config", config_name="lewm_mv_so101")
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(cfg.split_file).read_text())
    steps = cfg.wm.context + cfg.wm.rollout
    train_set = make_dataset(cfg, split["train_episodes"], steps)
    val_set = make_dataset(cfg, split["val_episodes"], steps)
    horizon_set = make_dataset(cfg, split["val_episodes"], cfg.wm.context + cfg.checks.horizon_k)
    tfs, norms = [img_tf(v, cfg.img_size) for v in cfg.views], {}
    for col in cfg.data.keys_to_load:
        if col not in cfg.views:
            norms[col] = get_column_normalizer(train_set, col, col)  # stats from the train episodes only
            tfs.append(norms[col])
    tf = spt.data.transforms.Compose(*tfs)
    for d in (train_set, val_set, horizon_set):
        d.transform = tf
    if decision_on(cfg):  # W1-D: the decision term needs decoy windows (train only; checks stay as G0)
        assert cfg.data.frameskip == 1 and cfg.loss.decision.horizon <= cfg.wm.rollout
        train_set = DecoyWindows(train_set, norms["action"].lambd, cfg.loss.decision.shift)
    print(f"multi-view {list(cfg.views)}: {len(train_set)} train / {len(val_set)} val windows of {steps} frames at "
          f"{30 // cfg.data.frameskip} Hz, revision {split['revision'][:12]}", flush=True)

    gen = torch.Generator().manual_seed(cfg.seed)
    lc = OmegaConf.to_container(cfg.loader, resolve=True)
    if lc["num_workers"] == 0:
        lc.pop("persistent_workers"); lc.pop("prefetch_factor")
    train = torch.utils.data.DataLoader(train_set, **lc, generator=gen)
    # The check loaders only run at epoch ends: 2 workers, not persistent, no extra prefetch. With 8 persistent
    # workers each, the 33-frame horizon windows alone buffered ~10 GB and the host OOM-killed the run (2026-10-10).
    vc = {**lc, "shuffle": False, "drop_last": False, "num_workers": min(2, lc["num_workers"])}
    vc.pop("persistent_workers", None); vc.pop("prefetch_factor", None)
    if vc["num_workers"] > 0:
        vc["prefetch_factor"] = 1
    val = torch.utils.data.DataLoader(val_set, **vc)
    hz = torch.utils.data.DataLoader(horizon_set, **{**vc, "batch_size": 16, "shuffle": True},
                                     generator=torch.Generator().manual_seed(cfg.seed))

    if cfg.get("spatial"):  # W3: DINOv2-initialized encoder trained end-to-end, token latents (spatial_lewm.py)
        from spatial_lewm import SpatialLeWM

        assert cfg.loss.idm.weight == 0, "the inverse-dynamics head is not implemented for token latents"
        model = SpatialLeWM(views=len(cfg.views), **cfg.spatial)
    else:
        model = hydra.utils.instantiate(cfg.model)
        model.fuse = nn.Linear(cfg.embed_dim, cfg.embed_dim)  # mean over views -> learned linear projection
    if cfg.loss.idm.weight > 0:
        h = cfg.loss.idm.hidden_dim
        model.idm = nn.Sequential(nn.Linear(2 * cfg.embed_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                  nn.Linear(h, cfg.data.action_dim))
    if cfg.get("init_from"):  # fine-tune: start from a trained multi-view checkpoint (weights_final.pt of an earlier run)
        ckpt = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"initialized from {cfg.init_from}", flush=True)
    total = cfg.trainer.max_epochs * len(train)
    if cfg.trainer.get("max_steps"):  # the cosine schedule ends at the step cap
        total = min(total, cfg.trainer.max_steps)
    optim = {"model_opt": {"modules": "model", "optimizer": dict(cfg.optimizer),
                           "scheduler": {"type": "LinearWarmupCosineAnnealingLR",
                                         "warmup_steps": max(1, int(0.01 * total)), "max_steps": total},
                           # spt runs manual optimization and steps the scheduler after every optimizer step
                           # (module.py), whatever "interval" says; "step" matches what actually happens.
                           "interval": "step"}}
    module = spt.Module(model=model, sigreg=SIGReg(**cfg.loss.sigreg.kwargs), forward=partial(forward, cfg=cfg),
                        optim=optim)
    OmegaConf.save(cfg, out / "config.yaml")
    (out / "split.json").write_text(json.dumps(split, indent=1))
    tcfg = {k: v for k, v in OmegaConf.to_container(cfg.trainer, resolve=True).items() if v is not None}
    trainer = pl.Trainer(**tcfg, callbacks=[Recorder(cfg, out, val, hz)], logger=False, enable_checkpointing=False,
                         default_root_dir=str(out))
    spt.Manager(trainer=trainer, module=module, data=spt.data.DataModule(train=train, val=val))()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run()
