"""WS-168: train and evaluate GAWM-S (released model code, ga-wm @ 662e6d5) on the SO-101 bowls data.

  python gawm_so101.py train --data DIR --model gawm|nowm --train-split train172|quarter_seed0 --seed 0 --steps 20000 --out RUN
  python gawm_so101.py eval  --data DIR --run RUN            (also runs at the end of `train`)
  python gawm_so101.py checks --data DIR                     (PROTOCOL section 6: shapes, masks, hand L1, param counts)

Deviation D1 (our own trainer) and D2 (design choices) are in journal/workstack/notes/WS-168.md. Nothing here sets a bar.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

GA = os.environ.get("GAWM_REPO", "/s/ga-wm")
sys.path.insert(0, GA)
SPLITS = Path(__file__).parent / "splits"
H = 32
FUT = (6, 12)  # frames at 30 fps = future_time_offsets_s 0.2, 0.4 of the released config
TAG, SPEC = "so101", "so101_joint_6"
ENC = os.environ.get("GAWM_ENC") or __import__("glob").glob("/hf/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/snapshots/*")[0]  # local snapshot dir (the repo's resolver would go to ModelScope)
REF_CFG = os.environ.get("GAWM_REF_CFG", "/s/ref_libero/config.yaml")


def split(name):
    return json.load(open(SPLITS / f"{name}.json"))["episodes"]


class Data:
    def __init__(self, d):
        self.d = d
        self.meta = json.load(open(f"{d}/episodes.json"))
        z = np.load(f"{d}/arrays.npz")
        self.action, self.state, self.task, self.ep_of = z["action"], z["state"], z["task_index"], z["ep_of_frame"]
        self.top = np.load(f"{d}/frames_top.npy", mmap_mode="r")
        self.wrist = np.load(f"{d}/frames_wrist.npy", mmap_mode="r")
        self.by_orig = {m["orig"]: i for i, m in enumerate(self.meta["episodes"])}
        self.tasks = {int(self.task[m["start"]]): m["task"] for m in self.meta["episodes"]}

    def frames_of(self, orig_eps):
        out = []
        for e in orig_eps:
            if e in self.by_orig:
                m = self.meta["episodes"][self.by_orig[e]]
                out.append(np.arange(m["start"], m["start"] + m["length"]))
        return np.concatenate(out) if out else np.zeros(0, int)


def norm_stats(data, train_eps):
    idx = data.frames_of(train_eps)
    a, s = data.action[idx], data.state[idx]
    return {"a_mean": a.mean(0), "a_std": np.maximum(a.std(0), 1e-6), "s_mean": s.mean(0), "s_std": np.maximum(s.std(0), 1e-6)}


class DS(torch.utils.data.Dataset):
    def __init__(self, data, idx, ns):
        self.d, self.idx, self.ns = data, idx, ns

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        d, ns = self.d, self.ns
        g = int(self.idx[i])
        m = d.meta["episodes"][int(d.ep_of[g])]
        s0, n = m["start"], m["length"]
        t = g - s0
        view = lambda j: [np.asarray(d.top[s0 + j]), np.asarray(d.wrist[s0 + j])]
        fj = [min(t + f, n - 1) for f in FUT]
        k = np.arange(t, t + H)
        valid = k < n
        a = (d.action[s0 + np.minimum(k, n - 1)] - ns["a_mean"]) / ns["a_std"]
        st = (d.state[g] - ns["s_mean"]) / ns["s_std"]
        return {
            "image": view(t), "future_images": [view(j) for j in fj],
            "action": a.astype(np.float32), "state": st.astype(np.float32), "lang": m["task"],
            "robot_tag": TAG, "action_spec_id": SPEC, "future_time_offsets_s": [0.0, 0.2, 0.4],
            "action_valid_mask": valid, "future_frame_valid_mask": np.array([True] + [t + f < n for f in FUT]),
            "view_valid_mask": [True, True], "_ep": int(m["orig"]), "_t": t,
        }


def build_model(kind, tasks):
    from omegaconf import OmegaConf
    from starVLA.model.framework.WM4A.GAWM import GAWM
    cfg = OmegaConf.load(REF_CFG)
    fw = cfg.framework
    fw.action_model.action_horizon = H
    fw.action_model.embodiment_heads = {TAG: {"action_dim": 6, "action_horizon": H, "action_spec_id": SPEC,
                                              "gripper_indices": [5], "state_dim": 6}}
    fw.lang_cond = {"type": "text", "embed_dim": 384, "text_hidden_dim": 256, "text_depth": 2, "text_heads": 4,
                    "text_ffn_dim": 512, "text_max_length": 128, "text_dropout": 0.1}
    fw.world_model.camera_names = ["top", "wrist"]
    fw.world_model.vision_encoder_path = ENC
    if kind == "nowm":
        for k in ("loss_latent_weight", "visual_token_diversity_weight", "visual_token_variance_weight"):
            fw.world_model[k] = 0.0
    cfg.datasets = {"vla_data": {}}
    model = GAWM(cfg)
    if kind == "nowm":
        model.world_model = NullWM(model.world_model)
    return model


class NullWM(torch.nn.Module):
    """Table II's 'w/o world model': the action head sees the current tokens only (repeated to keep the head's shape);
    no forecast, no latent / feature losses. The wrapped predictor is never called."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        for p in inner.parameters():
            p.requires_grad_(False)

    def _fut(self, latent):
        return latent[:, :1].expand(-1, 2, *latent.shape[2:])

    def forward(self, latent, **kw):
        z = latent.new_zeros(())
        return {"pred_future_latent": self._fut(latent), "latent_loss": z, "latent_cosine_loss": z}

    def regress_future(self, latent, **kw):
        return self._fut(latent)


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def count(model):
    tr = sum(p.numel() for p in trainable(model))
    fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": tr, "frozen": fr}


def collate(b):
    return b


@torch.no_grad()
def evaluate(model, data, ns, eps, bs=64, limit=None):
    model.eval()
    idx = data.frames_of(eps)
    if limit:
        idx = idx[:limit]
    ds = DS(data, idx, ns)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate)
    rows = []
    for ex in dl:
        for e in ex:
            e["image"] = e["image"]
        out = model.predict_action(ex)["normalized_actions"]
        for e, p in zip(ex, out):
            a0 = e["action"][0]
            rows.append((e["_ep"], e["_t"], np.abs(p[0] - a0).mean(), np.abs((e["state"] - a0)).mean()))
    model.train()
    return rows


def per_episode(rows):
    out = {}
    for ep, t, l1, nul in rows:
        out.setdefault(ep, []).append((l1, nul))
    return {int(e): {"l1": float(np.mean([x[0] for x in v])), "identity_null": float(np.mean([x[1] for x in v])), "frames": len(v)}
            for e, v in sorted(out.items())}


def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    data = Data(a.data)
    train_eps = split(a.train_split)
    ns = norm_stats(data, split("train172"))
    model = build_model(a.model, data.tasks).cuda().train()
    cnt = count(model)
    print("params", cnt, flush=True)
    opt = torch.optim.AdamW(trainable(model), lr=a.lr, weight_decay=a.wd)
    warm = a.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    idx = data.frames_of(train_eps)
    ds = DS(data, idx, ns)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=a.steps * a.bs, generator=torch.Generator().manual_seed(a.seed))
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, sampler=sampler, num_workers=a.workers, collate_fn=collate,
                                     drop_last=True, persistent_workers=True, prefetch_factor=4)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    for step, ex in enumerate(dl, 1):
        o = model(ex)
        loss = o["action_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
        opt.step(); sched.step()
        if step % a.log_every == 0 or step == 1:
            rec = {"step": step, "loss": float(loss), "l1": float(o["l1_action_loss"]), "latent": float(o["latent_loss"]),
                   "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith(("backbone.", "world_model.inner.")) or False},
               out / "trainable.pt")
    json.dump({"args": vars(a), "params": cnt, "norm": {k: v.tolist() for k, v in ns.items()}}, open(out / "run.json", "w"))
    rows = evaluate(model, data, ns, split("heldout10"), limit=a.eval_limit)
    json.dump(per_episode(rows), open(out / "heldout_l1.json", "w"), indent=1)
    print("heldout", per_episode(rows), flush=True)


def checks(a):
    data = Data(a.data)
    ns = norm_stats(data, split("train172"))
    eps = [e for e in data.by_orig]
    print("episodes", eps, "frames", [data.meta["episodes"][data.by_orig[e]]["length"] for e in eps])
    ds = DS(data, data.frames_of(eps), ns)
    n = len(ds)
    last = data.meta["episodes"][1]["start"] + data.meta["episodes"][1]["length"] - 1
    ex = ds[int(np.where(data.frames_of(eps) == last)[0][0])]
    print("action", ex["action"].shape, "valid mask last frame of ep", ex["action_valid_mask"].sum(), "future valid", ex["future_frame_valid_mask"])
    ex5 = [ds[i] for i in range(len(ds) - 384, len(ds) - 384 + 5)]  # first 5 frames of the last (held-out) episode
    for e in ex5:
        assert e["action"].shape == (H, 6) and e["state"].shape == (6,)
    model = build_model(a.model if hasattr(a, "model") else "gawm", data.tasks).cuda().eval()
    print("params gawm", count(model))
    p = model.predict_action(ex5)["normalized_actions"]
    hand = [float(np.mean(np.abs(p[i][0] - ex5[i]["action"][0]))) for i in range(5)]
    ev = evaluate(model, data, ns, [e for e in eps][-1:], limit=5)
    print("hand L1 on 5 frames", hand, "evaluate() same frames of ep", eps[-1], [round(float(r[2]), 4) for r in ev], "null", [round(float(r[3]), 4) for r in ev])
    del model; torch.cuda.empty_cache()
    m2 = build_model("nowm", data.tasks).cuda()
    print("params nowm", count(m2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "checks"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="gawm", choices=["gawm", "nowm"])
    ap.add_argument("--train-split", default="train172")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--out", default="run")
    a = ap.parse_args()
    {"train": train, "checks": checks}[a.cmd](a)
