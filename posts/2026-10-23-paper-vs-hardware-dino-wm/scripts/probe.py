"""C1 probe pipeline for the multi-view LeWM: PROBE_PROTOCOL.md with its 2026-10-11 addendum.

    selfcheck  this script's frame / action path against the training data loader (same z, same normalized actions)
    extract    latents of the training data (train + val episodes, an anchor every 3 frames) and of the gate rollouts
               (one anchor per labelled trial, 15 frames before the first grasp close) -> probe_features_*.npz
    analyze    A rank diagnostics, B what the latent encodes, C full latent vs top PCs, failure probe -> probe_results.json

Per anchor t: the encoder stages at t (CLS per view, projector per view, mean over views, fused z_t), the K = 30
rollout from the real frames t-2..t with the executed actions (imagined z^_{t+30}), and the real z_{t+30}.

    WM_SPLIT=projects/vla-jepa-world-model/wm_train/split.json python projects/vla-jepa-world-model/wm_train/probe.py \\
        extract <weights dir> --data <transcoded training snapshot> --rollouts <dir of transcoded rollout datasets>
    python projects/vla-jepa-world-model/wm_train/probe.py analyze <weights dir>
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_action_sensitivity as E  # noqa: E402  load(), rollout()

REPO = Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) > 3 else Path(__file__).resolve().parent  # flat copy on vast
CATALOG = REPO / "shared" / "gate" / "data" / "rollout_catalog.csv"
FPS, C, K, ONSET, THR, STRIDE, NPC = 30, 3, 30, 15, 5.0, 3, 14
CAMS = {"pixels": "observation.images.front_top_camera", "wrist": "observation.images.wrist_camera"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet, as the training transform
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- data

def open_ds(root: Path):
    """Each item: frames t-2, t-1, t, t+30 per camera; actions a_{t-2}..a_{t+30} (the C + K window of the rollout);
    state at t and t+30. Index = global frame index of t."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dt = {cam: [d / FPS for d in (-2, -1, 0, K)] for cam in CAMS.values()}
    dt["action"] = [d / FPS for d in range(-(C - 1), K + 1)]
    dt["observation.state"] = [0.0, K / FPS]
    return LeRobotDataset(f"local/{root.name}", root=root, delta_timestamps=dt)


def frame_table(root: Path):
    """episode -> (global index of its first frame, raw actions (T, 6), raw states (T, 6))."""
    import pandas as pd

    df = pd.concat([pd.read_parquet(p, columns=["index", "episode_index", "frame_index", "action", "observation.state"])
                    for p in sorted(root.glob("data/*/*.parquet"))]).sort_values("index")
    out = {}
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        assert (np.diff(g["index"].to_numpy()) == 1).all() and g["frame_index"].iloc[0] == 0
        out[int(ep)] = (int(g["index"].iloc[0]), np.stack(g["action"].to_numpy()), np.stack(g["observation.state"].to_numpy()))
    return out


def phases(gripper_action: np.ndarray):
    """(c1, o2) of the open / close / open pattern (prechecks/common.phase_windows rule), or (-1, -1)."""
    is_open = gripper_action > THR
    opens = np.flatnonzero(is_open[1:] & ~is_open[:-1]) + 1
    closes = np.flatnonzero(~is_open[1:] & is_open[:-1]) + 1
    if is_open[0] or not len(opens):
        return -1, -1
    c1 = closes[closes > opens[0]]
    if not len(c1):
        return -1, -1
    o2 = opens[opens > c1[0]]
    return (int(c1[0]), int(o2[0])) if len(o2) else (-1, -1)


def action_stats(train_root: Path, train_eps):
    tab = frame_table(train_root)
    a = np.concatenate([tab[e][1] for e in train_eps])
    return a.mean(0).astype(np.float32), a.std(0).astype(np.float32)


class Items(torch.utils.data.Dataset):
    def __init__(self, ds, idx):
        self.ds, self.idx = ds, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        it = self.ds[int(self.idx[i])]
        return {v: (it[cam] - MEAN) / STD for v, cam in CAMS.items()} | {"action": it["action"], "state": it["observation.state"]}


def batches(ds, idx, bs=64, workers=4):
    return torch.utils.data.DataLoader(Items(ds, idx), batch_size=bs, num_workers=workers, shuffle=False)


# ---------------------------------------------------------------- model

@torch.no_grad()
def encode_stages(model, frames):
    """frames: view -> (N, 3, H, W). Same order of operations as train_lewm_mv.encode."""
    x = torch.stack([frames[v] for v in CAMS], 1)
    n, nv = x.shape[:2]
    cls = model.encoder(x.flatten(0, 1).to(DEV), interpolate_pos_encoding=True).last_hidden_state[:, 0]
    proj = model.projector(cls)
    mean = proj.view(n, nv, -1).mean(1)
    return cls.view(n, nv, -1), proj.view(n, nv, -1), mean, model.fuse(mean)


@torch.no_grad()
def featurize(model, b, a_mean, a_std):
    """One batch -> stages at t, z_ctx (t-2..t), z_{t+30}, imagined z^_{t+30} with the executed actions."""
    n = b["action"].shape[0]
    frames = {v: b[v].flatten(0, 1) for v in CAMS}  # (N*4, 3, H, W)
    cls, proj, mean, z = encode_stages(model, frames)
    cls, proj, mean, z = (x.view(n, 4, *x.shape[1:]) for x in (cls, proj, mean, z))
    a = ((b["action"] - a_mean) / a_std).to(DEV)
    zhat = E.rollout(model, z[:, :C], model.action_encoder(a), C, K)[:, -1]
    return {"cls": cls[:, 2], "proj": proj[:, 2], "mean": mean[:, 2], "z_t": z[:, 2], "z_ctx": z[:, :C],
            "z30": z[:, 3], "zhat30": zhat, "a_norm": a}


def to_np(d):
    return {k: v.float().cpu().numpy() for k, v in d.items()}


# ---------------------------------------------------------------- commands

def selfcheck(args):
    """z and normalized actions of val episode 0, window 5, through the training loader vs through this script."""
    import train_lewm_mv as T

    model, cfg = E.load(args.weights, DEV)
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    cfg.dataset_root = str(args.data)
    train_set = T.make_dataset(cfg, split["train_episodes"], C + K)
    ref = T.make_dataset(cfg, split["val_episodes"][:1], C + K)
    norm = [T.get_column_normalizer(train_set, col, col) for col in cfg.data.keys_to_load if col not in cfg.views]
    ref.transform = T.spt.data.transforms.Compose(*[T.img_tf(v, cfg.img_size) for v in cfg.views], *norm)
    it = ref[5]  # frames 5..37 of the episode (stride-1 windows)
    with torch.no_grad():
        z_ref, _ = T.encode(model, {v: it[v][:C].unsqueeze(0).to(DEV) for v in cfg.views} |
                            {"action": torch.nan_to_num(it["action"][:C]).unsqueeze(0).to(DEV)}, cfg.views)
    a_mean, a_std = action_stats(args.data, split["train_episodes"])
    ds = open_ds(args.data)
    start = frame_table(args.data)[split["val_episodes"][0]][0]
    b = next(iter(batches(ds, [start + 5 + 2], bs=1, workers=0)))  # anchor t = frame 7: context 5, 6, 7
    f = featurize(model, b, torch.as_tensor(a_mean), torch.as_tensor(a_std))
    dz = float((f["z_ctx"][0] - z_ref[0]).abs().max())
    da = float((f["a_norm"][0, :C].cpu() - it["action"][:C]).abs().max())
    print(f"max |z diff| {dz:.2e} (z scale {float(z_ref.abs().mean()):.3f}); max |normalized action diff| {da:.2e}")
    assert dz < 1e-2 and da < 1e-3, "this script's input path differs from the training loader"
    print("selfcheck ok")


def extract(args):
    model, cfg = E.load(args.weights, DEV)
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    a_mean, a_std = (torch.as_tensor(x) for x in action_stats(args.data, split["train_episodes"]))

    # training data: every STRIDE-th frame with 2 frames of context before and 30 after
    tab = frame_table(args.data)
    rows = []
    for name, eps in (("train", split["train_episodes"]), ("val", split["val_episodes"])):
        for ep in eps:
            start, act, _ = tab[ep]
            c1, o2 = phases(act[:, 5])
            for t in range(C - 1, len(act) - K, STRIDE):
                rows.append((start + t, ep, t, int(name == "val"), c1, o2))
    rows = np.array(rows)
    out = {k: [] for k in ("cls", "proj", "mean", "z_t", "z30", "zhat30", "state", "action")}
    ds = open_ds(args.data)
    for i, b in enumerate(batches(ds, rows[:, 0], workers=args.workers)):
        f = to_np(featurize(model, b, a_mean, a_std))
        for k in ("cls", "proj", "mean", "z_t", "z30", "zhat30"):
            out[k].append(f[k])
        out["state"].append(b["state"].numpy()); out["action"].append(b["action"].numpy())
        if i % 20 == 0:
            print(f"train data: {min((i + 1) * 64, len(rows))}/{len(rows)}", flush=True)
    np.savez_compressed(args.weights / "probe_features_train.npz", ep=rows[:, 1], frame=rows[:, 2], is_val=rows[:, 3],
                        c1=rows[:, 4], o2=rows[:, 5], **{k: np.concatenate(v) for k, v in out.items()})

    # gate rollouts: one anchor per labelled trial
    trials = [r for r in csv.DictReader(open(CATALOG)) if r["labeled"] == "True"]
    recs, feats = [], []
    for name in sorted({r["dataset"] for r in trials}):
        root = args.rollouts / name
        tab, ds = frame_table(root), open_ds(root)
        sel = [r for r in trials if r["dataset"] == name]
        idx = []
        for r in sel:
            start, act, _ = tab[int(r["episode"])]
            g = act[:, 5]
            is_open = g > THR
            closes = np.flatnonzero(~is_open[1:] & is_open[:-1]) + 1
            no_close = not len(closes)
            t = int(np.argmax(g[:-1])) if no_close else int(closes[0]) - ONSET
            t = int(np.clip(t, C - 1, len(g) - K - 1))
            idx.append(start + t)
            recs.append({"trial_id": r["trial_id"], "session": name, "episode": int(r["episode"]), "anchor": t,
                         "no_close": no_close, "success": r["success"] == "True",
                         "missed_grasp": r["failure_mode"].startswith("missed_grasp")})
        for b in batches(ds, idx, workers=0):
            feats.append(to_np(featurize(model, b, a_mean, a_std)))
        print(f"rollouts: {name}: {len(sel)} trials", flush=True)
    f = {k: np.concatenate([x[k] for x in feats]) for k in ("z_t", "zhat30", "z_ctx", "a_norm")}
    # control: the same context with the executed actions of a different trial (fixed seeded derangement)
    rng = np.random.default_rng(0)
    perm = np.arange(len(recs))
    while (perm == np.arange(len(recs))).any():
        perm = rng.permutation(len(recs))
    with torch.no_grad():
        a = torch.as_tensor(f["a_norm"][perm], device=DEV)
        zc = E.rollout(model, torch.as_tensor(f["z_ctx"], device=DEV), model.action_encoder(a), C, K)[:, -1]
    np.savez_compressed(args.weights / "probe_features_rollouts.npz", z_t=f["z_t"], zhat30=f["zhat30"],
                        zhat30_other=zc.float().cpu().numpy(), chunk=f["a_norm"][:, C - 1:C - 1 + K].reshape(len(recs), -1),
                        perm=perm, meta=np.array(json.dumps(recs)))
    print(f"wrote probe_features_train.npz ({len(rows)} anchors), probe_features_rollouts.npz ({len(recs)} trials)")


# ---------------------------------------------------------------- analysis

def spectrum(x):
    x = x - x.mean(0)
    ev = np.clip(np.linalg.eigvalsh(np.cov(x.T))[::-1], 0, None)
    p = ev / ev.sum()
    cum = np.cumsum(p)
    return {"eff_rank": round(float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())), 2),
            "participation_ratio": round(float(ev.sum() ** 2 / (ev ** 2).sum()), 2),
            **{f"pcs_{int(q * 100)}": int(np.searchsorted(cum, q) + 1) for q in (0.9, 0.95, 0.99)},
            "dim": x.shape[1]}


def auroc(y, s):
    from scipy.stats import rankdata

    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_eval(xtr, ytr, xte, yte, kind):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(xtr)
    xtr, xte = sc.transform(xtr), sc.transform(xte)
    if kind == "r2":
        m = Ridge(alpha=1.0).fit(xtr, ytr)
        return round(float(r2_score(yte, m.predict(xte))), 4)
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(xtr, ytr)
    return round(auroc(yte, m.decision_function(xte)), 4)


def targets(st, act, frame, c1, o2):
    """Per anchor: (target, kind, mask) for the B probes, at the frame the state / action belong to."""
    ok = c1 >= 0
    closed = act[:, 5] <= THR
    return {
        "arm_pose_r2": (st[:, :5], "r2", np.ones(len(st), bool)),
        "gripper_pos_r2": (st[:, 5], "r2", np.ones(len(st), bool)),
        "gripper_open_auroc": (st[:, 5] > THR, "auc", np.ones(len(st), bool)),
        "grasp_onset_auroc": ((frame >= c1 - ONSET) & (frame < c1), "auc", ok),
        "contact_auroc": ((frame >= c1) & (frame < o2), "auc", ok & closed),
    }


def analyze(args):
    d = dict(np.load(args.weights / "probe_features_train.npz"))
    tr, va = d["is_val"] == 0, d["is_val"] == 1
    res = {"anchors": {"train": int(tr.sum()), "val": int(va.sum())},
           "episodes_without_phase_pattern": int(len(np.unique(d["ep"][d["c1"] < 0])))}

    # A. rank diagnostics per stage, train vs val
    stages = {"encoder_cls_front": d["cls"][:, 0], "encoder_cls_wrist": d["cls"][:, 1],
              "projector_front": d["proj"][:, 0], "projector_wrist": d["proj"][:, 1],
              "mean_over_views": d["mean"], "fused_z": d["z_t"], "imagined_zhat30": d["zhat30"], "real_z30": d["z30"]}
    res["A_rank"] = {s: {"train": spectrum(x[tr]), "val": spectrum(x[va])} for s, x in stages.items()}

    # PCA basis: real z_t of the train-episode frames, no labels
    mu = d["z_t"][tr].mean(0)
    _, sv, vt = np.linalg.svd(d["z_t"][tr] - mu, full_matrices=False)
    k95 = res["A_rank"]["fused_z"]["train"]["pcs_95"]
    bases = {"full": None, f"top{NPC}": vt[:NPC]} | ({f"top{k95}_95pct": vt[:k95]} if k95 != NPC else {})
    proj = lambda x, w: x if w is None else (x - mu) @ w.T  # noqa: E731

    # B + C. what the latent encodes: real z_t (targets at t), real z_{t+30} vs imagined z^_{t+30} (targets at t+30)
    st, act = d["state"], d["action"]
    tgt_t = targets(st[:, 0], act[:, C - 1], d["frame"], d["c1"], d["o2"])
    tgt_30 = targets(st[:, 1], act[:, C - 1 + K], d["frame"] + K, d["c1"], d["o2"])
    res["B_encodes"] = {}
    for feat, x, tg in (("real_z_t", d["z_t"], tgt_t), ("real_z30", d["z30"], tgt_30), ("imagined_zhat30", d["zhat30"], tgt_30)):
        for bname, w in bases.items():
            xf = proj(x, w)
            row = {}
            for tname, (y, kind, m) in tg.items():
                a, b = tr & m, va & m
                row[tname] = fit_eval(xf[a], y[a], xf[b], y[b], kind)
            row["val_positives"] = {t: int(y[va & m].sum()) for t, (y, k, m) in tg.items() if k == "auc"}
            res["B_encodes"][f"{feat}/{bname}"] = row
            print(feat, bname, row, flush=True)
    # leakage baselines for the t+30 targets: the executed chunk a_t..a_{t+29} is what the rollout sees besides the
    # context; if it decodes a target as well as z^_{t+30} does, that row says nothing about the world model
    chunk = act[:, C - 1:C - 1 + K].reshape(len(act), -1)
    for feat, x in (("actions_only_chunk", chunk), ("real_z_t_plus_chunk", np.concatenate([d["z_t"], chunk], 1))):
        row = {}
        for tname, (y, kind, m) in tgt_30.items():
            a, b = tr & m, va & m
            row[tname] = fit_eval(x[a], y[a], x[b], y[b], kind)
        res["B_encodes"][f"{feat}/full"] = row
        print(feat, row, flush=True)

    # failure probe (protocol): leave-one-session-out, pooled AUROC, bootstrap CIs
    r = dict(np.load(args.weights / "probe_features_rollouts.npz"))
    meta = json.loads(str(r["meta"]))
    sess = np.array([m["session"] for m in meta])
    succ = np.array([m["success"] for m in meta])
    missed = np.array([m["missed_grasp"] for m in meta])
    sets = {}
    for name, key in (("1_current_z_t", "z_t"), ("3_imagined_true_actions", "zhat30"), ("4_imagined_other_actions", "zhat30_other")):
        for bname, w in bases.items():
            sets[f"{name}/{bname}"] = proj(r[key], w)
    sets["2_raw_chunk/full"] = r["chunk"]
    res["failure_probe"] = {"trials": len(meta), "failures": int((~succ).sum()), "missed_grasps": int(missed.sum()),
                            "no_close_anchors": int(sum(m["no_close"] for m in meta)), "sessions": len(set(sess))}
    rng = np.random.default_rng(0)
    for lname, keep, y in (("all_failures", np.ones(len(meta), bool), ~succ), ("missed_grasp", succ | missed, ~succ)):
        yk, sk = y[keep], sess[keep]
        scores, skipped = {}, {}
        for fname, x in sets.items():
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            xk, s = x[keep], np.full(keep.sum(), np.nan)
            skipped[fname] = []
            for g in np.unique(sk):
                te, trn = sk == g, sk != g
                if len(np.unique(yk[trn])) < 2:
                    skipped[fname].append(str(g)); continue
                sc = StandardScaler().fit(xk[trn])
                m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(xk[trn]), yk[trn])
                s[te] = m.decision_function(sc.transform(xk[te]))
            scores[fname] = s
        ok = ~np.isnan(np.stack(list(scores.values()))).any(0)
        boot = rng.integers(0, ok.sum(), size=(10000, ok.sum()))
        yb = yk[ok]
        def ci(f):  # noqa: E306
            v = np.array([auroc(yb[i], f[i]) for i in boot])
            v = v[~np.isnan(v)]
            return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
        out = {"n": int(ok.sum()), "positives": int(yb.sum()), "skipped_folds": {k: v for k, v in skipped.items() if v}}
        for fname, s in scores.items():
            out[fname] = {"auroc": round(auroc(yb, s[ok]), 3), "ci95": ci(s[ok])}
        pairs = [("3_imagined_true_actions/full", "1_current_z_t/full"),
                 ("3_imagined_true_actions/full", "4_imagined_other_actions/full")]
        pairs += [(f"{n}/full", f"{n}/top{NPC}") for n in ("1_current_z_t", "3_imagined_true_actions", "4_imagined_other_actions")]
        out["paired_differences"] = {}
        for a, b in pairs:
            sa, sb = scores[a][ok], scores[b][ok]
            v = np.array([auroc(yb[i], sa[i]) - auroc(yb[i], sb[i]) for i in boot])
            v = v[~np.isnan(v)]
            out["paired_differences"][f"{a} - {b}"] = {"diff": round(auroc(yb, sa) - auroc(yb, sb), 3),
                                                       "ci95": [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]}
        res["failure_probe"][lname] = out
        print(lname, json.dumps(out, indent=1), flush=True)

    (args.weights / "probe_results.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {args.weights / 'probe_results.json'}")


# ---------------------------------------------------------------- policy confound (after an outside review, 2026-10-11)

TC_STRIDE = 6  # time course: an anchor every 0.2 s along each trial


def timecourse(args):
    """z_t and z^_{t+30} at an anchor every TC_STRIDE frames along every labelled trial -> probe_timecourse.npz."""
    model, _ = E.load(args.weights, DEV)
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    a_mean, a_std = (torch.as_tensor(x) for x in action_stats(args.data, split["train_episodes"]))
    meta = json.loads(str(np.load(args.weights / "probe_features_rollouts.npz")["meta"]))
    trial, frame, c1s, zt, zh = [], [], [], [], []
    for name in sorted({m["session"] for m in meta}):
        root = args.rollouts / name
        tab, ds = frame_table(root), open_ds(root)
        for i, m in enumerate(meta):
            if m["session"] != name:
                continue
            start, act, _ = tab[m["episode"]]
            is_open = act[:, 5] > THR
            closes = np.flatnonzero(~is_open[1:] & is_open[:-1]) + 1
            ts = np.arange(C - 1, len(act) - K - 1, TC_STRIDE)
            for b in batches(ds, start + ts, workers=0):
                f = to_np(featurize(model, b, a_mean, a_std))
                zt.append(f["z_t"]); zh.append(f["zhat30"])
            trial += [i] * len(ts); frame += ts.tolist(); c1s += [int(closes[0]) if len(closes) else -1] * len(ts)
        print(f"timecourse: {name}", flush=True)
    np.savez_compressed(args.weights / "probe_timecourse.npz", trial=np.array(trial), frame=np.array(frame),
                        c1=np.array(c1s), z_t=np.concatenate(zt), zhat30=np.concatenate(zh))


def confound(args):
    """Policy-only vs latent vs policy + latent under leave-one-session-out; risk along each trial."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    r = dict(np.load(args.weights / "probe_features_rollouts.npz"))
    meta = json.loads(str(r["meta"]))
    sess = np.array([m["session"] for m in meta])
    fam = np.array([s.split("_")[1] if not s.startswith("rollout_vla_jepa") else "vla_jepa" for s in sess])
    fams = sorted(set(fam))
    onehot = (fam[:, None] == np.array(fams)[None]).astype(float)
    y = np.array([not m["success"] for m in meta])
    turbo = fam == "turbovla"

    def loso(x, dense=None):
        s = np.full(len(y), np.nan)
        dense_s = None if dense is None else np.full(len(dense[0]), np.nan)
        for g in np.unique(sess):
            te, trn = sess == g, sess != g
            if len(np.unique(y[trn])) < 2:
                continue
            sc = StandardScaler().fit(x[trn])
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(x[trn]), y[trn])
            s[te] = m.predict_proba(sc.transform(x[te]))[:, 1]
            if dense is not None:
                rows = np.isin(dense[0], np.flatnonzero(te))
                dense_s[rows] = m.predict_proba(sc.transform(dense[1][rows]))[:, 1]
        return s, dense_s

    sets = {"policy_only": onehot, "z_t": r["z_t"], "zhat30": r["zhat30"],
            "policy+z_t": np.concatenate([onehot, r["z_t"]], 1), "policy+zhat30": np.concatenate([onehot, r["zhat30"]], 1)}
    scores = {k: loso(x)[0] for k, x in sets.items()}
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(y), size=(10000, len(y)))
    def ci(f):  # noqa: E306
        v = np.array([f(i) for i in boot]); v = v[~np.isnan(v)]
        return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    res = {"families": {f: [int((fam == f).sum()), int(y[fam == f].sum())] for f in fams}, "auroc": {}, "diff": {},
           "non_turbo_only": {}}
    for k, s in scores.items():
        res["auroc"][k] = {"auroc": round(auroc(y, s), 3), "ci95": ci(lambda i, s=s: auroc(y[i], s[i]))}
        nt = ~turbo
        res["non_turbo_only"][k] = round(auroc(y[nt], s[nt]), 3)
    for a, b in (("policy+z_t", "policy_only"), ("policy+zhat30", "policy_only"), ("policy+zhat30", "policy+z_t")):
        sa, sb = scores[a], scores[b]
        res["diff"][f"{a} - {b}"] = {"diff": round(auroc(y, sa) - auroc(y, sb), 3),
                                     "ci95": ci(lambda i: auroc(y[i], sa[i]) - auroc(y[i], sb[i]))}

    # risk along each trial: the probe trained on the protocol anchors of the other sessions, applied every 0.2 s
    tc = dict(np.load(args.weights / "probe_timecourse.npz"))
    res["timecourse"] = {}
    for feat in ("z_t", "zhat30"):
        _, ds = loso(r[feat], (tc["trial"], tc[feat]))
        rel = (tc["frame"] - tc["c1"]) / FPS  # seconds relative to the first grasp close
        groups = {"turbo_fail": turbo & y, "other_fail": ~turbo & y, "turbo_success": turbo & ~y, "other_success": ~turbo & ~y}
        bins = np.arange(-6, 6.5, 1.0)
        curve = {}
        for gname, gm in groups.items():
            rows = np.isin(tc["trial"], np.flatnonzero(gm))
            curve[gname] = {"trials": int(gm.sum()),
                            "mean_risk_by_second_rel_close": [None if not ((rel >= lo) & (rel < lo + 1) & rows).any() else
                                                              round(float(np.nanmean(ds[(rel >= lo) & (rel < lo + 1) & rows])), 3)
                                                              for lo in bins]}
        # "before anything has gone wrong": first second of each trial
        first = tc["frame"] < C - 1 + FPS
        per_trial_first = np.array([np.nanmean(ds[first & (tc["trial"] == i)]) for i in range(len(meta))])
        res["timecourse"][feat] = {"bins_s_rel_first_close": bins.tolist(), "curves": curve,
                                   "auroc_fail_vs_success_first_second": round(auroc(y, per_trial_first), 3),
                                   "auroc_fail_vs_success_first_second_non_turbo": round(auroc(y[~turbo], per_trial_first[~turbo]), 3),
                                   "auroc_turbo_vs_other_first_second": round(auroc(turbo, per_trial_first), 3)}
    (args.weights / "probe_confound.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["selfcheck", "extract", "analyze", "timecourse", "confound"])
    ap.add_argument("weights", type=Path, help="dir with weights_final.pt (out/multiview of a run)")
    ap.add_argument("--data", type=Path, help="transcoded training snapshot (selfcheck, extract)")
    ap.add_argument("--rollouts", type=Path, help="dir of transcoded rollout datasets (extract)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(0)
    {"selfcheck": selfcheck, "extract": extract, "analyze": analyze, "timecourse": timecourse, "confound": confound}[args.command](args)


if __name__ == "__main__":
    main()
