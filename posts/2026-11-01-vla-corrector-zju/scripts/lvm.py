#!/usr/bin/env python3
"""WS-087, PROTOCOL.md (frozen b8baece): VLA-Corrector's (2607.01804) latent-space vision monitor on our rollouts.

Inside lerobot-jetson:public on the Orin (GPU):
    python lvm.py features --root DEMOS_224 --out feats/demos [--episodes all]
    python lvm.py features --root ROLLOUTS --out feats/rollouts --resize224 --episodes g16
    python lvm.py train --feats feats/demos --split SPLIT.json --out model.pt
    python lvm.py score --feats feats/rollouts --model model.pt --g16 g16b_episodes.csv --onsets onsets.csv --out results/
"""
import argparse, csv, glob, json, math
from pathlib import Path

import numpy as np

POLICY, REV = "ravediamond/smolvla_2026-09-24_14-34-40", "c1d76f6"
CAMS = ("observation.images.front_top_camera", "observation.images.wrist_camera")
STRIDE = 2  # deviation 1 (PROTOCOL.md): features at 15 Hz for compute; K counts samples, 5 samples = the paper's 10 frames
K, W, L_ON, L_OFF, P = 10 // STRIDE, 15, 3.0, 2.0, 5
WIDTH, BLOCKS, BETA = 2048, 4, 1.0


# ------------------------------------------------------------------------------------------------------ features
def frames(path, t0, t1, fps, resize):
    import av
    out = []
    n = int(round((t1 - t0) * fps))
    with av.open(path) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"
        c.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            if round(t * fps) < round(t0 * fps):
                continue
            if len(out) >= n:
                break
            if resize:
                fr = fr.reformat(width=224, height=224, interpolation="AREA")
            out.append(fr.to_ndarray(format="rgb24"))
    return np.stack(out)


def cmd_features(a):
    import pandas as pd, torch
    from huggingface_hub import snapshot_download
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    pol = SmolVLAPolicy.from_pretrained(snapshot_download(POLICY, revision=REV)).to("cuda").eval()
    vm = pol.model.vlm_with_expert.get_vlm_model(); vm.vision_model.to(torch.bfloat16); vm.connector.to(torch.bfloat16)  # deviation 2
    info = json.load(open(f"{a.root}/meta/info.json")); fps = info["fps"]
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.root}/meta/episodes/*/*.parquet"))])
    data = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action"])
                      for f in sorted(glob.glob(f"{a.root}/data/*/*.parquet"))])
    want = None
    if a.episodes == "g16":
        want = {int(r["episode_index"]) for r in csv.DictReader(open(a.g16)) if r["in_g16_set"] == "True"}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for _, r in eps.sort_values("episode_index").iterrows():
        ep = int(r.episode_index)
        if (want is not None and ep not in want) or (out / f"ep{ep}.npz").exists():
            continue
        fr = {}
        for cam in CAMS:
            path = f"{a.root}/" + info["video_path"].format(video_key=cam, chunk_index=int(r[f"videos/{cam}/chunk_index"]),
                                                              file_index=int(r[f"videos/{cam}/file_index"]))
            fr[cam] = frames(path, float(r[f"videos/{cam}/from_timestamp"]), float(r[f"videos/{cam}/to_timestamp"]), fps, a.resize224)[::STRIDE]
        n0 = min(len(v) for v in fr.values()); zs = {c: [] for c in CAMS}
        with torch.no_grad():
            for i in range(0, n0, 32):
                batch = {c: torch.from_numpy(fr[c][i:i + 32]).permute(0, 3, 1, 2).float().div(255).to("cuda") for c in CAMS}
                imgs, _ = pol.prepare_images(batch)
                for c, img in zip(CAMS, imgs):
                    zs[c].append(pol.model.vlm_with_expert.embed_image(img).float().mean(1).cpu().numpy())
        Z = [np.concatenate(zs[c]) for c in CAMS]
        n = min(len(z) for z in Z)
        act = np.stack(data[data.episode_index == ep].sort_values("frame_index").action.to_numpy())[::STRIDE][:n]
        np.savez(out / f"ep{ep}.npz", Z=np.concatenate([z[:n] for z in Z], 1).astype(np.float16), action=act.astype(np.float32))
        print(f"episode {ep}: {n} frames", flush=True)


# ------------------------------------------------------------------------------------------------------ corrector
def model(zdim, adim):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(s):
            super().__init__(); s.f = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH))

        def forward(s, x):
            return x + s.f(x)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.a = nn.Linear(adim, 256); s.i = nn.Linear(zdim + 256, WIDTH)
            s.b = nn.Sequential(*[Block() for _ in range(BLOCKS)]); s.o = nn.Linear(WIDTH, zdim)

        def forward(s, z, a):
            return s.o(s.b(s.i(__import__("torch").cat([z, s.a(a)], -1))))
    return M()


def pairs(files):
    Z, A, D = [], [], []
    for f in files:
        d = np.load(f); z = d["Z"].astype(np.float32); a = d["action"]
        Z.append(z[:-K]); A.append(a[:-K]); D.append(z[K:] - z[:-K])
    return np.concatenate(Z), np.concatenate(A), np.concatenate(D)


def loss_fn(p, t):
    import torch.nn.functional as F
    return ((p - t) ** 2).sum(-1).mean() + BETA * (1 - F.cosine_similarity(p, t, dim=-1)).mean()


def cmd_train(a):
    import torch
    sp = json.load(open(a.split)); torch.manual_seed(0)
    tr = pairs([f"{a.feats}/ep{e}.npz" for e in sp["train_episodes"]]); va = pairs([f"{a.feats}/ep{e}.npz" for e in sp["val_episodes"]])
    am, asd = tr[1].mean(0), tr[1].std(0) + 1e-6
    T = [torch.from_numpy(x).cuda() for x in tr]; V = [torch.from_numpy(x).cuda() for x in va]
    norm = lambda x: (x - torch.from_numpy(am).cuda()) / torch.from_numpy(asd).cuda()  # noqa: E731
    m = model(T[0].shape[1], T[1].shape[1]).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = math.ceil(len(T[0]) / 512); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30 * steps, eta_min=3e-6)
    best, bad, log = 1e9, 0, []
    for ep in range(30):
        m.train(); perm = torch.randperm(len(T[0]), device="cuda")
        for i in range(steps):
            j = perm[i * 512:(i + 1) * 512]
            l = loss_fn(m(T[0][j], norm(T[1][j])), T[2][j]); opt.zero_grad(); l.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_fn(m(V[0][i:i + 2048], norm(V[1][i:i + 2048])), V[2][i:i + 2048]).item()
                                for i in range(0, len(V[0]), 2048)]))
        log.append(vl); print(f"epoch {ep}: val {vl:.4f}", flush=True)
        if vl < best:
            best, bad = vl, 0; torch.save({"state": m.state_dict(), "am": am, "asd": asd, "zdim": T[0].shape[1], "adim": T[1].shape[1]}, a.out)
        else:
            bad += 1
            if bad >= 5:
                break
    json.dump({"val_loss": log, "best": best, "n_train_pairs": len(T[0]), "n_val_pairs": len(V[0])}, open(a.out + ".json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------------ monitor
def trigger(E):
    """Interrupt frames under App. B.2: window W, T_on/T_off from median + MAD, patience P."""
    c, out = 0, []
    for t in range(W, len(E)):
        win = E[t - W:t]; me = np.median(win); mad = np.median(np.abs(win - me))
        if E[t] > me + L_ON * mad:
            c += 1
        elif E[t] < me + L_OFF * mad:
            c = 0
        if c >= P:
            out.append(t); c = 0
    return out


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float); n1, n0 = y.sum(), (~y).sum()
    r = np.argsort(np.argsort(s)) + 1.0
    for v in np.unique(s):  # average ranks for ties
        r[s == v] = r[s == v].mean()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cmd_score(a):
    import torch
    ck = torch.load(a.model, weights_only=False); m = model(ck["zdim"], ck["adim"]).cuda().eval(); m.load_state_dict(ck["state"])
    rows = [r for r in csv.DictReader(open(a.g16)) if r["in_g16_set"] == "True"]
    on = {int(r["episode_index"]): int(r["onset_frame"]) for r in csv.DictReader(open(a.onsets)) if r.get("onset_frame", "").strip().lstrip("-").isdigit()} if a.onsets else {}
    rng = np.random.default_rng(0)
    acts = {int(r["episode_index"]): np.load(f"{a.feats}/ep{r['episode_index']}.npz")["action"] for r in rows}

    def E_of(z, act):
        with torch.no_grad():
            zt = torch.from_numpy(z[:-K]).cuda(); at = (torch.from_numpy(act[: len(z) - K]).cuda() - torch.from_numpy(ck["am"]).cuda()) / torch.from_numpy(ck["asd"]).cuda()
            pred = m(zt, at); obs = torch.from_numpy(z[K:] - z[:-K]).cuda()
            return (1 - torch.nn.functional.cosine_similarity(pred, obs, dim=-1)).cpu().numpy()

    res = []
    for r in rows:
        ep = int(r["episode_index"]); d = np.load(f"{a.feats}/ep{ep}.npz"); z = d["Z"].astype(np.float32)
        E = E_of(z, d["action"]); ints = trigger(E)
        same = [e for e in acts if e != ep and next(x for x in rows if int(x["episode_index"]) == e)["policy"] == r["policy"]]
        other = acts[same[rng.integers(len(same))]]
        sh = np.resize(other, (len(z), other.shape[1]))
        Es = E_of(z, sh); ints_s = trigger(Es)
        res.append({"episode_index": ep, "policy": r["policy"], "failure": r["outcome"] == "failure", "frames": len(z),
                    "flagged": bool(ints), "first_interrupt": ints[0] if ints else None, "n_interrupts": len(ints),
                    "max_E": float(E.max()), "flagged_shuffled": bool(ints_s), "max_E_shuffled": float(Es.max()),
                    "onset": on.get(ep)})
    F = [x for x in res if x["failure"]]; S = [x for x in res if not x["failure"]]
    kf, ks = sum(x["flagged"] for x in F), sum(x["flagged"] for x in S)
    out = {"n_failures": len(F), "n_successes": len(S), "failures_flagged": kf, "successes_flagged": ks,
           "wilson_failures": wilson(kf, len(F)), "wilson_successes": wilson(ks, len(S)),
           "pass": kf >= 5 and ks <= 2,
           "auroc_maxE": auroc([x["failure"] for x in res], [x["max_E"] for x in res]),
           "shuffled": {"failures_flagged": sum(x["flagged_shuffled"] for x in F), "successes_flagged": sum(x["flagged_shuffled"] for x in S),
                        "auroc_maxE": auroc([x["failure"] for x in res], [x["max_E_shuffled"] for x in res])},
           "flagged_failures_before_onset": sum(1 for x in F if x["flagged"] and x["onset"] is not None and (x["first_interrupt"] + K) * STRIDE < x["onset"]),
           "interrupts_per_min_successes": sum(x["n_interrupts"] for x in S) / (sum(x["frames"] for x in S) * STRIDE / 30 / 60)}
    rng2 = np.random.default_rng(0); y = np.array([x["failure"] for x in res]); s = np.array([x["max_E"] for x in res]); v = []
    for _ in range(2000):
        i = rng2.integers(0, len(y), len(y))
        if y[i].any() and (~y[i]).any():
            v.append(auroc(y[i], s[i]))
    out["auroc_maxE_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"{a.out}/lvm_summary.json", "w"), indent=1); json.dump(res, open(f"{a.out}/lvm_episodes.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features"); f.add_argument("--root", required=True); f.add_argument("--out", required=True)
    f.add_argument("--resize224", action="store_true"); f.add_argument("--episodes", default="all"); f.add_argument("--g16")
    t = sub.add_parser("train"); t.add_argument("--feats", required=True); t.add_argument("--split", required=True); t.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--feats", required=True); s.add_argument("--model", required=True)
    s.add_argument("--g16", required=True); s.add_argument("--onsets"); s.add_argument("--out", required=True)
    a = ap.parse_args(); {"features": cmd_features, "train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
