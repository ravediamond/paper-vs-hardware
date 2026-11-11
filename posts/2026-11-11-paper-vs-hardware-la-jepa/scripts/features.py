"""WS-167 step 1: the precomputed feature store (paper Appendix B: encode every demonstration once with the frozen V-JEPA 2.1 ViT-L).

  python features.py build  --data DIR --out DIR --encoder vjepa21|stub [--stride 2]
  python features.py text   --data DIR --out DIR [--stub]
  python features.py stats  --data DIR --out DIR
  python features.py check  --data DIR --out DIR       (shapes + one hand re-encode of a stored token block)

--data is a projects/gawm/prepare.py output (arrays.npz, episodes.json, frames_{top,wrist}.npy: N x 256 x 256 x 3 uint8).
Store layout in --out:
  obs_{top,wrist}.npy   (N, 64, 1024) fp16   2-frame tube clip (t-1, t) -> one temporal grid of 16x16 tokens -> 2x2 mean pool -> 8x8 = 64 tokens
  fut_{top,wrist}.npy   (Na, 8*16, 1024) fp16  tube-motion: frames t+2, t+4, ..., t+32 (16 frames) jointly encoded -> 8 temporal grids of 16x16
                         -> 4x4 mean pool -> 8 x (4x4) = 128 tokens; only for anchor frames (local t % stride == 0)
  anchor.npy (Na,) global frame index;  fut_ok.npy (Na, 8) grid j valid iff its last frame t+4+4j <= n-1 (missing futures are masked from the loss)
  norm.npz   per-channel mean/std of obs tokens and (separately) of future tokens on the train172 anchors
  text.npz   FLAN-T5-large token embeddings of the instructions
Choices not fixed by the paper are in journal/workstack/notes/WS-167.md.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

SPLITS = Path(__file__).resolve().parent.parent / "gawm" / "splits"   # same fixed splits as WS-168
CAMS = ("top", "wrist")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CKPT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"


def split(name):
    return json.load(open(SPLITS / f"{name}.json"))["episodes"]


class StubEncoder:
    """Random patch embedding with the real interface and shapes (B,3,T,256,256) -> (B, T/2*256, 1024). Smoke tests only; never a result."""

    def __init__(self, dev):
        import torch
        g = torch.Generator().manual_seed(0)
        self.w = (torch.randn(1024, 3, 2, 16, 16, generator=g) * 0.02).to(dev)
        self.dev = dev

    def __call__(self, x):
        import torch
        import torch.nn.functional as F
        y = F.conv3d(x.float(), self.w, stride=(2, 16, 16))        # (B, 1024, T/2, 16, 16)
        return F.layer_norm(y.flatten(2).transpose(1, 2), (1024,))


def load_vjepa21(dev, repo=None, ckpt=None):
    """Official code (facebookresearch/vjepa2 @ 204698b), weights vjepa2_1_vitl_dist_vitG_384.pt, key ema_encoder (as in hub/backbones.py).
    RoPE + interpolate_rope=True lets the 384-trained encoder run at 256^2. Runs on the vast instance (code from an external repo)."""
    import torch
    repo = repo or os.environ.get("VJEPA2_REPO", "/root/g/vjepa2")
    sys.path.insert(0, repo)
    from app.vjepa_2_1.models import vision_transformer as vt
    enc = vt.vit_large(patch_size=16, img_size=(256, 256), num_frames=64, tubelet_size=2, use_sdpa=True, use_SiLU=False,
                       wide_SiLU=True, uniform_power=False, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
                       n_output_distillation=1, return_all_tokens=True)
    ckpt = ckpt or os.environ.get("VJEPA2_CKPT")
    sd = torch.load(ckpt, map_location="cpu") if ckpt else torch.hub.load_state_dict_from_url(CKPT_URL, map_location="cpu")
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd["ema_encoder"].items()}
    r = enc.load_state_dict(sd, strict=False)
    bad = [k for k in list(r.missing_keys) + list(r.unexpected_keys) if "pos_embed" not in k]
    print("vjepa2.1 load: missing", len(r.missing_keys), "unexpected", len(r.unexpected_keys), "non-pos_embed problems", bad[:8], flush=True)
    assert not bad, "encoder weights did not load cleanly"
    enc = enc.to(dev).eval().half()
    return enc


def prep(frames_u8, dev):
    """(B, T, 256, 256, 3) uint8 -> (B, 3, T, 256, 256) fp16 ImageNet-normalised"""
    import torch
    x = torch.from_numpy(frames_u8).to(dev).permute(0, 4, 1, 2, 3).float() / 255.0
    m = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1, 1)
    s = torch.tensor(STD, device=dev).view(1, 3, 1, 1, 1)
    return ((x - m) / s).half()


def encode(enc, x):
    """x (B,3,T,256,256) -> (B, T//2, 16, 16, 1024) float32"""
    import torch
    # The real encoder is half precision but its RoPE path is fp32 (interpolate_rope makes the positions float32), so q and k come out fp32 and v fp16 and
    # F.scaled_dot_product_attention rejects the mix (WS-167 vast launch 1, 2026-11-08). One autocast region makes every attention input the same dtype
    # (fp16 on cuda, bf16 on cpu); the stub encoder is already single-dtype and stays outside it.
    amp = not isinstance(enc, StubEncoder)
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.float16 if x.is_cuda else torch.bfloat16, enabled=amp):
        y = enc(x)
    if isinstance(y, (list, tuple)):
        y = y[-1]
    B, T = x.shape[0], x.shape[2]
    assert y.shape == (B, (T // 2) * 256, 1024), ("unexpected encoder output", tuple(y.shape))
    return y.float().view(B, T // 2, 16, 16, 1024)


def pool(y, k):
    """(B, T, 16, 16, C) -> (B, T, 16/k, 16/k, C) mean pool"""
    B, T, Y, X, C = y.shape
    return y.view(B, T, Y // k, k, X // k, k, C).mean(dim=(3, 5))


def clip_index(m, t, offs):
    """global frame indices of local offsets from local frame t, clamped to the episode"""
    s0, n = m["start"], m["length"]
    return s0 + np.clip(t + np.asarray(offs), 0, n - 1)


def build(a):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    meta = json.load(open(f"{a.data}/episodes.json"))
    eps = meta["episodes"]
    N = meta["total"]
    enc = StubEncoder(dev) if a.encoder == "stub" else load_vjepa21(dev)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames = {c: np.load(f"{a.data}/frames_{c}.npy", mmap_mode="r") for c in CAMS}
    anchors = np.concatenate([m["start"] + np.arange(0, m["length"], a.stride) for m in eps])
    Na = len(anchors)
    np.save(out / "anchor.npy", anchors)
    ep_of = np.concatenate([np.full(m["length"], k) for k, m in enumerate(eps)])
    fut_ok = np.zeros((Na, 8), bool)
    for i, g in enumerate(anchors):
        m = eps[ep_of[g]]
        t = g - m["start"]
        fut_ok[i] = [t + 4 + 4 * j <= m["length"] - 1 for j in range(8)]
    np.save(out / "fut_ok.npy", fut_ok)
    # resumable: progress.json holds the next unfinished row of each stage; a restart (vast container loss, 2026-11-10) reopens the files r+ and continues
    pf = out / "progress.json"
    prog = json.load(open(pf)) if pf.exists() and a.resume else {}
    mode = "r+" if prog else "w+"
    obs = {c: np.lib.format.open_memmap(out / f"obs_{c}.npy", mode, np.float16, (N, 64, 1024)) for c in CAMS}
    fut = {c: np.lib.format.open_memmap(out / f"fut_{c}.npy", mode, np.float16, (Na, 128, 1024)) for c in CAMS}

    def commit(key, mm, nxt):
        mm.flush(); prog[key] = int(nxt)
        json.dump(prog, open(str(pf) + ".tmp", "w")); os.replace(str(pf) + ".tmp", pf)
    t0 = time.time()
    B = a.bs
    # observations: every frame, 2-frame tube clip ending at the current frame
    for c in CAMS:
        for lo in range(prog.get(f"obs_{c}", 0), N, B):
            g = np.arange(lo, min(N, lo + B))
            idx = np.stack([[clip_index(eps[ep_of[x]], x - eps[ep_of[x]]["start"], [-1, 0])] for x in g])[:, 0]  # (b, 2)
            x = np.stack([frames[c][i] for i in idx.reshape(-1)]).reshape(len(g), 2, 256, 256, 3)
            y = pool(encode(enc, prep(x, dev)), 2)                 # (b, 1, 8, 8, 1024)
            obs[c][g] = y.flatten(1, 3).cpu().numpy().astype(np.float16)
            if (lo // B) % 50 == 0:
                print(f"obs {c} {lo}/{N} {time.time() - t0:.0f}s", flush=True)
            if (lo // B) % 50 == 49: commit(f"obs_{c}", obs[c], lo + B)
        commit(f"obs_{c}", obs[c], N)
    # futures: tube-motion at the anchors, 16 frames t+2 ... t+32 jointly encoded
    bf = max(1, B // 8)
    for c in CAMS:
        for lo in range(prog.get(f"fut_{c}", 0), Na, bf):
            ii = np.arange(lo, min(Na, lo + bf))
            idx = np.stack([clip_index(eps[ep_of[anchors[i]]], anchors[i] - eps[ep_of[anchors[i]]]["start"], 2 + 2 * np.arange(16)) for i in ii])
            x = np.stack([frames[c][j] for j in idx.reshape(-1)]).reshape(len(ii), 16, 256, 256, 3)
            y = pool(encode(enc, prep(x, dev)), 4)                 # (b, 8, 4, 4, 1024)
            fut[c][ii] = y.flatten(1, 3).cpu().numpy().astype(np.float16)
            if (lo // bf) % 50 == 0:
                print(f"fut {c} {lo}/{Na} {time.time() - t0:.0f}s", flush=True)
            if (lo // bf) % 50 == 49: commit(f"fut_{c}", fut[c], lo + bf)
        commit(f"fut_{c}", fut[c], Na)
    json.dump({"stride": a.stride, "N": int(N), "Na": int(Na), "encoder": a.encoder, "vjepa2_commit": "204698b45b3712590f06245fbfba32d3be539812",
               "ckpt": CKPT_URL, "sec": round(time.time() - t0, 1)}, open(out / "store.json", "w"))
    print("build done", round(time.time() - t0), "s", flush=True)


def stats(a):
    meta = json.load(open(f"{a.data}/episodes.json"))
    by = {m["orig"]: k for k, m in enumerate(meta["episodes"])}
    train = [e for e in split("train172") if e in by]
    out = Path(a.out)
    anchors = np.load(out / "anchor.npy")
    ep_of = np.concatenate([np.full(m["length"], k) for k, m in enumerate(meta["episodes"])])
    tr = np.isin(ep_of[anchors], [by[e] for e in train])
    rows = np.where(tr)[0]
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(rows, min(len(rows), 3000), replace=False))
    res = {}
    for kind in ("obs", "fut"):
        s = np.zeros(1024); s2 = np.zeros(1024); n = 0
        for c in CAMS:
            arr = np.load(out / f"{kind}_{c}.npy", mmap_mode="r")
            sel = anchors[rows] if kind == "obs" else rows
            x = np.asarray(arr[sel]).astype(np.float64).reshape(-1, 1024)
            s += x.sum(0); s2 += (x ** 2).sum(0); n += len(x)
        mu = s / n
        res[f"{kind}_mean"] = mu.astype(np.float32)
        res[f"{kind}_std"] = np.sqrt(np.maximum(s2 / n - mu ** 2, 1e-8)).astype(np.float32)
    np.savez(out / "norm.npz", **res)
    print("norm", {k: (float(v.mean()), float(v.min())) for k, v in res.items()})


def text(a):
    meta = json.load(open(f"{a.data}/episodes.json"))
    texts = sorted({m["task"] for m in meta["episodes"]})
    L = 32
    if a.stub:
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((len(texts), L, 1024)).astype(np.float16) * 0.1
        mask = np.zeros((len(texts), L), bool); mask[:, :8] = True
    else:
        import torch
        from transformers import AutoTokenizer, T5EncoderModel
        tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
        m = T5EncoderModel.from_pretrained("google/flan-t5-large").eval()
        t = tok(texts, padding="max_length", truncation=True, max_length=L, return_tensors="pt")
        with torch.no_grad():
            emb = m(**t).last_hidden_state.numpy().astype(np.float16)
        mask = t["attention_mask"].numpy().astype(bool)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(a.out) / "text.npz", texts=np.array(texts), emb=emb, mask=mask)
    print("text", texts, emb.shape, mask.sum(1))


def check(a):
    import torch
    out = Path(a.out)
    st = json.load(open(out / "store.json"))
    for c in CAMS:
        o = np.load(out / f"obs_{c}.npy", mmap_mode="r"); f = np.load(out / f"fut_{c}.npy", mmap_mode="r")
        print(c, "obs", o.shape, o.dtype, "fut", f.shape, "finite", bool(np.isfinite(np.asarray(o[:64])).all() and np.isfinite(np.asarray(f[:8])).all()))
    dev = torch.device("cpu")
    enc = StubEncoder(dev) if st["encoder"] == "stub" else load_vjepa21(torch.device("cuda"))
    meta = json.load(open(f"{a.data}/episodes.json")); m = meta["episodes"][-1]
    g = m["start"] + 5
    fr = np.load(f"{a.data}/frames_top.npy", mmap_mode="r")
    x = np.stack([fr[g - 1], fr[g]])[None]
    y = pool(encode(enc, prep(x, enc.dev if hasattr(enc, "dev") else torch.device("cuda"))), 2).flatten(1, 3).cpu().numpy()[0]
    stored = np.load(out / "obs_top.npy", mmap_mode="r")[g].astype(np.float32)
    print("hand re-encode vs stored obs token block: max abs diff", float(np.abs(y - stored).max()), "(fp16 rounding expected)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "stats", "text", "check"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="vjepa21", choices=["vjepa21", "stub"])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--resume", action="store_true", help="build: continue from out/progress.json (after a lost container)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--stub", action="store_true")
    a = ap.parse_args()
    {"build": build, "stats": stats, "text": text, "check": check}[a.cmd](a)
