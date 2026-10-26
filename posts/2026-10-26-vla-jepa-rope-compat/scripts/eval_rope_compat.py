"""VLA-JEPA rotary compatibility (BRIEF.md steps 1-2 and 4; PROTOCOL.md): a legacy-trained world predictor under the
legacy rotary, under the fixed rotary (#4835), and the copy-last baseline, on exactly the same encoder / Qwen tensors.

Runs in `lerobot-jetson:rc` with the lerobot checkout at d3793b7 first on PYTHONPATH (the image's own tree is untouched)
and the pre-fix parent 2e88044 mounted for the parity check:

    PYTHONPATH=/src/src python eval_rope_compat.py checks  --ckpt <M1 snapshot> --root <so101_bowls_all_trimmed> \\
        --split split.json --prefix-tree /prefix --out checks.json        # step 1-2: no fixed-rotary number on data
    PYTHONPATH=/src/src python eval_rope_compat.py compare --ckpt <M1 snapshot> --root <...> --split split.json \\
        --windows-per-episode 8 --out results.json                        # step 4, after CHECKS.md is committed

Qwen3-VL and V-JEPA 2 are built from their configs (their weights are in the checkpoint and the Orin has no room for a
second copy); the checkpoint then fills every tensor, checked key by key (strict).
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CAM_MAP = {"observation.images.front_top_camera": "observation.images.exterior_1_left",
           "observation.images.wrist_camera": "observation.images.exterior_2_left"}


# ---- rotary variants --------------------------------------------------------------------------------------------
def rope_legacy(x, pos):
    """Pre-#4835 expansion (what every existing checkpoint was trained with): the frequency vector is repeated
    [f0..f_{d/2-1}, f0..f_{d/2-1}], so adjacent pairs get two different frequencies."""
    _, _, _, dim = x.size()
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    freqs = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freqs.sin().squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = freqs.cos().squeeze(-1).repeat(1, 1, 1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return x * emb_cos + y * emb_sin


def set_mode(mode):
    import lerobot.policies.vla_jepa.world_model as W

    if not hasattr(W, "_rope_fixed"):
        W._rope_fixed = W.rotate_queries_or_keys
    W.rotate_queries_or_keys = rope_legacy if mode == "legacy" else W._rope_fixed


# ---- loading ----------------------------------------------------------------------------------------------------
def patch_loaders():
    import transformers
    from transformers import AutoConfig

    import lerobot.policies.vla_jepa.modeling_vla_jepa as M
    import lerobot.policies.vla_jepa.qwen_interface as Q

    class QwenFromConfig:
        @staticmethod
        def from_pretrained(name, torch_dtype=None, **_):
            return transformers.Qwen3VLForConditionalGeneration._from_config(AutoConfig.from_pretrained(name),
                                                                               torch_dtype=torch_dtype)

    class AutoModelFromConfig:
        @staticmethod
        def from_pretrained(name, torch_dtype=None, **_):
            return transformers.AutoModel.from_config(AutoConfig.from_pretrained(name), torch_dtype=torch_dtype)

    Q.Qwen3VLForConditionalGeneration = QwenFromConfig
    M.AutoModel = AutoModelFromConfig


def load(ckpt):
    from safetensors import safe_open

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    patch_loaders()
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.enable_world_model, cfg.device = True, "cuda"
    pol = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).eval()
    with safe_open(str(Path(ckpt) / "model.safetensors"), "pt") as f:
        ck = set(f.keys())
    sd = set(pol.state_dict())
    wm = ("model.video_predictor.", "model.video_encoder.")
    keys = {"missing_world_model": sorted(k for k in sd - ck if k.startswith(wm)),
            "unexpected_world_model": sorted(k for k in ck - sd if k.startswith(wm)),
            "missing_other": sorted(k for k in sd - ck if not k.startswith(wm)),
            "unexpected_other": sorted(k for k in ck - sd if not k.startswith(wm)),
            "predictor_tensors": sum(k.startswith("model.video_predictor.") for k in ck),
            "encoder_tensors": sum(k.startswith("model.video_encoder.") for k in ck)}
    # the loaded values must be the checkpoint's (a strict check by value on the predictor, not only by name)
    with safe_open(str(Path(ckpt) / "model.safetensors"), "pt") as f:
        psd = pol.state_dict()
        bad = [k for k in ck if k.startswith(wm) and not torch.equal(f.get_tensor(k).to(psd[k].dtype).cpu(),
                                                                       psd[k].cpu())]
    keys["value_mismatch_world_model"] = bad[:10]
    pre, _ = make_pre_post_processors(cfg, pretrained_path=ckpt,
                                      preprocessor_overrides={"device_processor": {"device": "cuda"},
                                                              "rename_observations_processor": {"rename_map": CAM_MAP}})
    return cfg, pol, pre, keys


def dataset(root, cfg):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    d = cfg.observation_delta_indices
    ds = LeRobotDataset(f"local/{Path(root).name}", root=root, video_backend="pyav",
                        delta_timestamps={c: [i / 30 for i in d] for c in CAM_MAP})
    eps = ds.meta.episodes
    spans = {int(eps["episode_index"][k]): (int(eps["dataset_from_index"][k]), int(eps["dataset_to_index"][k]))
             for k in range(len(eps["episode_index"]))}
    return ds, d, spans


def windows(spans, episodes, n, last_delta):
    """PROTOCOL.md: n start frames per episode, uniform over the valid range, numpy default_rng(0) per episode index.
    `default_rng(0) per episode index` is read as default_rng([0, episode]) (one stream per episode, seeded by both)."""
    out = []
    for e in episodes:
        lo, hi = spans[e]
        valid = hi - lo - last_delta                  # start s needs s + last_delta < length
        rng = np.random.default_rng([0, e])
        out += [(e, int(s)) for s in np.sort(rng.integers(0, valid, n))]
    return out


# ---- one window: encoders once, then the predictor per arm --------------------------------------------------------
@torch.no_grad()
def encode(pol, pre, item):
    m = pol.model
    b = {k: (v[None] if torch.is_tensor(v) else [v] if isinstance(v, str) else v) for k, v in item.items()}
    inputs = pol._prepare_model_inputs(pre(b), training=True)
    _, action_tokens = m._encode_qwen(inputs["images"], inputs["instructions"], need_action_tokens=True)
    videos = inputs["videos"]
    nv = m.config.num_world_model_views
    if videos.shape[1] < nv:
        videos = torch.cat([videos, videos[:, :1].repeat(1, nv - videos.shape[1], 1, 1, 1, 1)], dim=1)
    elif videos.shape[1] > nv:
        videos = videos[:, :nv]
    bb, v, t, c, h, w = videos.shape
    px = m.video_processor(videos=list(videos.reshape(bb * v, t, c, h, w)), return_tensors="pt",
                           device=m.video_encoder.device, do_rescale=False)["pixel_values_videos"]
    tub = m.video_encoder.config.tubelet_size
    emb = m._merge_views(m.video_encoder.get_vision_features(pixel_values_videos=px), bb, v)
    t_tot = m.config.num_video_frames // tub
    t_ctx = t_tot - 1
    tpf = emb.shape[1] // t_tot
    if m.config.causal_world_model_context:
        inp = m._merge_views(m._causal_video_embeddings(px, tub, t_ctx), bb, v)
    else:
        inp = emb[:, : tpf * t_ctx, :]
    gt = emb[:, tpf:, :]
    na = t_ctx * m.config.num_action_tokens_per_timestep
    if action_tokens.shape[1] < na:
        action_tokens = torch.cat([action_tokens, action_tokens[:, -1:].repeat(1, na - action_tokens.shape[1], 1)], 1)
    return inp.float(), gt.float(), action_tokens[:, :na].float(), tpf


@torch.no_grad()
def predict(predictor, inp, act, mode):
    set_mode(mode)
    return predictor(inp, act)


def l1(a, b):
    return float(F.l1_loss(a, b, reduction="mean"))


def per_position(a, b, tpf):
    """Exploratory: L1 per predicted temporal position."""
    n = a.shape[1] // tpf
    return [float(F.l1_loss(a[:, i * tpf:(i + 1) * tpf], b[:, i * tpf:(i + 1) * tpf])) for i in range(n)]


# ---- commands -----------------------------------------------------------------------------------------------------
def cmd_checks(a):
    t0 = time.time()
    cfg, pol, pre, keys = load(a.ckpt)
    torch.cuda.synchronize()
    res = {"versions": {"torch": torch.__version__, "lerobot_file": __import__("lerobot").__file__,
                        "transformers": __import__("transformers").__version__},
           "load_s": time.time() - t0, "strict_keys": keys,
           "causal_world_model_context": bool(cfg.causal_world_model_context),
           "observation_delta_indices": cfg.observation_delta_indices}
    if keys["missing_world_model"] or keys["unexpected_world_model"] or keys["value_mismatch_world_model"]:
        Path(a.out).write_text(json.dumps(res, indent=1))
        raise SystemExit("world-model keys missing / unexpected / not equal: stop (BRIEF step 1.3)")
    ds, d, spans = dataset(a.root, cfg)
    val = json.loads(Path(a.split).read_text())["val_episodes"]
    wins = windows(spans, val[:3], 2, d[-1])                         # 6 windows for the checks
    pred = pol.model.video_predictor
    torch.cuda.reset_peak_memory_stats()
    enc = []
    for e, s in wins:
        enc.append(encode(pol, pre, ds[spans[e][0] + s]))
    res["peak_vram_mb_batch1"] = torch.cuda.max_memory_allocated() / 2 ** 20

    # 1. parity: legacy patch on d3793b7 vs the unpatched 2e88044 predictor, same weights, same inputs (fp32)
    spec = importlib.util.spec_from_file_location("wm_prefix", Path(a.prefix_tree) / "src/lerobot/policies/vla_jepa/world_model.py")
    WP = importlib.util.module_from_spec(spec)
    sys.modules["wm_prefix"] = WP
    spec.loader.exec_module(WP)
    import lerobot.policies.vla_jepa.world_model as W

    old = rebuild_predictor(pol, WP.ActionConditionedVideoPredictor)      # 2e88044's class, unpatched
    parity = []
    old.load_state_dict(pred.state_dict(), strict=True)
    old.to(next(pred.parameters()).device).eval()
    for inp, gt, act, tpf in enc:
        p_new = predict(pred, inp, act, "legacy")
        with torch.no_grad():
            p_old = old(inp, act)
        parity.append(float((p_new - p_old).abs().max()))
    res["parity_max_abs"] = max(parity)

    # 2. determinism: legacy twice, same batch
    l_a = [l1(predict(pred, i, ac, "legacy"), g) for i, g, ac, _ in enc]
    l_b = [l1(predict(pred, i, ac, "legacy"), g) for i, g, ac, _ in enc]
    res["determinism_identical"] = l_a == l_b
    res["legacy_l1_check_windows"] = l_a
    res["copy_last_l1_check_windows"] = [l1(i, g) for i, g, _, _ in enc]

    # 3. smoke: a randomly initialised predictor (same config, seed 0) under legacy, and copy_last
    torch.manual_seed(0)
    rnd = rebuild_predictor(pol, W.ActionConditionedVideoPredictor).to(next(pred.parameters()).device).eval()
    res["random_init_legacy_l1"] = [l1(predict(rnd, i, ac, "legacy"), g) for i, g, ac, _ in enc]

    # 4. rotary sanity on random tensors (no checkpoint, no data)
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, 64, dtype=torch.float64)
    pos = torch.arange(16, dtype=torch.float64)                   # as ACRoPEAttention passes it: (N,)
    set_mode("fixed")
    yl, yf = rope_legacy(x, pos), W.rotate_queries_or_keys(x, pos)
    xp, fp = x.unflatten(-1, (-1, 2)), yf.unflatten(-1, (-1, 2))
    ang = torch.atan2(fp[..., 1] * xp[..., 0] - fp[..., 0] * xp[..., 1], (fp * xp).sum(-1))   # rotation per pair
    omega = 1.0 / 10000 ** (torch.arange(32, dtype=torch.float64) / 32)
    want = torch.remainder(pos[:, None] * omega + np.pi, 2 * np.pi) - np.pi                 # (N, d/2), one per pair
    res["rotary_sanity"] = {
        "legacy_vs_fixed_max_abs": float((yl - yf).abs().max()),
        "fixed_pair_norm_preserved_max_err": float((fp.norm(dim=-1) - xp.norm(dim=-1)).abs().max()),
        "fixed_pair_angle_vs_one_frequency_max_err": float((torch.remainder(ang - want + np.pi, 2 * np.pi) - np.pi).abs().max()),
        "legacy_pair_norm_preserved_max_err": float((yl.unflatten(-1, (-1, 2)).norm(dim=-1) - xp.norm(dim=-1)).abs().max())}

    # 5. cost: seconds per window (encoders once + legacy + fixed + copy_last), fixed-mode output discarded unseen
    times = []
    for e, s in windows(spans, val[3:5], 2, d[-1]):
        torch.cuda.synchronize(); t1 = time.perf_counter()
        i, g, ac, _ = encode(pol, pre, ds[spans[e][0] + s])
        predict(pred, i, ac, "legacy"); predict(pred, i, ac, "fixed"); l1(i, g)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t1)
    res["seconds_per_window"] = {"median": float(np.median(times)), "max": float(max(times)), "n": len(times)}
    set_mode("fixed")
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def rebuild_predictor(pol, cls):
    """A fresh predictor of class `cls` with the live predictor's constructor arguments (VLAJEPAModel.__init__)."""
    m, cfg = pol.model, pol.model.config
    enc = m.video_encoder.config
    image_size = getattr(enc, "image_size", None) or next(iter(cfg.image_features.values())).shape[-1]
    import inspect

    src = inspect.getsource(type(m).__init__)
    assert "ActionConditionedVideoPredictor(" in src
    return cls(num_frames=cfg.num_video_frames // enc.tubelet_size, img_size=(image_size, image_size), patch_size=16,
               tubelet_size=1, embed_dim=enc.hidden_size * cfg.num_world_model_views,
               action_embed_dim=m.qwen.model.config.hidden_size, predictor_embed_dim=enc.hidden_size,
               depth=cfg.predictor_depth, num_heads=cfg.predictor_num_heads, mlp_ratio=cfg.predictor_mlp_ratio,
               num_action_tokens_per_step=cfg.num_action_tokens_per_timestep)


def cmd_compare(a):
    cfg, pol, pre, keys = load(a.ckpt)
    ds, d, spans = dataset(a.root, cfg)
    val = json.loads(Path(a.split).read_text())["val_episodes"]
    pred = pol.model.video_predictor
    rows = []
    for e, s in windows(spans, val, a.windows_per_episode, d[-1]):
        i, g, ac, tpf = encode(pol, pre, ds[spans[e][0] + s])
        pl, pf = predict(pred, i, ac, "legacy"), predict(pred, i, ac, "fixed")
        lo, hi = spans[e]
        rows.append({"episode": e, "start": s, "third": int(3 * s / (hi - lo)),
                     "legacy": l1(pl, g), "fixed": l1(pf, g), "copy_last": l1(i, g),
                     "pos_legacy": per_position(pl, g, tpf), "pos_fixed": per_position(pf, g, tpf),
                     "pos_copy_last": per_position(i, g, tpf)})
        print(f"ep {e} s {s}: legacy {rows[-1]['legacy']:.4f} fixed {rows[-1]['fixed']:.4f} copy {rows[-1]['copy_last']:.4f}",
              flush=True)
    eps = sorted({r["episode"] for r in rows})
    per = {arm: np.array([np.mean([r[arm] for r in rows if r["episode"] == e]) for e in eps])
           for arm in ("legacy", "fixed", "copy_last")}
    rng = np.random.default_rng(0)
    boots = {k: [] for k in ("R", "fixed-legacy", "fixed-copy_last", "legacy-copy_last")}
    for _ in range(2000):
        ix = rng.integers(0, len(eps), len(eps))
        L, Fx, C = per["legacy"][ix].mean(), per["fixed"][ix].mean(), per["copy_last"][ix].mean()
        boots["R"].append(Fx / L); boots["fixed-legacy"].append(Fx - L)
        boots["fixed-copy_last"].append(Fx - C); boots["legacy-copy_last"].append(L - C)
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    L, Fx, C = (float(per[k].mean()) for k in ("legacy", "fixed", "copy_last"))
    summ = {"episodes": len(eps), "windows": len(rows), "mean_l1": {"legacy": L, "fixed": Fx, "copy_last": C},
            "R": Fx / L, "differences": {"fixed-legacy": Fx - L, "fixed-copy_last": Fx - C, "legacy-copy_last": L - C},
            "ci95": ci, "exploratory_frac_episodes_fixed_worse": float(np.mean(per["fixed"] > per["legacy"]))}
    Path(a.out).write_text(json.dumps({"strict_keys": keys, "summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("checks", "compare"):
        p = sub.add_parser(name)
        for k in ("ckpt", "root", "split", "out"):
            p.add_argument("--" + k, required=True)
    sub.choices["checks"].add_argument("--prefix-tree", required=True)
    sub.choices["compare"].add_argument("--windows-per-episode", type=int, required=True)
    a = ap.parse_args()
    {"checks": cmd_checks, "compare": cmd_compare}[a.cmd](a)
