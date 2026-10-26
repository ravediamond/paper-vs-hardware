"""Gate A: RoboJEPA forward-prediction on our held-out SO-101 bowls episodes (G0-RJ-robojepa-gate.md).

Run from the robo_jepa repo root (PYTHONPATH=$PWD), one CUDA GPU:

    python gate_a_eval.py --models 300m:300 100m:150 1b:300 --models-dir /data/robojepa --data /data/bowls_val \
        --encoder /data/vjepa2_1_vitG_384.pt --out-dir results/ --cache-dir cache/ --settings a b --n-clips 200

Follows scripts/evals/eval_droid_holdout.py (same loader, transforms, history selection and rollout), with the SO-101 joint-state
embodiment ('so-101_joints', id 25), 30 -> 5 Hz (6-frame stride), 16-frame clips, 240x320, ImageNet normalization. Our frames are 4:3 (480x640), so the aspect range is
fixed to 4/3 (full frame resized to 240x320, no random crop); the DROID eval's [1.333, 1.777] would crop at random.
Per prefix t (history = frames 0..t, i.e. t+1 context frames) and rollout step j = 1..9 the model predicts frame t+1+j; the L1 is the mean
absolute error over views, tokens and channels of the layer-normed encoder features (their compute_loss). Arms, same clips:
  model_true            true actions, true states fed back (the released eval; the true future state is an input of the rollout)
  model_shuffled_cons   actions of a clip of ANOTHER episode, states re-integrated from them (s_{k+1} = s_k + a_k): state-consistent
  model_shuffled_truestate  same shuffled actions, true states (isolates how much the state input carries)
  copy_last             z_t repeated (last real frame of the history; horizon j+1 steps, as the model)
Camera settings: a = front_top_camera -> view id 6 (front), wrist 2; b = front_top_camera -> 5 (top), wrist 2 (ids from the training
config view_map; there is no 'unmapped' id: the loader raises on a name that matches no view_map key).
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from app.robo_jepa.datasets.robot_video_dataset import RobotVideoDataset
from app.robo_jepa.transforms import make_transforms
from app.robo_jepa.world_model import WorldModel

DTYPE = torch.bfloat16
EMBODIMENT = "so-101_joints"
RAW_FPS, FPS, INTERMEDIATE_FRAMES_FPS = 30, 5, 30          # configs/scaling/high_compute_cooldown/*.yaml, Lerobot_joints
POSE_DIM, GRIPPER_DIM, JOINTS_DIM = 6, 1, 5
PAD_SIZE, FRAMES_PER_CLIP, MIN_LENGTH_TO_DROP = 64, 16, 16
STATE_DIMS = JOINTS_DIM + GRIPPER_DIM
ROLL = 10                                                  # rollout steps incl. the unreported 10th, as the released eval
CAMERAS = ["front_top_camera", "wrist_camera"]
VIEW_MAPS = {"a": {"front_top_camera": 6, "wrist_camera": 2}, "b": {"front_top_camera": 5, "wrist_camera": 2}}
EVAL_AUGS = {"random_horizontal_flip": False, "random_resize_aspect_ratio": [4 / 3, 4 / 3], "random_resize_scale": [1.0, 1.0],
             "reprob": 0.0, "auto_augment": False, "motion_shift": False, "crop_height": 240, "crop_width": 320}
ARMS = ("model_true", "model_shuffled_cons", "model_shuffled_truestate", "copy_last")


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_dir(models_dir, size, ckpt, encoder):
    tmp = tempfile.mkdtemp(prefix=f"wm_{size}_")
    config = json.load(open(os.path.join(models_dir, f"wm-{size}", "config.json")))
    config["attn_impl"] = "sdpa"
    json.dump(config, open(os.path.join(tmp, "config.json"), "w"), indent=2)
    os.symlink(ckpt, os.path.join(tmp, "world_model.pth.tar"))
    os.symlink(encoder, os.path.join(tmp, "encoder.pth.tar"))
    return tmp


def encode_clips(encoder, data, setting, n_clips, seed, device):
    ds = RobotVideoDataset(
        dataset_name="bowls_val", data_path=None, cameras=CAMERAS, view_map=VIEW_MAPS[setting], embodiment_name=EMBODIMENT,
        raw_fps=RAW_FPS, fps=FPS, pose_dim=POSE_DIM, gripper_dim=GRIPPER_DIM, joints_dim=JOINTS_DIM,
        intermediate_frames_fps=INTERMEDIATE_FRAMES_FPS, s3_profile=None, s3_prefix=data.rstrip("/") + "/", pad_size=PAD_SIZE,
        frames_per_clip=FRAMES_PER_CLIP, min_length_to_drop=MIN_LENGTH_TO_DROP, min_size=n_clips,
        transform=make_transforms(**EVAL_AUGS), read_intermediate_frames=False, action_source="state_delta", use_joint_state=True)
    seed_all(seed)                                           # same clip windows for both settings (same seed, same sampler)
    clips = []
    for i in range(n_clips):
        (rgb, _, action, state, extr, pmask, view_list, dt, start, name, emb, _) = ds[i]
        video = rgb.unsqueeze(0).to(device, dtype=DTYPE)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
            import einops
            b, _, v, t = video.shape[:4]
            h_, w_ = (s // encoder.vit.patch_size for s in video.shape[-2:])
            flat = einops.repeat(einops.rearrange(video, "b c v t h w -> (b t v) c h w"), "b c h w -> b c t h w", t=1)
            feats = einops.rearrange(encoder(flat.float()), "(b t v) (h w) c -> b t v h w c", b=b, t=t, v=v, h=h_, w=w_)
        clips.append({"features": feats.to(DTYPE).cpu(), "action": action.unsqueeze(0), "state": state.unsqueeze(0),
                      "extrinsics": extr.unsqueeze(0), "padding_mask": pmask.unsqueeze(0), "view_list": view_list.unsqueeze(0),
                      "dt": dt.unsqueeze(0), "episode": str(name), "start": int(start)})
    return clips


def shuffle_partner(clips, seed):
    """index of a clip from another episode (and with the same number of valid frames) for each clip."""
    rng = np.random.default_rng(seed)
    out = []
    for i, c in enumerate(clips):
        ok = [j for j, d in enumerate(clips) if d["episode"] != c["episode"] and d["action"].shape == c["action"].shape]
        out.append(int(rng.choice(ok)))
    return out


def integrate(state, action):
    """states re-integrated from `action`: s_0 true, s_{k+1} = s_k + a_k on the first STATE_DIMS slots (padding stays 0)."""
    s = torch.zeros_like(state)
    s[:, 0, :STATE_DIMS] = state[:, 0, :STATE_DIMS]
    n = min(action.shape[1], state.shape[1] - 1)
    s[:, 1:1 + n, :STATE_DIMS] = state[:, :1, :STATE_DIMS] + action[:, :n, :STATE_DIMS].cumsum(1)
    return s


def _cached_rollout():
    """the released eval's KV-cache rollout (scripts/evals/eval_droid_holdout.py): same inputs, targets and loss, up to bf16 noise."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("evd", os.path.join(os.getcwd(), "scripts/evals/eval_droid_holdout.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.cached_rollout


@torch.no_grad()
def run_clip(wm, c, partner, device, emb_ids, kv_cache=False):
    to = lambda x: x.to(device, dtype=DTYPE)  # noqa: E731
    action, state, extr, dt = to(c["action"]), to(c["state"]), to(c["extrinsics"]), to(c["dt"])
    pmask, view_list = to(c["padding_mask"]), c["view_list"].to(device)
    video_f = F.layer_norm(c["features"].to(device), (c["features"].shape[-1],)) if wm.encoder_layernorm else c["features"].to(device)
    if video_f.shape[1] == action.shape[1] + 1:
        video_f, pmask = video_f[:, :-1], pmask[:, :-1]
    shuf_a = to(partner["action"])
    arms = {"model_true": (action, state), "model_shuffled_cons": (shuf_a, integrate(state, shuf_a)),
            "model_shuffled_truestate": (shuf_a, state)}
    total = int(pmask.sum(1).min().item())
    # as the released eval: a prefix counts only if >= ROLL target frames follow it (here t = 0..4)
    ts = [t for t in range(total + 1 - ROLL) if video_f.shape[1] - (t + 2) >= ROLL]
    res = {k: [] for k in ARMS}
    for name, (act, st) in arms.items():
        with torch.amp.autocast("cuda", dtype=DTYPE):
            pred, _, pred_st, _ = wm.forward(video_f, act, st, extr, view_map=view_list, dt=dt, embodiment_ids=emb_ids)
            for t in ts:
                hist = wm.select_history(video_f, act, pred, states=st, predicted_states=pred_st, camera_extrinsics=extr,
                                         dt=dt, prefix_len=t + 1, suffix_len=1)
                fs = video_f[:, t + 2:]
                if kv_cache:
                    cl = _cached_rollout()(wm, hist, act, st, dt, video_f, pmask, t, view_list, emb_ids)
                    res[name].append([float(x) for x in cl["l1_loss"]])
                    continue
                losses, _ = wm.rollout(hist, actions_suffix=act[:, t + 1:], features_suffix=fs if fs.shape[1] >= ROLL else None,
                                       states_suffix=st[:, t + 2:], dt=dt[:, t + 2:], action_noise=0.0, suffix_len=1,
                                       run_backward=False, padding_mask=pmask[:, t + 1:-1], rollout_steps=ROLL - 1,
                                       view_map=view_list, embodiment_ids=emb_ids)
                res[name].append([float(x) for x in losses["l1_loss"]])
    for t in ts:                                             # copy-last: step j targets frame t+1+j, prediction = frame t
        res["copy_last"].append([float((video_f[:, t] - video_f[:, t + 1 + j]).abs().float().mean()) for j in range(1, ROLL)])
    return res                                               # arm -> [prefix t][j-1]


def load_wm(models_dir, size, step, encoder_file, load_encoder, device):
    ckpt = os.path.join(models_dir, f"wm-{size}", f"cooldown_{size}_{step}k.pth.tar")
    tmp = load_dir(models_dir, size, ckpt, encoder_file)
    try:
        wm = WorldModel.load_pretrained(tmp, load_decoder=False, load_encoder=load_encoder, device=device)
    finally:
        shutil.rmtree(tmp)
    wm.eval()
    wm.predictor_window_steps = 1
    wm.eval_rollout_steps = ROLL
    wm.rollout_steps = 2
    wm.rollout_stop_gradient = True
    return wm, ckpt


def summarize(per_clip):
    """arm -> {mean_1_9, step9, ctx2_mean_1_9, per_horizon}; ctx2 = prefix t=1 only (2 real context frames)."""
    out = {}
    for k in ARMS:
        per = np.array([np.mean(c["l1"][k], axis=0) for c in per_clip])                        # clips x 9 (mean over prefixes)
        ctx2 = np.array([c["l1"][k][1] for c in per_clip if len(c["l1"][k]) > 1])
        out[k] = {"per_horizon": per.mean(0).tolist(), "mean_1_9": float(per.mean()), "step9": float(per[:, 8].mean()),
                  "ctx2_mean_1_9": float(ctx2.mean()), "ctx2_step9": float(ctx2[:, 8].mean())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="size:step, e.g. 300m:300 100m:150 1b:300 (first one loads the encoder)")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", help="encoder features are saved here once and reused by every predictor")
    ap.add_argument("--settings", nargs="+", default=["a", "b"], choices=["a", "b"])
    ap.add_argument("--n-clips", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-clips", type=int, help="smoke only")
    ap.add_argument("--kv-cache", action="store_true", help="roll out through the KV cache (released eval option, same numbers up to bf16 noise)")
    ap.add_argument("--check-cache", type=int, default=0, help="compare cached vs uncached on this many clips and print the max difference")
    a = ap.parse_args()
    device = "cuda"
    torch.cuda.set_device(0)
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    models = [m.split(":") for m in a.models]
    wm, ckpt = load_wm(a.models_dir, models[0][0], models[0][1], a.encoder, True, device)
    encoder = wm.encoder
    clips = {}
    for setting in a.settings:                                # V-JEPA 2.1-G encoding happens once per setting, for all predictors
        cache = os.path.join(a.cache_dir, f"clips_{setting}_{a.n_clips}_{a.seed}.pt") if a.cache_dir else None
        if cache and os.path.exists(cache):
            clips[setting] = torch.load(cache)
        else:
            clips[setting] = encode_clips(encoder, a.data, setting, a.n_clips, a.seed, device)
            if cache:
                os.makedirs(a.cache_dir, exist_ok=True)
                torch.save(clips[setting], cache)
        print(f"encoded setting {setting}: {len(clips[setting])} clips ({time.time() - t0:.0f}s)", flush=True)
    enc_sha = sha256(a.encoder)
    episodes = json.load(open(os.path.join(a.data, "episodes.json")))["episodes"]
    for mi, (size, step) in enumerate(models):
        if mi:
            del wm
            torch.cuda.empty_cache()
            wm, ckpt = load_wm(a.models_dir, size, step, a.encoder, False, device)
            wm.encoder = encoder
        emb_list = list(wm.model.embodiment_list)
        emb_ids_one = torch.tensor([emb_list.index(EMBODIMENT)], device=device)
        meta = {"size": size, "step_k": int(step), "checkpoint": os.path.basename(ckpt), "checkpoint_bytes": os.path.getsize(ckpt),
                "checkpoint_sha256": sha256(ckpt), "encoder": os.path.basename(a.encoder), "encoder_sha256": enc_sha,
                "robo_jepa_commit": sh("git", "rev-parse", "HEAD"), "gate_commit": os.environ.get("GATE_COMMIT", "n/a"),
                "embodiment": EMBODIMENT, "embodiment_id": emb_list.index(EMBODIMENT), "raw_fps": RAW_FPS, "fps": FPS,
                "frames_per_clip": FRAMES_PER_CLIP, "n_clips": a.n_clips, "seed": a.seed, "torch": torch.__version__,
                "gpu": sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"), "episodes": episodes}
        for setting in a.settings:
            cl = clips[setting]
            partners = shuffle_partner(cl, a.seed)
            if a.check_cache and mi == 0 and setting == a.settings[0]:
                diffs = []
                for i in range(a.check_cache):
                    r0 = run_clip(wm, cl[i], cl[partners[i]], device, emb_ids_one, False)
                    r1 = run_clip(wm, cl[i], cl[partners[i]], device, emb_ids_one, True)
                    diffs.append(max(float(np.abs(np.array(r0[k]) - np.array(r1[k])).max()) for k in ARMS if k != "copy_last"))
                    print(f"cache check clip {i}: max |cached - uncached| = {diffs[-1]:.5f}; magnitude {np.mean(r0['model_true']):.4f}", flush=True)
            n = a.max_clips or len(cl)
            per_clip = []
            t1 = time.time()
            for i in range(n):
                r = run_clip(wm, cl[i], cl[partners[i]], device, emb_ids_one, a.kv_cache)
                per_clip.append({"episode": cl[i]["episode"], "start": cl[i]["start"], "partner_episode": cl[partners[i]]["episode"], "l1": r})
                if (i + 1) % 5 == 0:
                    print(f"{size}@{step}k setting {setting}: {i + 1}/{n} clips ({time.time() - t0:.0f}s)", flush=True)
            summ = summarize(per_clip)
            out = {**meta, "setting": setting, "view_map": VIEW_MAPS[setting], "clips": n, "summary": summ, "per_clip": per_clip,
                   "eval_seconds": time.time() - t1, "seconds_since_start": time.time() - t0}
            json.dump(out, open(os.path.join(a.out_dir, f"gateA_{size}_{step}k_setting{setting}.json"), "w"))
            print(f"{size}@{step}k setting {setting}: mean1-9 " + " ".join(f"{k}={summ[k]['mean_1_9']:.4f}" for k in ARMS), flush=True)


if __name__ == "__main__":
    main()
