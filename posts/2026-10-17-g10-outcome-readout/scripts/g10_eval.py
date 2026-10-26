#!/usr/bin/env python3
"""G10 evaluation: Can LeWM anticipate the colour judge? (G10_PROTOCOL.md)

Evaluates W1 and W2 against the G9c-3 judge labels on the 88 test rollouts:
1. Step 1: Readout AUROC at judge frame t_j (W1 vs W2 top vs W2 top+wrist).
2. Step 2: Anticipation AUROC at t_j - 30 (z_{t_j-30} vs raw actions vs imagined \hat{z}_{t_j} vs corrupted vs policy).
3. Mid-chunk surprise error: ||z_t - \hat{z}_t|| curves comparing successes vs failure modes.

Leave-one-session-out cross-validation with L2 logistic regression (C=1.0, balanced, StandardScaler).
Bootstrap 95% CIs (10,000 resamples, seed 0).
"""

import os
import sys
import glob
import json
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import hydra
from torch import nn
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[3]
WM_DIR = Path(__file__).resolve().parent
RESULTS_DIR = WM_DIR / "results" / "g10"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ORIN_MERGED = Path("<local path>")
WM_ROLLOUTS = Path("<local path>")
M1_CACHE = Path(os.path.expanduser("~/.cache/huggingface/lerobot/ravediamond"))
EPISODE_MAP_PATH = REPO / "data" / "rollouts" / "so101-bowls-policy-rollouts_episode_map.csv"
G9C3_SCORES_PATH = REPO / "projects" / "vla-jepa-world-model" / "judge" / "results" / "g9c3" / "g9c3_test_scores.csv"
W1_CKPT_PATH = Path("<local path>")

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEV}")

# ImageNet normalization for ViT
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Action normalization stats for W1 from training set
ACTION_MEAN = torch.tensor([3.823351, -32.272697, 30.039783, 74.51135, -10.073078, 5.0724087], dtype=torch.float32)
ACTION_STD = torch.tensor([17.493738, 42.95796, 40.683086, 9.202332, 7.6247225, 10.140567], dtype=torch.float32)


def load_w1_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = hydra.utils.instantiate(cfg.model)
    model.fuse = nn.Linear(cfg.embed_dim, cfg.embed_dim)
    if cfg.loss.idm.weight > 0:
        h = cfg.loss.idm.hidden_dim
        model.idm = nn.Sequential(
            nn.Linear(2 * cfg.embed_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, cfg.data.action_dim)
        )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), cfg


def load_w2_encoder(device: str):
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    return model.to(device).eval()


def get_cam_target(dataset: str, ep: int, cam_key: str, src_map: dict, ep_map: dict):
    """Resolves directory root and internal episode index for a given camera stream."""
    if cam_key == "observation.images.front_top_camera":
        if dataset == "orin":
            return ORIN_MERGED, ep
        else:
            return M1_CACHE / dataset, ep
    else:  # wrist_camera
        if dataset == "orin":
            sess = src_map[ep]
            src_ep = ep_map[ep]
            sess_dir = WM_ROLLOUTS / sess
            if (sess_dir / "videos" / cam_key).exists():
                return sess_dir, src_ep
            return None, None
        else:
            sess_dir = M1_CACHE / dataset
            if (sess_dir / "videos" / cam_key).exists():
                return sess_dir, ep
            return None, None


def decode_frames_pyav(root: Path, ep: int, cam_key: str, frames_needed: list[int]):
    info = json.load(open(root / "meta" / "info.json"))
    meta_df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))]).set_index("episode_index")
    m = meta_df.loc[ep]
    vpath = str(root / info["video_path"].format(
        video_key=cam_key, chunk_index=int(m[f"videos/{cam_key}/chunk_index"]), file_index=int(m[f"videos/{cam_key}/file_index"])
    ))
    t0 = float(m[f"videos/{cam_key}/from_timestamp"])
    fps = info["fps"]
    
    needed_set = set(frames_needed)
    first_f = min(frames_needed)
    last_f = max(frames_needed)
    
    got = {}
    with av.open(vpath) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        t_seek = max(0, t0 + (first_f - 2) / fps)
        c.seek(int(t_seek / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            f_idx = round(float(fr.pts * s.time_base - t0) * fps)
            if f_idx in needed_set:
                got[f_idx] = fr.to_ndarray(format="rgb24")
            if f_idx >= last_f and len(got) == len(needed_set):
                break
    
    missing = needed_set - set(got.keys())
    if missing:
        avail = sorted(got.keys())
        if avail:
            for mf in missing:
                nearest = min(avail, key=lambda x: abs(x - mf))
                got[mf] = got[nearest]
        else:
            raise RuntimeError(f"Could not decode any frames from {vpath}")
    return got


def load_actions(root: Path, ep: int, start_f: int, end_f: int):
    files = sorted(glob.glob(f"{root}/data/*/*.parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", "action"])
        dfs.append(df[df["episode_index"] == ep])
    df_ep = pd.concat(dfs).sort_values("frame_index")
    actions = np.stack(df_ep[(df_ep["frame_index"] >= start_f) & (df_ep["frame_index"] < end_f)]["action"].to_numpy())
    return actions


@torch.no_grad()
def encode_w1_images(model, top_tensors, wrist_tensors):
    """Encodes paired (N, 3, 224, 224) top and wrist tensors into W1 fused latents (N, 192)."""
    top_n = (top_tensors - IMAGENET_MEAN.to(top_tensors.device)) / IMAGENET_STD.to(top_tensors.device)
    wrist_n = (wrist_tensors - IMAGENET_MEAN.to(wrist_tensors.device)) / IMAGENET_STD.to(wrist_tensors.device)
    x = torch.stack([top_n, wrist_n], dim=1)  # (N, 2, 3, 224, 224)
    n = x.shape[0]
    cls = model.encoder(x.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state[:, 0]
    proj = model.projector(cls)
    mean_proj = proj.view(n, 2, -1).mean(dim=1)
    z = model.fuse(mean_proj)
    return z


@torch.no_grad()
def encode_w2_patches(model, img_tensors):
    """Encodes (N, 3, 224, 224) images into 8x8 pooled, layer-normed patch tokens (N, 64, 384)."""
    img_n = (img_tensors - IMAGENET_MEAN.to(img_tensors.device)) / IMAGENET_STD.to(img_tensors.device)
    h = model(pixel_values=img_n).last_hidden_state[:, 1:]  # (N, 256, 384)
    n, _, c = h.shape
    h = h.view(n, 8, 2, 8, 2, c).mean(dim=(2, 4)).reshape(n, 64, c)
    h = F.layer_norm(h, [c])
    return h


def preprocess_cv2_frame(frame_rgb):
    t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
    return t


def fit_loso(X, y, sessions):
    """Leave-one-session-out cross validation with L2 Logistic Regression."""
    y = np.asarray(y, dtype=int)
    sessions = np.asarray(sessions)
    unique_sess = np.unique(sessions)
    preds = np.full(len(y), np.nan)
    
    for s in unique_sess:
        test_mask = (sessions == s)
        train_mask = ~test_mask
        
        y_train = y[train_mask]
        if len(np.unique(y_train)) < 2:
            continue
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X[train_mask])
        X_test_sc = scaler.transform(X[test_mask])
        
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=0)
        clf.fit(X_train_sc, y_train)
        preds[test_mask] = clf.predict_proba(X_test_sc)[:, 1]
        
    return preds


def bootstrap_auroc(y_true, y_pred, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point_auc = roc_auc_score(y_true, y_pred)
    
    boot_aucs = []
    indices = rng.integers(0, n, size=(n_boot, n))
    for idx in indices:
        y_t_b = y_true[idx]
        if len(np.unique(y_t_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_t_b, y_pred[idx]))
        
    ci_low = float(np.percentile(boot_aucs, 2.5))
    ci_high = float(np.percentile(boot_aucs, 97.5))
    return float(point_auc), [ci_low, ci_high]


def bootstrap_paired_diff(y_true, y_pred_a, y_pred_b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point_diff = roc_auc_score(y_true, y_pred_a) - roc_auc_score(y_true, y_pred_b)
    
    diffs = []
    indices = rng.integers(0, n, size=(n_boot, n))
    for idx in indices:
        y_t_b = y_true[idx]
        if len(np.unique(y_t_b)) < 2:
            continue
        auc_a = roc_auc_score(y_t_b, y_pred_a[idx])
        auc_b = roc_auc_score(y_t_b, y_pred_b[idx])
        diffs.append(auc_a - auc_b)
        
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    return float(point_diff), [ci_low, ci_high]


def main():
    print("=" * 80)
    print("G10 EVALUATION: WORLD MODEL ANTICIPATION OF THE JUDGE")
    print("=" * 80)
    
    # 1. Load episode catalog and filter invalid
    scores_df = pd.read_csv(G9C3_SCORES_PATH)
    valid_df = scores_df[~scores_df["invalid"]].copy().reset_index(drop=True)
    print(f"Loaded {len(scores_df)} episodes, {len(valid_df)} valid after excluding invalid.")
    
    # Map Orin episodes to original session name
    orin_map = pd.read_csv(EPISODE_MAP_PATH)
    src_map = dict(zip(orin_map["merged_ep"], orin_map["source"]))
    ep_map = dict(zip(orin_map["merged_ep"], orin_map["src_ep"]))
    
    def get_session(row):
        if row["dataset"] == "orin":
            return src_map[row["episode"]]
        return row["dataset"]

    valid_df["session"] = valid_df.apply(get_session, axis=1)
    
    # Policy family for confounding check
    def get_family(s):
        if "turbovla" in s:
            return "turbovla"
        elif "smolvla_vlacpp" in s:
            return "smolvla_vlacpp"
        elif "smolvla" in s:
            return "smolvla"
        elif "act" in s:
            return "act"
        elif "vla_jepa" in s:
            return "vla_jepa"
        return "other"

    valid_df["family"] = valid_df["session"].apply(get_family)
    
    # Targets: Judge verdict (primary per protocol) and human ground truth
    valid_df["y_judge"] = (valid_df["s"] >= 0.5).astype(int)
    valid_df["y_gt"] = (valid_df["outcome"] == "success").astype(int)
    
    print("\nDataset Summary:")
    print(f"Total valid episodes: {len(valid_df)}")
    print(f"Judge labels: {valid_df['y_judge'].sum()} Success / {(1 - valid_df['y_judge']).sum()} Failure")
    print(f"Ground-truth labels: {valid_df['y_gt'].sum()} Success / {(1 - valid_df['y_gt']).sum()} Failure")
    print(f"Unique sessions: {valid_df['session'].nunique()}")
    print(f"Non-TurboVLA episodes: {(valid_df['family'] != 'turbovla').sum()}")

    # 2. Load models
    print("\nLoading W1 model...")
    w1_model, w1_cfg = load_w1_model(W1_CKPT_PATH, DEV)
    print("Loading W2 DINOv2-S encoder...")
    w2_encoder = load_w2_encoder(DEV)
    
    # 3. Extract features across all episodes
    print("\nExtracting features for all valid episodes...")
    t0 = time.time()
    
    features = {
        "w1_z_tj": [],             # W1 at judge frame
        "w1_z_anchor": [],         # W1 at anchor t_j - 30
        "w1_zhat_tj": [],          # W1 imagined at t_j (true actions)
        "w1_zhat_tj_corrupt": [],  # W1 imagined at t_j (corrupted actions)
        "w2_top_tj": [],           # W2 top camera patch tokens at t_j (24576d)
        "w2_both_tj": [],          # W2 top + wrist patch tokens at t_j (49152d)
        "w2_top_anchor": [],       # W2 top camera at t_j - 30
        "w2_both_anchor": [],      # W2 top + wrist at t_j - 30
        "raw_actions": [],         # 30-step executed actions (180d)
        "surprise_curves": [],     # ||z_{t_j-30+k} - \hat{z}_{t_j-30+k}|| for k=1..30
    }
    
    # Pre-collect all executed actions (32 actions: anchor - 2 to tj)
    all_actions = []
    for idx, row in valid_df.iterrows():
        ep = int(row["episode"])
        tj = int(row["judge_frame"])
        anchor = tj - 30
        action_root = ORIN_MERGED if row["dataset"] == "orin" else (M1_CACHE / row["dataset"])
        acts = load_actions(action_root, ep, anchor - 2, tj)
        assert len(acts) == 32, f"Expected 32 actions for ep {ep}, got {len(acts)}"
        all_actions.append(acts)
        
    # Seeded permutation of actions for corrupted action control
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(valid_df))
    for i in range(len(valid_df)):
        if perm[i] == i:
            perm[i] = (i + 1) % len(valid_df)
            
    # Process episodes
    for i, row in valid_df.iterrows():
        ep = int(row["episode"])
        tj = int(row["judge_frame"])
        anchor = tj - 30
        ds = row["dataset"]
        
        if (i + 1) % 10 == 0 or i == 0 or i == len(valid_df) - 1:
            print(f"Processing episode {i + 1}/{len(valid_df)} (elapsed {time.time() - t0:.1f}s)...")
            
        needed_frames = list(range(anchor - 2, tj + 1))
        
        # Resolve camera paths
        top_root, top_ep = get_cam_target(ds, ep, "observation.images.front_top_camera", src_map, ep_map)
        wrist_root, wrist_ep = get_cam_target(ds, ep, "observation.images.wrist_camera", src_map, ep_map)
        
        top_frames = decode_frames_pyav(top_root, top_ep, "observation.images.front_top_camera", needed_frames)
        if wrist_root is not None:
            wrist_frames = decode_frames_pyav(wrist_root, wrist_ep, "observation.images.wrist_camera", needed_frames)
        else:
            # Fallback for the 10 trimmed episodes lacking wrist video
            wrist_frames = top_frames
        
        # Convert to tensors (33, 3, 224, 224)
        top_t = torch.stack([preprocess_cv2_frame(top_frames[f]) for f in needed_frames]).to(DEV)
        wrist_t = torch.stack([preprocess_cv2_frame(wrist_frames[f]) for f in needed_frames]).to(DEV)
        
        # 1. W1 features
        with torch.no_grad():
            z_seq = encode_w1_images(w1_model, top_t, wrist_t)  # (33, 192)
            
            z_ctx = z_seq[:3].unsqueeze(0)  # (1, 3, 192)
            z_anchor = z_seq[2]             # (192,)
            z_tj = z_seq[-1]                # (192,)
            z_true_future = z_seq[3:]       # (30, 192) true latents along rollout
            
            # Actions (32 actions: anchor-2 to tj)
            acts_true = torch.from_numpy(all_actions[i]).float()
            acts_corrupt = torch.from_numpy(all_actions[perm[i]]).float()
            
            # Normalize actions
            a_norm_true = ((acts_true - ACTION_MEAN) / ACTION_STD).unsqueeze(0).to(DEV)
            a_norm_corrupt = ((acts_corrupt - ACTION_MEAN) / ACTION_STD).unsqueeze(0).to(DEV)
            
            # Autoregressive rollout with true actions (30 steps)
            a_emb_true = w1_model.action_encoder(a_norm_true)
            ctx_curr = z_ctx
            preds_true = []
            for k in range(30):
                nxt = w1_model.predict(ctx_curr[:, -3:], a_emb_true[:, k:k + 3])[:, -1:]
                preds_true.append(nxt)
                ctx_curr = torch.cat([ctx_curr, nxt], dim=1)
            zhat_seq_true = torch.cat(preds_true, dim=1).squeeze(0)  # (30, 192)
            zhat_tj = zhat_seq_true[-1]                                # (192,)
            
            # Autoregressive rollout with corrupted actions
            a_emb_corrupt = w1_model.action_encoder(a_norm_corrupt)
            ctx_curr_c = z_ctx
            preds_corrupt = []
            for k in range(30):
                nxt = w1_model.predict(ctx_curr_c[:, -3:], a_emb_corrupt[:, k:k + 3])[:, -1:]
                preds_corrupt.append(nxt)
                ctx_curr_c = torch.cat([ctx_curr_c, nxt], dim=1)
            zhat_tj_corrupt = torch.cat(preds_corrupt, dim=1).squeeze(0)[-1]  # (192,)
            
            # Surprise error curve: L2 norm between z_t and \hat{z}_t for k=1..30
            surprise_curve = torch.norm(z_true_future - zhat_seq_true, p=2, dim=-1).cpu().numpy()
            
            # 2. W2 features
            # At t_j (last frame):
            w2_top_tj_tok = encode_w2_patches(w2_encoder, top_t[-1:].to(DEV)).squeeze(0)  # (64, 384)
            w2_wrist_tj_tok = encode_w2_patches(w2_encoder, wrist_t[-1:].to(DEV)).squeeze(0)
            
            # At anchor (index 2):
            w2_top_anc_tok = encode_w2_patches(w2_encoder, top_t[2:3].to(DEV)).squeeze(0)
            w2_wrist_anc_tok = encode_w2_patches(w2_encoder, wrist_t[2:3].to(DEV)).squeeze(0)
            
            # Flatten
            w2_top_tj_vec = w2_top_tj_tok.flatten().cpu().numpy()
            w2_both_tj_vec = torch.cat([w2_top_tj_tok, w2_wrist_tj_tok], dim=0).flatten().cpu().numpy()
            w2_top_anc_vec = w2_top_anc_tok.flatten().cpu().numpy()
            w2_both_anc_vec = torch.cat([w2_top_anc_tok, w2_wrist_anc_tok], dim=0).flatten().cpu().numpy()
            
        # Append features
        features["w1_z_tj"].append(z_tj.cpu().numpy())
        features["w1_z_anchor"].append(z_anchor.cpu().numpy())
        features["w1_zhat_tj"].append(zhat_tj.cpu().numpy())
        features["w1_zhat_tj_corrupt"].append(zhat_tj_corrupt.cpu().numpy())
        features["w2_top_tj"].append(w2_top_tj_vec)
        features["w2_both_tj"].append(w2_both_tj_vec)
        features["w2_top_anchor"].append(w2_top_anc_vec)
        features["w2_both_anchor"].append(w2_both_anc_vec)
        features["raw_actions"].append(acts_true[2:].flatten().numpy())
        features["surprise_curves"].append(surprise_curve)
        
    print(f"\nFeature extraction complete in {time.time() - t0:.1f}s.")
    
    for k in features:
        features[k] = np.array(features[k])
        print(f"Feature '{k}' shape: {features[k].shape}")
        
    np.savez_compressed(
        RESULTS_DIR / "g10_features.npz",
        **features,
        y_judge=valid_df["y_judge"].to_numpy(),
        y_gt=valid_df["y_gt"].to_numpy(),
        sessions=valid_df["session"].to_numpy(),
        families=valid_df["family"].to_numpy(),
        outcomes=valid_df["outcome"].to_numpy(),
        failure_modes=valid_df["failure_mode"].fillna("").to_numpy(),
    )
    print(f"Saved extracted features to {RESULTS_DIR / 'g10_features.npz'}")
    
    # Policy identity baseline (one-hot policy family)
    families = sorted(valid_df["family"].unique())
    policy_onehot = np.stack([(valid_df["family"] == f).astype(float) for f in families], axis=1)
    
    y_judge = valid_df["y_judge"].to_numpy()
    y_gt = valid_df["y_gt"].to_numpy()
    sessions = valid_df["session"].to_numpy()
    non_turbo = (valid_df["family"] != "turbovla").to_numpy()

    # =========================================================================
    # STEP 1: READOUT AUROC AT JUDGE FRAME t_j
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: READOUT AUROC AT JUDGE FRAME t_j")
    print("=" * 80)
    
    step1_features = {
        "W1 (Top + Wrist fused, 192d)": features["w1_z_tj"],
        "W2 Top view (24,576d)": features["w2_top_tj"],
        "W2 Top + Wrist (49,152d)": features["w2_both_tj"],
    }
    
    step1_results = {}
    for name, X in step1_features.items():
        preds_judge = fit_loso(X, y_judge, sessions)
        auc_j, ci_j = bootstrap_auroc(y_judge, preds_judge)
        
        preds_gt = fit_loso(X, y_gt, sessions)
        auc_gt, ci_gt = bootstrap_auroc(y_gt, preds_gt)
        
        preds_nt = fit_loso(X[non_turbo], y_judge[non_turbo], sessions[non_turbo])
        auc_nt, ci_nt = bootstrap_auroc(y_judge[non_turbo], preds_nt)
        
        step1_results[name] = {
            "dim": X.shape[1],
            "auroc_judge": round(auc_j, 4),
            "ci95_judge": [round(ci_j[0], 4), round(ci_j[1], 4)],
            "auroc_gt": round(auc_gt, 4),
            "ci95_gt": [round(ci_gt[0], 4), round(ci_gt[1], 4)],
            "auroc_judge_non_turbo": round(auc_nt, 4),
            "ci95_judge_non_turbo": [round(ci_nt[0], 4), round(ci_nt[1], 4)],
            "pass_step1": auc_j >= 0.90,
        }
        print(f"{name:32s} | AUROC (Judge): {auc_j:.4f} [{ci_j[0]:.3f}, {ci_j[1]:.3f}] | GT: {auc_gt:.4f} | Non-Turbo: {auc_nt:.4f} | Pass >=0.90: {auc_j >= 0.90}")

    # =========================================================================
    # STEP 2: ANTICIPATION AUROC AT ANCHOR t_j - 30 (1.0s lookahead)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: ANTICIPATION AUROC AT ANCHOR t_j - 30 (1.0s Lookahead)")
    print("=" * 80)
    
    step2_feature_sets = {
        "(1) Current latent z_{t_j-30} (W1)": features["w1_z_anchor"],
        "(2) Raw executed actions a_{t..t+30}": features["raw_actions"],
        "(3) Imagined latent zhat_{t_j} (True Actions)": features["w1_zhat_tj"],
        "(4) Imagined latent zhat_{t_j} (Corrupted Actions)": features["w1_zhat_tj_corrupt"],
        "(5) Policy identity baseline": policy_onehot,
    }
    
    step2_preds = {}
    step2_results = {}
    for name, X in step2_feature_sets.items():
        preds = fit_loso(X, y_judge, sessions)
        auc, ci = bootstrap_auroc(y_judge, preds)
        
        preds_gt = fit_loso(X, y_gt, sessions)
        auc_gt, ci_gt = bootstrap_auroc(y_gt, preds_gt)
        
        step2_preds[name] = preds
        step2_results[name] = {
            "dim": X.shape[1],
            "auroc_judge": round(auc, 4),
            "ci95_judge": [round(ci[0], 4), round(ci[1], 4)],
            "auroc_gt": round(auc_gt, 4),
            "ci95_gt": [round(ci_gt[0], 4), round(ci_gt[1], 4)],
        }
        print(f"{name:50s} | AUROC (Judge): {auc:.4f} [{ci[0]:.3f}, {ci[1]:.3f}] | GT: {auc_gt:.4f}")
        
    # Paired differences for Step 2
    p3 = step2_preds["(3) Imagined latent zhat_{t_j} (True Actions)"]
    p1 = step2_preds["(1) Current latent z_{t_j-30} (W1)"]
    p4 = step2_preds["(4) Imagined latent zhat_{t_j} (Corrupted Actions)"]
    p5 = step2_preds["(5) Policy identity baseline"]
    
    diff_3_vs_1, ci_3_vs_1 = bootstrap_paired_diff(y_judge, p3, p1)
    diff_3_vs_4, ci_3_vs_4 = bootstrap_paired_diff(y_judge, p3, p4)
    diff_3_vs_5, ci_3_vs_5 = bootstrap_paired_diff(y_judge, p3, p5)
    
    paired_diffs = {
        "(3) Imagined - (1) Current z": {"diff": round(diff_3_vs_1, 4), "ci95": [round(ci_3_vs_1[0], 4), round(ci_3_vs_1[1], 4)]},
        "(3) Imagined - (4) Corrupted Actions": {"diff": round(diff_3_vs_4, 4), "ci95": [round(ci_3_vs_4[0], 4), round(ci_3_vs_4[1], 4)]},
        "(3) Imagined - (5) Policy Identity": {"diff": round(diff_3_vs_5, 4), "ci95": [round(ci_3_vs_5[0], 4), round(ci_3_vs_5[1], 4)]},
    }
    
    print("\nPaired Differences (True Imagined vs Baselines):")
    for k, v in paired_diffs.items():
        print(f"  {k:45s}: {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
        
    pass_step2 = (
        step2_results["(3) Imagined latent zhat_{t_j} (True Actions)"]["auroc_judge"] >= 0.80
        and diff_3_vs_1 >= 0.05 and ci_3_vs_1[0] > 0
        and diff_3_vs_4 > 0
        and diff_3_vs_5 > 0
    )
    print(f"\nStep 2 Overall Verdict: {'PASS' if pass_step2 else 'FAIL'}")

    # =========================================================================
    # MID-CHUNK SURPRISE ERROR CURVES
    # =========================================================================
    print("\n" + "=" * 80)
    print("MID-CHUNK SURPRISE ERROR CURVES (||z_t - \\hat{z}_t|| along k=1..30)")
    print("=" * 80)
    
    surprise_curves = features["surprise_curves"]  # (87, 30)
    outcomes = valid_df["outcome"].to_numpy()
    failure_modes = valid_df["failure_mode"].fillna("unspecified").to_numpy()
    
    groups = {
        "success": outcomes == "success",
        "failure_all": outcomes == "failure",
        "missed_grasp": (outcomes == "failure") & (failure_modes == "missed_grasp"),
        "stalled": (outcomes == "failure") & (failure_modes == "stalled"),
        "wrong_object": (outcomes == "failure") & (failure_modes == "wrong_object"),
    }
    
    surprise_stats = {}
    print(f"{'Group':20s} | {'Count':5s} | {'k=1 MSE':10s} | {'k=10 MSE':10s} | {'k=20 MSE':10s} | {'k=30 MSE':10s} | {'Mean Error':10s}")
    print("-" * 80)
    for gname, mask in groups.items():
        if mask.sum() == 0:
            continue
        crvs = surprise_curves[mask]
        mean_crv = crvs.mean(axis=0)
        std_crv = crvs.std(axis=0)
        surprise_stats[gname] = {
            "n": int(mask.sum()),
            "k1": round(float(mean_crv[0]), 4),
            "k10": round(float(mean_crv[9]), 4),
            "k20": round(float(mean_crv[19]), 4),
            "k30": round(float(mean_crv[29]), 4),
            "mean_overall": round(float(crvs.mean()), 4),
            "curve_mean": [round(float(x), 4) for x in mean_crv],
            "curve_std": [round(float(x), 4) for x in std_crv],
        }
        print(f"{gname:20s} | {mask.sum():5d} | {mean_crv[0]:10.4f} | {mean_crv[9]:10.4f} | {mean_crv[19]:10.4f} | {mean_crv[29]:10.4f} | {crvs.mean():10.4f}")

    # Compile final JSON report
    report = {
        "date": "2026-10-17",
        "protocol": "G10 (G10_PROTOCOL.md)",
        "test_set": {
            "total_episodes": len(scores_df),
            "valid_episodes": len(valid_df),
            "judge_success": int(valid_df["y_judge"].sum()),
            "judge_failure": int((1 - valid_df["y_judge"]).sum()),
            "gt_success": int(valid_df["y_gt"].sum()),
            "gt_failure": int((1 - valid_df["y_gt"]).sum()),
            "unique_sessions": int(valid_df["session"].nunique()),
        },
        "step1_readout": step1_results,
        "step2_anticipation": {
            "features": step2_results,
            "paired_differences": paired_diffs,
            "pass_step2": pass_step2,
        },
        "surprise_error": surprise_stats,
    }
    
    with open(RESULTS_DIR / "g10_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote full results to {RESULTS_DIR / 'g10_results.json'}")


if __name__ == "__main__":
    main()
