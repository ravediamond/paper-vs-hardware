"""Shared helpers for the best-of-N reranking pre-checks (VLA-JEPA on so101_bowls_all_trimmed).

Runs inside lerobot-jetson:rc with the image's own lerobot (same vla_jepa code as the training commit 8c920c427):

    IMAGE=lerobot-jetson:rc LEROBOT_SRC=image shared/gate/jetson_run.sh --python projects/vla-jepa-world-model/prechecks/<script>.py

Units: "normalized" = the policy's action space, MIN_MAX to [-1, 1] with the checkpoint's dataset stats, clipped to
[-1, 1] as the deployed postprocessor does; "raw" = joint degrees, gripper in the follower's 0-100 range.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]  # road-to-robotics/
sys.path.insert(0, str(REPO_ROOT / "shared" / "gate"))  # monitor_rollout

POLICY_REPO = "ravediamond/vla_jepa_so101_bowls_trimmed"
DATASET_REPO = "ravediamond/so101_bowls_all_trimmed"
FPS = 30
CHUNK = 30  # chunk_size = n_action_steps of the checkpoint
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER = 5
ARM = slice(0, 5)
OUT = Path(__file__).resolve().parent / "out"
SEED = 1234


def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def policy_path() -> Path:
    import monitor_rollout  # scripts/gate: root-only snapshot (never the 9 intermediate checkpoints)

    return monitor_rollout.resolve_policy_path(POLICY_REPO, None)


def load_policy(device: str = "cuda"):
    """Policy + pre/post processors exactly as deployed (world model off)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

    path = policy_path()
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.enable_world_model = False
    cfg.device = device
    policy = VLAJEPAPolicy.from_pretrained(path, config=cfg).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=path)
    return policy, pre, post, cfg, path


def action_minmax(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from safetensors.torch import load_file

    stats = load_file(str(path / "policy_postprocessor_step_2_unnormalizer_processor.safetensors"))
    return stats["action.min"].numpy(), stats["action.max"].numpy()


def to_norm(raw: np.ndarray, amin: np.ndarray, amax: np.ndarray) -> np.ndarray:
    return 2.0 * (raw - amin) / (amax - amin) - 1.0


def to_raw(norm: np.ndarray, amin: np.ndarray, amax: np.ndarray) -> np.ndarray:
    return (np.clip(norm, -1.0, 1.0) + 1.0) / 2.0 * (amax - amin) + amin


def norm_to_deg_scale(amin: np.ndarray, amax: np.ndarray) -> np.ndarray:
    """Multiply a normalized-unit distance/std per joint by this to get raw units."""
    return (amax - amin) / 2.0


def dataset_root() -> Path:
    hits = glob.glob(
        str(Path.home() / ".cache/huggingface/lerobot/hub/datasets--ravediamond--so101_bowls_all_trimmed/snapshots/*")
    )
    return Path(sorted(hits)[-1])


def load_tables():
    """Per-frame actions/state/indices from the parquet files (no video decoding)."""
    import pandas as pd

    root = dataset_root()
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)))
    df = df.sort_values("index").reset_index(drop=True)
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    return df, tasks


def phase_windows(gripper: np.ndarray, thr: float = 5.0, half: int = 15) -> dict[str, tuple[int, int]] | None:
    """Task phases from the gripper trace of one episode (0 = closed, ~15-35 = open).

    Trimmed episodes start closed; the gripper opens on the way to the bowl, closes on its rim (grasp), stays
    closed while carrying it, opens to release it, then closes for the return. With o1 = first open frame,
    c1 = first close after o1 (grasp), o2 = first open after c1 (release):
      start     [0, min(o1, 15))         first 0.5 s, before the gripper opens
      approach  [o1, c1 - 15)            gripper open, moving to the bowl
      grasp     [c1 - 15, c1 + 15)       +-0.5 s around closing on the bowl
      transport [c1 + 15, o2 - 15)       holding the bowl
      release   [o2 - 15, o2 + 15)       +-0.5 s around opening over the target
    Returns None when the trace does not show this open / close / open pattern.
    """
    is_open = gripper > thr
    opens = np.flatnonzero(is_open[1:] & ~is_open[:-1]) + 1
    closes = np.flatnonzero(~is_open[1:] & is_open[:-1]) + 1
    if is_open[0] or not len(opens):
        return None
    o1 = opens[0]
    c1s = closes[closes > o1]
    if not len(c1s):
        return None
    c1 = c1s[0]
    o2s = opens[opens > c1]
    if not len(o2s):
        return None
    o2 = o2s[0]
    windows = {
        "start": (0, min(o1, half)),
        "approach": (o1, c1 - half),
        "grasp": (c1 - half, c1 + half),
        "transport": (c1 + half, o2 - half),
        "release": (o2 - half, o2 + half),
    }
    return {k: v for k, v in windows.items() if v[1] > v[0]}


@torch.no_grad()
def encode(policy, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Backbone once: Qwen3-VL conditioning tokens + normalized state (what the head consumes)."""
    inputs = policy._prepare_model_inputs(batch, training=False)
    tokens, _ = policy.model._encode_qwen(inputs["images"], inputs["instructions"], need_action_tokens=False)
    state = inputs.get("state")
    return tokens.float(), (state.float() if state is not None else None)


@torch.no_grad()
def sample_head(
    policy,
    tokens: torch.Tensor,
    state: torch.Tensor | None,
    noise: torch.Tensor,
    steps: int | None = None,
) -> torch.Tensor:
    """VLAJEPAActionHead.predict_action with explicit initial noise [N, H, D] and step count, batched over N.

    Same Euler loop as the original (action_head.py, predict_action); with steps=None and noise drawn as there,
    the output is identical (checked by check_head_equivalence).
    """
    head = policy.model.action_model
    n = noise.shape[0]
    cond = tokens.expand(n, *tokens.shape[1:]) if tokens.shape[0] == 1 else tokens
    st = None if state is None else (state.expand(n, *state.shape[1:]) if state.shape[0] == 1 else state)
    steps = steps or head.num_inference_timesteps
    actions = noise.to(device=cond.device, dtype=cond.dtype)
    dt = 1.0 / max(steps, 1)
    for step in range(steps):
        t_value = int(step / float(max(steps, 1)) * head.config.action_num_timestep_buckets)
        timesteps = torch.full((n,), t_value, device=cond.device, dtype=torch.long)
        hidden = head._build_inputs(actions, st, timesteps)
        pred = head.model(hidden_states=hidden, encoder_hidden_states=cond, timestep=timesteps)
        actions = actions + dt * head.action_decoder(pred[:, -head.action_horizon :])
    return actions


def draw_noise(policy, n: int, seed: int, scale: float = 1.0, device: str = "cuda") -> torch.Tensor:
    head = policy.model.action_model
    gen = torch.Generator(device=device).manual_seed(seed)
    return scale * torch.randn(n, head.action_horizon, head.config.action_dim, generator=gen, device=device)


@torch.no_grad()
def check_head_equivalence(policy, tokens: torch.Tensor, state: torch.Tensor | None) -> float:
    """Max |difference| between sample_head and the original predict_action given the same noise."""
    head = policy.model.action_model
    torch.manual_seed(0)
    ref = head.predict_action(tokens, state)
    torch.manual_seed(0)
    noise = torch.randn(1, head.action_horizon, head.config.action_dim, dtype=tokens.dtype, device=tokens.device)
    return float((sample_head(policy, tokens, state, noise) - ref).abs().max())


def fix_owner(*paths: Path) -> None:
    """The container runs as root: hand files written under the repo back to the repo's owner."""
    import os

    uid, gid = REPO_ROOT.stat().st_uid, REPO_ROOT.stat().st_gid
    for p in paths:
        for q in [p, *(p.rglob("*") if p.is_dir() else [])]:
            os.chown(q, uid, gid)


def chunk_dist(a: np.ndarray, b: np.ndarray, dims=slice(None)) -> np.ndarray:
    """Distance between action chunks [..., H, D]: mean over the H steps of the L2 norm over the selected dims."""
    return np.linalg.norm(a[..., dims] - b[..., dims], axis=-1).mean(axis=-1)
