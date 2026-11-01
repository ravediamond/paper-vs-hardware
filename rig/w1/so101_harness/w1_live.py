"""W1 surprise reflex, live: the G10 procedure applied on the robot.

G10 (`projects/vla-jepa-world-model/wm_train/g10_eval.py`) computed surprise offline as follows, and the reflex
thresholds (2.2 at k <= 5, 10.0 at k >= 15) were read from those curves:

- frames: RGB uint8 -> float [0, 1] -> bilinear resize to 224x224 -> ImageNet normalization; top and wrist views go
  through the shared ViT, CLS -> projector, the two views are averaged, then `fuse`: one 192-d latent per frame;
- context: the latents of the anchor frame and the 2 frames before it;
- actions: the executed actions of those 2 earlier frames, then the actions from the anchor on, normalized with the
  training statistics and embedded as one sequence by `action_encoder`;
- rollout: z_hat(k+1) = predict(ctx[-3:], a_emb[k:k+3])[-1], appended to the context, for k = 0, 1, ...;
- surprise at step k: || z(anchor + k) - z_hat(k) ||, k = 1..30.

Live, the anchor is the frame observed when a chunk starts executing (sync: each chunk; async: each chunk swap), the
earlier frames and actions come from `record_tick`, which the runner calls every tick, and step k is the check after
the k-th action of that chunk was sent. The live frame at step k is captured right after the k-th send, at most one
tick earlier than G10's frame (anchor + k): the G11-B normal trials check that the thresholds still give no false
alarms.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import SurpriseConfig
from .system1_runner import LeWMSurpriseMonitor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# W1's action normalization: training-set statistics, as in g10_eval.py (degrees; gripper 0-100).
W1_ACTION_MEAN = (3.823351, -32.272697, 30.039783, 74.51135, -10.073078, 5.0724087)
W1_ACTION_STD = (17.493738, 42.95796, 40.683086, 9.202332, 7.6247225, 10.140567)
CONTEXT = 3


def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> (3, 224, 224) float in [0, 1], bilinear, no padding (g10_eval.preprocess_cv2_frame)."""
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    return F.interpolate(t.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)


_KEY_REMAP = (  # transformers ViT key names across versions (as in jetson/run_g11_phase_b_live.py)
    ("encoder.layers.", "encoder.layer."),
    (".attention.q_proj.", ".attention.attention.query."),
    (".attention.k_proj.", ".attention.attention.key."),
    (".attention.v_proj.", ".attention.attention.value."),
    (".attention.o_proj.", ".attention.output.dense."),
    (".mlp.fc1.", ".intermediate.dense."),
    (".mlp.fc2.", ".output.dense."),
)


def load_state_dict_strict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Strict load, retrying once with the ViT key remap. Never `strict=False`: a silently dropped key would leave
    random weights in the encoder and the reflex would look plausible and be wrong."""
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as first:
        remapped = {}
        for k, v in state_dict.items():
            for old, new in _KEY_REMAP:
                if old in k:
                    k = k.replace(old, new)
            remapped[k] = v
        try:
            model.load_state_dict(remapped, strict=True)
        except RuntimeError as second:
            raise RuntimeError(
                "W1 state dict does not match the model, with or without remap. W1 was saved with transformers "
                "5.19.0; older versions (e.g. the image's 5.5.4) nest the ViT layers as encoder.encoder.layer. Run with "
                "W1=1 shared/gate/jetson_run.sh (adds transformers 5.19.0), see "
                f"shared/gate/data/2026-10-18_w1_noise_floor/README.md.\n{second}"
            ) from first


class W1LiveMonitor(LeWMSurpriseMonitor):
    """Drop-in for `LeWMSurpriseMonitor` in `System1Runner`, computing surprise as G10 did."""

    wants_history = True

    def __init__(self, model: nn.Module, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
                 action_mean=W1_ACTION_MEAN, action_std=W1_ACTION_STD) -> None:
        # The base class only sets device and buffers here; the model is ours.
        super().__init__(allow_mock=True, device=str(device))
        for name in ("encoder", "projector", "fuse", "action_encoder", "predict"):
            if not hasattr(model, name):
                raise TypeError(f"W1 model has no `{name}`")
        self.model = model.to(device=self.device, dtype=dtype).eval()
        self.allow_mock = False
        self.dtype = dtype
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self.a_mean = torch.tensor(action_mean, device=self.device, dtype=torch.float32)
        self.a_std = torch.tensor(action_std, device=self.device, dtype=torch.float32)
        self._history: deque[tuple[torch.Tensor, torch.Tensor, np.ndarray]] = deque(maxlen=CONTEXT - 1)
        self._ctx: torch.Tensor | None = None
        self._ctx_actions: np.ndarray | None = None
        self.last_curve: dict[int, float] = {}
        self.timings_ms: list[float] = []

    # ---- loading ------------------------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None) -> "W1LiveMonitor":
        """Builds W1 from `weights_final.pt` (config + state_dict), as g10_eval.load_w1_model does. Needs hydra,
        omegaconf and stable-worldmodel @ 21446f1 (the `lerobot-jetson-wm` image has them). FP16 on CUDA, as in
        G11-A; FP32 elsewhere."""
        import hydra
        from omegaconf import OmegaConf

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(ckpt["config"])
        model = hydra.utils.instantiate(cfg.model)
        model.fuse = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        if cfg.loss.idm.weight > 0:
            h = cfg.loss.idm.hidden_dim
            model.idm = nn.Sequential(nn.Linear(2 * cfg.embed_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                      nn.Linear(h, cfg.data.action_dim))
        load_state_dict_strict(model, ckpt["state_dict"])
        dtype = torch.float16 if str(dev).startswith("cuda") else torch.float32
        return cls(model, device=dev, dtype=dtype)

    # ---- per tick -----------------------------------------------------------------------------
    def record_tick(self, top_rgb: np.ndarray, wrist_rgb: np.ndarray, action: np.ndarray) -> None:
        """The frame observed at the start of a tick and the action sent on it."""
        self._history.append((preprocess(top_rgb), preprocess(wrist_rgb), np.asarray(action, dtype=np.float32)))

    # ---- encoding -----------------------------------------------------------------------------
    @torch.inference_mode()
    def _encode(self, tops: list[torch.Tensor], wrists: list[torch.Tensor]) -> torch.Tensor:
        """(N, 192) latents for N frames (g10_eval.encode_w1_images)."""
        top = torch.stack(tops).to(self.device, self.dtype)
        wrist = torch.stack(wrists).to(self.device, self.dtype)
        x = torch.stack([(top - self.mean) / self.std, (wrist - self.mean) / self.std], dim=1)  # (N, 2, 3, H, W)
        n = x.shape[0]
        cls = self.model.encoder(x.flatten(0, 1), interpolate_pos_encoding=True).last_hidden_state[:, 0]
        proj = self.model.projector(cls)
        return self.model.fuse(proj.view(n, 2, -1).mean(dim=1))

    def encode(self, top_rgb: np.ndarray, wrist_rgb: np.ndarray) -> torch.Tensor:  # type: ignore[override]
        """Anchor encoding: the anchor frame with the 2 recorded frames before it, in one batch."""
        if len(self._history) < CONTEXT - 1:
            raise RuntimeError("W1 needs 2 recorded ticks before the first anchor (System1Runner primes them)")
        hist = list(self._history)
        z = self._encode([h[0] for h in hist] + [preprocess(top_rgb)], [h[1] for h in hist] + [preprocess(wrist_rgb)])
        self._ctx = z.unsqueeze(0)  # (1, 3, 192)
        self._ctx_actions = np.stack([h[2] for h in hist])  # actions of the 2 earlier frames
        return z[-1]

    @torch.inference_mode()
    def imagine_chunk(self, z_0, action_chunk: np.ndarray, check_steps=(3, 10, 15, 20, 25)) -> None:  # type: ignore[override]
        self._expected_latents.clear()
        self.last_curve = {}
        if self._ctx is None or self._ctx_actions is None:
            raise RuntimeError("call encode() on the anchor frame first")
        n_steps = min(len(action_chunk), max(check_steps, default=0))
        acts = np.concatenate([self._ctx_actions, np.asarray(action_chunk, dtype=np.float32)[:n_steps]], axis=0)
        a = (torch.from_numpy(acts).to(self.device) - self.a_mean) / self.a_std
        a_emb = self.model.action_encoder(a.unsqueeze(0).to(self.dtype))
        ctx = self._ctx
        for k in range(n_steps):
            nxt = self.model.predict(ctx[:, -CONTEXT:], a_emb[:, k:k + CONTEXT])[:, -1:]
            ctx = torch.cat([ctx, nxt], dim=1)
            if k + 1 in check_steps:
                self._expected_latents[k + 1] = nxt[0, 0].clone()

    def check_step_surprise(self, step_k: int, top_rgb: np.ndarray, wrist_rgb: np.ndarray,
                            cfg: SurpriseConfig) -> tuple[bool, float, str, float]:  # type: ignore[override]
        if step_k not in self._expected_latents:
            return False, 0.0, "unchecked_step", 0.0
        t0 = time.monotonic()
        z_real = self._encode([preprocess(top_rgb)], [preprocess(wrist_rgb)])[0]
        err = float(torch.linalg.norm(z_real.float() - self._expected_latents[step_k].float()).item())
        ms = (time.monotonic() - t0) * 1000.0
        self.timings_ms.append(ms)
        self.last_curve[step_k] = round(err, 3)
        if step_k <= cfg.stall_step_max and err > cfg.stall_threshold_l2:
            return True, err, f"w1_stall_reflex_step_{step_k} (err={err:.2f} > {cfg.stall_threshold_l2})", ms
        if step_k >= cfg.slip_step_min and err > cfg.slip_threshold_l2:
            return True, err, f"w1_missed_grasp_reflex_step_{step_k} (err={err:.2f} > {cfg.slip_threshold_l2})", ms
        return False, err, "ok", ms

    def reset(self) -> None:
        self._history.clear()
        self._ctx = None
        self._ctx_actions = None
        self._expected_latents.clear()
