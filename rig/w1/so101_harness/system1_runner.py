"""LeWMSurpriseMonitor, copied verbatim from road-to-robotics src/so101_harness/system1_runner.py at cc83df4 (lines 44-168);
the rest of that module (the live runner, the recorder, the safety guard) is not part of the W1 environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .contracts import SurpriseConfig


class LeWMSurpriseMonitor:
    """System 1 World Model: Unrolls W1 (192-d) and monitors online prediction surprise."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str | None = None,
        allow_mock: bool = False,
    ) -> None:
        """`allow_mock=True` (tests and dry runs only) lets the monitor run without a checkpoint; its
        errors are then always 0, so the reflex can never fire. Otherwise a missing model raises."""
        self.allow_mock = allow_mock
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        else:
            self.device = torch.device(device)

        self.model: Any = None
        self._expected_latents: dict[int, torch.Tensor] = {}

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            self._load_checkpoint(Path(checkpoint_path))
        if self.model is None and not allow_mock:
            raise RuntimeError(
                f"W1 checkpoint not loaded ({checkpoint_path}); the surprise reflex would silently never fire. "
                "Pass allow_mock=True only for tests."
            )

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        try:
            raw = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if hasattr(raw, "eval"):
                self.model = raw
            elif isinstance(raw, dict) and "model" in raw and hasattr(raw["model"], "eval"):
                self.model = raw["model"]
            else:
                self.model = raw
            if hasattr(self.model, "eval"):
                self.model.eval()
                if self.device.type == "cuda":
                    self.model.half()
        except Exception as exc:
            print(f"[LeWMSurpriseMonitor] Warning: could not initialize model from {ckpt_path}: {exc}")
            self.model = None

    @torch.inference_mode()
    def encode(self, top_rgb: np.ndarray, wrist_rgb: np.ndarray) -> torch.Tensor:
        """Encodes dual 224x224 camera views into 192-d latent representation."""
        if self.model is None or not hasattr(self.model, "encode_multiview"):
            if not self.allow_mock:
                raise RuntimeError("W1 model has no encode_multiview")
            return torch.zeros(192, device=self.device, dtype=torch.float32)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        top_t = torch.from_numpy(top_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=dtype) / 255.0
        wrist_t = torch.from_numpy(wrist_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=dtype) / 255.0
        return self.model.encode_multiview(top_t, wrist_t).squeeze(0)

    @torch.inference_mode()
    def imagine_chunk(
        self,
        z_0: torch.Tensor,
        action_chunk: np.ndarray,
        check_steps: tuple[int, ...] = (3, 10, 15, 20, 25),
    ) -> None:
        """Rolls out AdaLN predictor forward along proposed action trajectory (~2.3 ms on Orin)."""
        self._expected_latents.clear()
        if self.model is None or not hasattr(self.model, "predict_next"):
            if not self.allow_mock:
                raise RuntimeError("W1 model has no predict_next")
            for s in check_steps:
                self._expected_latents[s] = z_0.clone()
            return

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        z_curr = z_0.unsqueeze(0).to(dtype=dtype)
        actions_t = torch.from_numpy(action_chunk).to(self.device, dtype=dtype)
        max_step = min(len(action_chunk), max(check_steps, default=0))

        for k in range(1, max_step + 1):
            act_k = actions_t[k - 1 : k]
            z_curr = self.model.predict_next(z_curr, act_k)
            if k in check_steps:
                self._expected_latents[k] = z_curr.squeeze(0).clone()

    # A monitor that needs the frames and actions before the anchor (W1LiveMonitor) sets this and implements
    # record_tick; the runner then captures the cameras every tick and records what it sent.
    wants_history = False

    def record_tick(self, top_rgb: np.ndarray, wrist_rgb: np.ndarray, action: np.ndarray) -> None:
        return None

    def reset(self) -> None:
        self._expected_latents.clear()

    @torch.inference_mode()
    def check_step_surprise(
        self,
        step_k: int,
        top_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        cfg: SurpriseConfig,
    ) -> tuple[bool, float, str, float]:
        """Evaluates prediction error ||z_k - z_hat_k||.

        Returns (should_abort, l2_error, reason, inference_ms).
        """
        if step_k not in self._expected_latents:
            return False, 0.0, "unchecked_step", 0.0

        t0 = time.monotonic()
        z_real = self.encode(top_rgb, wrist_rgb)
        z_hat = self._expected_latents[step_k]
        l2_err = float(torch.linalg.norm(z_real.float() - z_hat.float()).item())
        inf_ms = (time.monotonic() - t0) * 1000.0

        if step_k <= cfg.stall_step_max and l2_err > cfg.stall_threshold_l2:
            return True, l2_err, f"w1_stall_reflex_step_{step_k} (err={l2_err:.2f} > {cfg.stall_threshold_l2})", inf_ms

        if step_k >= cfg.slip_step_min and l2_err > cfg.slip_threshold_l2:
            return True, l2_err, f"w1_missed_grasp_reflex_step_{step_k} (err={l2_err:.2f} > {cfg.slip_threshold_l2})", inf_ms

        return False, l2_err, "ok", inf_ms


