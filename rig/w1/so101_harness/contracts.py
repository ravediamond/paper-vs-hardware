"""Data contracts and schemas for the SO-101 UMA System 0-3 harness.

Adapted from NVIDIA ASPIRE cap-harness artifacts.py and UMA latent stack v0.9.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Literal


@dataclass
class RunTiming:
    """Disjoint wall-clock timing attribution (borrowed from ASPIRE artifacts.py)."""

    buckets: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    _attributed: float = 0.0

    @contextlib.contextmanager
    def measure(self, bucket: str):
        started, attr_start = time.monotonic(), self._attributed
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            nested = self._attributed - attr_start
            net = max(elapsed - nested, 0.0)
            self.buckets[bucket] = self.buckets.get(bucket, 0.0) + net
            self.counts[bucket] = self.counts.get(bucket, 0) + 1
            self._attributed += net

    def snapshot(self, wall_s: float) -> dict[str, Any]:
        return {
            "wall_s": round(wall_s, 4),
            "attributed_s": round(self._attributed, 4),
            "unattributed_s": round(max(wall_s - self._attributed, 0.0), 4),
            "buckets": {
                k: {"seconds": round(v, 4), "calls": self.counts[k]}
                for k, v in sorted(self.buckets.items(), key=lambda item: -item[1])
            },
        }


@dataclass(frozen=True)
class SurpriseConfig:
    """Empirical surprise reflex thresholds validated during Gates G10 & G11."""

    check_steps: tuple[int, ...] = (3, 10, 15, 20, 25)
    stall_step_max: int = 5
    stall_threshold_l2: float = 2.20      # G10/G11: Stall error jumps to 2.66 at step 1-3 (vs 1.03 normal)
    slip_step_min: int = 15
    slip_threshold_l2: float = 10.00      # G10/G11: Missed grasp jumps to 14.07 (vs 5.34 normal)
    max_servo_temp_c: float = 58.0        # STS3215 thermal limit before mandatory cooldown
    gripper_min_aperture_deg: float = 0.0  # off (D46): a 12 floor kept the jaws off the bowl's rim (~1.1)
    shadow: bool = False                  # log would-be aborts, never abort (G11-B-cal failed 2026-10-19: robot day 1)


@dataclass
class SurpriseStepCheck:
    """Single-step surprise check outcome."""

    step_k: int
    l2_error: float
    aborted: bool
    reason: str
    inference_ms: float = 0.0
    would_abort: bool = False  # shadow mode: the reflex fired but the arm kept going
    chunk_step: int | None = None  # k within the executing chunk (the W1 check step); step_k counts the episode


@dataclass
class SurpriseTrace:
    """Full episode record of W1 surprise errors."""

    checks: list[SurpriseStepCheck] = field(default_factory=list)
    max_surprise_l2: float = 0.0
    aborted: bool = False
    abort_step: int | None = None
    abort_reason: str = "none"

    def record_check(self, check: SurpriseStepCheck) -> None:
        self.checks.append(check)
        if check.l2_error > self.max_surprise_l2:
            self.max_surprise_l2 = check.l2_error
        if check.aborted and not self.aborted:
            self.aborted = True
            self.abort_step = check.step_k
            self.abort_reason = check.reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [asdict(c) for c in self.checks],
            "max_surprise_l2": round(self.max_surprise_l2, 4),
            "aborted": self.aborted,
            "abort_step": self.abort_step,
            "abort_reason": self.abort_reason,
        }


@dataclass
class HardwareTelemetry:
    """Telemetry readings from STS3215 bus."""

    ok: bool
    max_temp_c: float
    hot_motor_index: int = 0
    temperatures_c: list[float] = field(default_factory=list)
    loads: list[float] = field(default_factory=list)
    voltages_v: list[float] = field(default_factory=list)
    warning: str | None = None


@dataclass
class BlobDetection:
    color: str
    area_px: float
    center_xy: tuple[float, float]
    box_xyxy: tuple[int, int, int, int]


@dataclass
class ContainmentVerdict:
    status: Literal["valid", "invalid"]
    task_success: bool
    failure_mode: str  # "none", "wrong_object", "out_of_bounds", "missed_placement", "source_missing_or_occluded", "motion_or_hand_in_view"
    details: dict[str, Any] = field(default_factory=dict)
    center_distance_px: float | None = None
    mode: str = "standard"


@dataclass
class PreconditionVerdict:
    ready: bool
    reason: str
    positions_xy: dict[str, tuple[float, float]] = field(default_factory=dict)
    distance_px: float | None = None


@dataclass
class EpisodeSummary:
    """Execution summary produced by System 1 runner."""

    steps_executed: int
    wall_duration_s: float
    hardware_fault: bool
    termination_reason: str
    w1_aborted: bool
    w1_max_surprise: float
    w1_abort_reason: str
    timing: dict[str, Any]
    surprise_trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class Outcome:
    """Schema v1.0 outcome record saved to outcome.json."""

    schema_version: str = "1.0"
    attempt_id: int = 1
    instruction: str = ""
    program_ok: bool = True
    task_success: bool = False
    termination_reason: str = "completed"
    w1_surprise_aborted: bool = False
    w1_max_surprise_l2: float = 0.0
    w1_abort_reason: str = "none"
    g9c3_verdict: dict[str, Any] = field(default_factory=dict)
    thermal_max_c: float = 0.0
    steps_executed: int = 0
    wall_duration_s: float = 0.0
    timing: dict[str, Any] = field(default_factory=dict)
    run_dir: str = ""
    skill_library_sha: str = ""

    def save(self, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(asdict(self), indent=2) + "\n")
