"""G14: do our policies oscillate between poses? (REKI's metric, operational definition in V1.0_REVIEW.md §6)

For each episode of a LeRobot v3 rollout dataset:
- split the executed joint trajectory (`observation.state`, the five arm joints) into the policy's chunks
  (synchronous rollouts: one chunk every `n_action_steps` frames from the start of the episode);
- mean velocity per chunk  v_k = (q_end - q_start) / steps;
- a reversal at chunk k when cos(v_{k-1}, v_k) < -0.5 and both chunks move more than 3 degrees in total;
- the episode oscillates when reversals occur at >= 3 consecutive chunks (REKI: "goes back and forth at least twice").

Units: the 3-degree threshold assumes joint states in degrees; the summary prints the arm-joint range to check.

A second, diagnostic column: whether the object's top-camera blob (G9c-3 colour mask) stays within 5 px over the
longest reversal run. It is not part of the criterion (the arm occludes the object while it moves).

Bar (fixed before running): REKI-style registers stay in v1.0 only if >= 10 % of failed episodes oscillate.

Usage:
  python scripts/run_g14_oscillation_audit.py --repo-id ravediamond/so101-bowls-policy-rollouts \
      --labels data/rollouts/so101-bowls-policy-rollouts_labels.csv --out shared/gate/data/<date>_g14
  # or --root <local LeRobot dataset dir> instead of --repo-id; add --blob to compute the diagnostic column
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ARM_JOINTS = slice(0, 5)  # SO-101: pan, lift, elbow, wrist_flex, wrist_roll (gripper excluded)

# n_action_steps of the checkpoints behind each rollout set (shared/gate/data/*/README.md):
# ACT 100, SmolVLA (PyTorch and vla.cpp) 50, VLA-JEPA 30, TurboVLA 12 (2026-10-01 runs).
DEFAULT_CHUNKS = {
    "act": 100,
    "smolvla": 50,
    "smolvla_vlacpp": 50,
    "smolvla_vlacpp_fa": 50,
    "vla_jepa": 30,
    "turbovla_a": 12,
    "turbovla_c": 12,
    "turbovla_trimmed": 12,
    "turbovla_v2": 12,
}
OBJECT_COLOR = "blue"  # every task in this set moves the blue bowl


@dataclass
class EpisodeResult:
    episode_index: int
    policy: str
    outcome: str
    failure_mode: str
    chunk_steps: int
    n_chunks: int
    n_reversals: int
    longest_run: int
    oscillating: bool
    run_start_frame: int
    run_end_frame: int
    blob_shift_px: float | None = None
    blob_stationary: bool | None = None


def chunk_velocities(q: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean velocity, displacement norm and start frame of each complete chunk of `steps` frames."""
    n = len(q) // steps
    starts = np.arange(n) * steps
    ends = starts + steps - 1
    disp = q[ends] - q[starts]
    return disp / steps, np.linalg.norm(disp, axis=1), starts


def reversal_runs(
    vel: np.ndarray, disp: np.ndarray, cos_thresh: float = -0.5, min_move_deg: float = 3.0
) -> np.ndarray:
    """Boolean per chunk k >= 1: chunk k reverses chunk k-1."""
    rev = np.zeros(len(vel), dtype=bool)
    for k in range(1, len(vel)):
        if disp[k - 1] <= min_move_deg or disp[k] <= min_move_deg:
            continue
        a, b = vel[k - 1], vel[k]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        rev[k] = cos < cos_thresh
    return rev


def longest_true_run(flags: np.ndarray) -> tuple[int, int]:
    """(length, start index) of the longest run of True."""
    best, best_start, cur, cur_start = 0, -1, 0, 0
    for i, f in enumerate(flags):
        if f:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best:
                best, best_start = cur, cur_start
        else:
            cur = 0
    return best, best_start


def audit_episode(
    q: np.ndarray, steps: int, *, cos_thresh: float = -0.5, min_move_deg: float = 3.0, consecutive: int = 3
) -> dict:
    vel, disp, starts = chunk_velocities(q[:, ARM_JOINTS], steps)
    rev = reversal_runs(vel, disp, cos_thresh, min_move_deg)
    run, run_start = longest_true_run(rev)
    if run > 0:
        first_chunk = run_start - 1  # the run compares chunk k with k-1
        last_chunk = run_start + run - 1
        f0, f1 = int(starts[first_chunk]), int(starts[last_chunk] + steps - 1)
    else:
        f0 = f1 = -1
    return {
        "n_chunks": int(len(vel)),
        "n_reversals": int(rev.sum()),
        "longest_run": int(run),
        "oscillating": bool(run >= consecutive),
        "run_start_frame": f0,
        "run_end_frame": f1,
    }


# ---------------------------------------------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------------------------------------------
def load_frames(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data/chunk-*/file-*.parquet under {root}")
    cols = ["episode_index", "frame_index", "observation.state"]
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def episode_states(frames: pd.DataFrame, episode: int) -> np.ndarray:
    ep = frames[frames.episode_index == episode].sort_values("frame_index")
    return np.stack(ep["observation.state"].to_numpy()).astype(np.float64)


def download(repo_id: str, with_videos: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    patterns = None if with_videos else ["data/*", "meta/*"]
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=patterns))


class TopVideo:
    """Reads single frames of the top camera for an episode (LeRobot v3 layout, PyAV)."""

    def __init__(self, root: Path, camera: str = "observation.images.front_top_camera") -> None:
        import av  # noqa: F401  (fail early if missing)

        self.root, self.camera = root, camera
        files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        self.meta = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        info = json.loads((root / "meta" / "info.json").read_text())
        self.fps = info["fps"]
        self.video_path = info["video_path"]

    def frame(self, episode: int, frame_index: int) -> np.ndarray:
        import av

        row = self.meta[self.meta.episode_index == episode].iloc[0]
        chunk = int(row[f"videos/{self.camera}/chunk_index"])
        file = int(row[f"videos/{self.camera}/file_index"])
        t = float(row[f"videos/{self.camera}/from_timestamp"]) + frame_index / self.fps
        path = self.root / self.video_path.format(video_key=self.camera, chunk_index=chunk, file_index=file)
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            c.seek(int(max(t - 1.0, 0) / stream.time_base), stream=stream, backward=True)
            for fr in c.decode(stream):
                if fr.time is not None and fr.time >= t - 0.5 / self.fps:
                    return fr.to_ndarray(format="rgb24")
        raise ValueError(f"frame {frame_index} of episode {episode} not found")


def blob_shift(video: TopVideo, episode: int, f0: int, f1: int, color: str = OBJECT_COLOR) -> float | None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from so101_harness.judge_g9c3 import DeterministicTopJudge

    judge = DeterministicTopJudge()
    a = judge.extract_blob(video.frame(episode, f0), color)
    b = judge.extract_blob(video.frame(episode, f1), color)
    if a is None or b is None:
        return None
    return float(np.hypot(*np.subtract(a["center_xy"], b["center_xy"])))


# ---------------------------------------------------------------------------------------------------------------
def summarize(results: list[EpisodeResult], bar: float = 0.10) -> dict:
    df = pd.DataFrame([asdict(r) for r in results])
    failed = df[df.outcome == "failure"]
    main = df[~df.policy.str.startswith("turbovla")]
    main_failed = main[main.outcome == "failure"]

    def share(d: pd.DataFrame) -> dict:
        return {"episodes": int(len(d)), "oscillating": int(d.oscillating.sum()),
                "share": round(float(d.oscillating.mean()), 3) if len(d) else None}

    out = {
        "all": share(df),
        "failed": share(failed),
        "succeeded": share(df[df.outcome == "success"]),
        "failed_without_turbovla": share(main_failed),
        "per_policy": {p: share(g) for p, g in df.groupby("policy")},
        "bar": f">= {bar:.0%} of failed episodes oscillate",
    }
    s = out["failed"]["share"]
    out["verdict"] = None if s is None else ("keep registers (G14 pass)" if s >= bar else "drop registers (G14 fail)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, help="local LeRobot v3 dataset directory")
    src.add_argument("--repo-id", help="Hugging Face dataset to download (data/ and meta/ only)")
    ap.add_argument("--labels", type=Path, required=True, help="per-episode labels CSV (episode_index, policy, outcome, ...)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", action="append", default=[], help="override policy=steps, repeatable")
    ap.add_argument("--cos", type=float, default=-0.5)
    ap.add_argument("--min-move-deg", type=float, default=3.0)
    ap.add_argument("--consecutive", type=int, default=3)
    ap.add_argument("--blob", action="store_true", help="also compute the object-stationary column (decodes video)")
    args = ap.parse_args(argv)

    chunks = dict(DEFAULT_CHUNKS)
    for item in args.chunk:
        k, v = item.split("=")
        chunks[k] = int(v)

    root = args.root if args.root else download(args.repo_id, with_videos=args.blob)
    frames = load_frames(root)
    labels = pd.read_csv(args.labels)
    video = TopVideo(root) if args.blob else None

    results: list[EpisodeResult] = []
    for row in labels.itertuples():
        if row.policy not in chunks:
            print(f"skip episode {row.episode_index}: no chunk size for policy {row.policy!r}", file=sys.stderr)
            continue
        q = episode_states(frames, int(row.episode_index))
        if int(row.reset_start) >= 0:  # drop the move back to the rest pose
            q = q[: int(row.reset_start)]
        r = audit_episode(q, chunks[row.policy], cos_thresh=args.cos, min_move_deg=args.min_move_deg,
                          consecutive=args.consecutive)
        res = EpisodeResult(int(row.episode_index), row.policy, row.outcome,
                            "" if pd.isna(row.failure_mode) else row.failure_mode, chunks[row.policy], **r)
        if video is not None and r["longest_run"] > 0:
            try:
                shift = blob_shift(video, res.episode_index, r["run_start_frame"], r["run_end_frame"])
                res.blob_shift_px = None if shift is None else round(shift, 1)
                res.blob_stationary = None if shift is None else shift < 5.0
            except Exception as exc:  # diagnostic only
                print(f"blob check failed for episode {res.episode_index}: {exc}", file=sys.stderr)
        results.append(res)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in results]).to_csv(args.out / "g14_episodes.csv", index=False)
    summary = summarize(results)
    arm = np.stack(frames["observation.state"].to_numpy())[:, ARM_JOINTS]
    # The 3-degree threshold assumes joint states in degrees (LeRobot `use_degrees`). Check this range: a
    # [-100, 100] normalized calibration would need --min-move-deg rescaled.
    summary["arm_state_range"] = {"min": np.round(arm.min(0), 1).tolist(), "max": np.round(arm.max(0), 1).tolist()}
    summary["params"] = {"cos": args.cos, "min_move_deg": args.min_move_deg, "consecutive": args.consecutive,
                         "chunks": chunks, "source": args.repo_id or str(args.root)}
    (args.out / "g14_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
