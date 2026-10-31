"""Read episodes of a local LeRobot v3.0 dataset: per-frame state and action, and decoded camera frames.

Layout (LeRobot v3.0): `data/chunk-*/file-*.parquet` (all episodes, one row per frame), `meta/info.json` (`fps`,
`video_path`), `meta/episodes/chunk-*/file-*.parquet` (per episode: `videos/<key>/chunk_index`, `file_index`,
`from_timestamp`, `to_timestamp`). Several episodes share one video file; an episode is the span
[from_timestamp, to_timestamp) of it.
"""

from __future__ import annotations

from functools import cached_property
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

TOP = "observation.images.front_top_camera"
WRIST = "observation.images.wrist_camera"


def download(repo_id: str, with_videos: bool = True) -> Path:
    from huggingface_hub import snapshot_download

    patterns = None if with_videos else ["data/*", "meta/*"]
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=patterns))


class EpisodeDataset:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = float(self.info["fps"])

    @cached_property
    def frames(self) -> pd.DataFrame:
        files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no data/chunk-*/file-*.parquet under {self.root}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    @cached_property
    def episodes_meta(self) -> pd.DataFrame:
        files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def episode_indices(self) -> list[int]:
        return sorted(int(e) for e in self.frames.episode_index.unique())

    def _rows(self, episode: int) -> pd.DataFrame:
        return self.frames[self.frames.episode_index == episode].sort_values("frame_index")

    def states(self, episode: int) -> np.ndarray:
        return np.stack(self._rows(episode)["observation.state"].to_numpy()).astype(np.float64)

    def actions(self, episode: int) -> np.ndarray:
        return np.stack(self._rows(episode)["action"].to_numpy()).astype(np.float64)

    def task(self, episode: int) -> str:
        rows = self._rows(episode)
        if "task" in rows:
            return str(rows["task"].iloc[0])
        tasks_file = self.root / "meta" / "tasks.parquet"
        if "task_index" in rows and tasks_file.exists():
            tasks = pd.read_parquet(tasks_file)
            idx = int(rows["task_index"].iloc[0])
            if "task" in tasks.columns:
                return str(tasks[tasks.get("task_index", tasks.index) == idx]["task"].iloc[0])
            return str(tasks.index[idx])
        return ""

    def _video(self, episode: int, camera: str) -> tuple[Path, float, float]:
        row = self.episodes_meta[self.episodes_meta.episode_index == episode].iloc[0]
        chunk = int(row[f"videos/{camera}/chunk_index"])
        file = int(row[f"videos/{camera}/file_index"])
        t0 = float(row[f"videos/{camera}/from_timestamp"])
        key_to = f"videos/{camera}/to_timestamp"
        t1 = float(row[key_to]) if key_to in row else float("inf")
        path = self.root / self.info["video_path"].format(video_key=camera, chunk_index=chunk, file_index=file)
        return path, t0, t1

    def iter_frames(self, episode: int, camera: str = TOP, stride: int = 1,
                    size: tuple[int, int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """(frame_index, RGB uint8 HWC) for every `stride`-th frame of the episode, decoded sequentially.
        `size` = (width, height) resizes (cv2, area interpolation)."""
        import av

        path, t0, t1 = self._video(episode, camera)
        n = len(self._rows(episode))
        next_idx = 0
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            c.seek(int(max(t0 - 1.0, 0) / stream.time_base), stream=stream, backward=True)
            for fr in c.decode(stream):
                if fr.time is None or fr.time < t0 - 0.5 / self.fps:
                    continue
                idx = int(round((fr.time - t0) * self.fps))
                if idx >= n or fr.time >= t1 - 0.5 / self.fps:
                    break
                if idx < next_idx:
                    continue
                if idx % stride == 0:
                    img = fr.to_ndarray(format="rgb24")
                    if size is not None:
                        import cv2

                        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    yield idx, img
                next_idx = idx + 1

    def frame(self, episode: int, frame_index: int, camera: str = TOP) -> np.ndarray:
        for idx, img in self.iter_frames(episode, camera):
            if idx == frame_index:
                return img
        raise ValueError(f"frame {frame_index} of episode {episode} not found")


def write_synthetic_dataset(root: Path, episodes: list[dict], fps: int = 30, cameras=(TOP, WRIST)) -> None:
    """Test helper: a LeRobot-v3-shaped dataset. Each episode dict: `states` (T, 6), optional `actions`,
    and `frames` {camera: list of HxWx3 uint8} (encoded with mpeg4, all episodes in one file per camera)."""
    import av

    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    rows, meta = [], []
    t = 0.0
    for e, ep in enumerate(episodes):
        T = len(ep["states"])
        acts = ep.get("actions", ep["states"])
        for i in range(T):
            rows.append({"episode_index": e, "frame_index": i, "timestamp": i / fps, "task": ep.get("task", ""),
                         "observation.state": np.asarray(ep["states"][i], np.float32),
                         "action": np.asarray(acts[i], np.float32)})
        m = {"episode_index": e, "length": T}
        for cam in cameras:
            m[f"videos/{cam}/chunk_index"] = 0
            m[f"videos/{cam}/file_index"] = 0
            m[f"videos/{cam}/from_timestamp"] = t
            m[f"videos/{cam}/to_timestamp"] = t + T / fps
        meta.append(m)
        t += T / fps
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(meta).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "info.json").write_text(json.dumps({
        "fps": fps, "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}))
    for cam in cameras:
        d = root / "videos" / cam / "chunk-000"
        d.mkdir(parents=True, exist_ok=True)
        first = episodes[0]["frames"][cam][0]
        with av.open(str(d / "file-000.mp4"), "w") as c:
            st = c.add_stream("mpeg4", rate=fps)
            st.width, st.height, st.pix_fmt = first.shape[1], first.shape[0], "yuv420p"
            st.options = {"qscale": "2"}
            for ep in episodes:
                for img in ep["frames"][cam]:
                    for p in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                        c.mux(p)
            for p in st.encode():
                c.mux(p)
