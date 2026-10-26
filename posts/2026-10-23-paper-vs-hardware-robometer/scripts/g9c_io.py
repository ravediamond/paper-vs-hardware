"""G9c: read the frames the judge needs from a LeRobot v3.0 dataset (front_top_camera).

For one episode: its first frame and the frames at 0, 0.2, ..., max_window_s seconds before its last frame.
"""
import glob, json

import av
import numpy as np
import pandas as pd

CAM = "observation.images.front_top_camera"


class Episodes:
    def __init__(self, root):
        self.root = root
        self.info = json.load(open(f"{root}/meta/info.json")); self.fps = self.info["fps"]
        self.meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))]).set_index("episode_index")

    def ids(self):
        return list(self.meta.index)

    def frames(self, ep, max_window_s, step_s, end_frame=None):
        """Returns first frame (bgr), [(t_from_end, bgr)] of the end window, and the window of the episode START
        (same spacing, from the first frame forward): used as 'not done' negatives on dev data.
        end_frame: judge as if the episode ended at this frame index (exploratory: operator-marked reset_start)."""
        m = self.meta.loc[ep]
        path = f"{self.root}/" + self.info["video_path"].format(
            video_key=CAM, chunk_index=int(m[f"videos/{CAM}/chunk_index"]), file_index=int(m[f"videos/{CAM}/file_index"]))
        t0, t1 = float(m[f"videos/{CAM}/from_timestamp"]), float(m[f"videos/{CAM}/to_timestamp"])
        last = t1 - 1.0 / self.fps if end_frame is None else min(t1 - 1.0 / self.fps, t0 + (end_frame - 1) / self.fps)
        offs = np.round(np.arange(0, max_window_s + 1e-9, step_s), 3)
        want = {"first": t0, **{f"end{o}": max(t0, last - o) for o in offs}}
        got = {}
        with av.open(path) as c:
            s = c.streams.video[0]; s.thread_type = "AUTO"
            for key, t in sorted(want.items(), key=lambda kv: kv[1]):
                c.seek(int(max(t - 0.5, 0) / s.time_base), stream=s, backward=True)
                for fr in c.decode(s):
                    if round(float(fr.pts * s.time_base) * self.fps) >= round(t * self.fps):
                        got[key] = fr.to_ndarray(format="bgr24"); break
        return got["first"], [(float(o), got[f"end{o}"]) for o in offs], float(t1 - t0)
