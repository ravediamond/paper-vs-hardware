"""Convert held-out episodes of ravediamond/so101_bowls_all_trimmed (LeRobot v3) to the robo_jepa on-disk format (docs/data.md).

    uv run --with pandas --with pyarrow --with h5py python convert_bowls.py --src <snapshot dir> --out <dir> --split ../wm_train/split.json [--n 25]

Per episode: data.h5 with trajectory/observation/robot_state/{joint_position [T,5] rad, gripper_position [T,1] = raw * pi/180,
cartesian_position [T,6] zeros (unused by the so-101_joints id)}, cameras/<name> -> mp4, and one H.264 mp4 per camera at 30 fps, cut from
the concatenated v3 video by the episode timestamps (frame count checked against the parquet length). Units: LeRobot values are degrees
(shoulder_lift reaches -102, so not the +-100 normalized range), converted with pi/180 as the robo_jepa SO-101 doc does for LeRobot.
Also writes dataset_index.csv and episodes.json (episode ids, lengths). Episodes are the G1 validation episodes of split.json.
"""

import argparse
import glob
import json
import math
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

CAMS = {"front_top_camera": "front_top_camera", "wrist_camera": "wrist_camera"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()
    split = json.loads(a.split.read_text())
    eps = split["val_episodes"][: a.n]
    meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(a.src / "meta/episodes/*/*.parquet")))]).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(a.src / "data/*/*.parquet")))]).set_index("index")
    a.out.mkdir(parents=True, exist_ok=True)
    rows, info = [], {}
    for ep in eps:
        m = meta.loc[ep]
        d = data.loc[int(m["dataset_from_index"]):int(m["dataset_to_index"]) - 1]
        T = len(d)
        st = np.stack(d["observation.state"].to_numpy()).astype(np.float64)
        ed = a.out / f"episode_{ep:04d}"
        ed.mkdir(exist_ok=True)
        frames = {}
        for cam in CAMS:
            k = f"videos/observation.images.{cam}"
            src = a.src / f"videos/observation.images.{cam}/chunk-{int(m[k + '/chunk_index']):03d}/file-{int(m[k + '/file_index']):03d}.mp4"
            t0, t1 = float(m[k + "/from_timestamp"]), float(m[k + "/to_timestamp"])
            dst = ed / f"{cam}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.6f}", "-i", str(src), "-t", f"{t1 - t0:.6f}", "-an", "-r", "30",
                            "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", str(dst)], check=True)
            n = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                                    "stream=nb_read_frames", "-of", "csv=p=0", str(dst)], capture_output=True, text=True, check=True).stdout.strip())
            frames[cam] = n
        with h5py.File(ed / "data.h5", "w") as f:
            rs = f.create_group("trajectory/observation/robot_state")
            rs["joint_position"] = (st[:, :5] * math.pi / 180).astype(np.float32)
            rs["gripper_position"] = (st[:, 5:6] * math.pi / 180).astype(np.float32)
            rs["cartesian_position"] = np.zeros((T, 6), np.float32)
            cg = f.create_group("cameras")
            for cam in CAMS:
                cg[cam] = f"{cam}.mp4".encode()
        rows.append(f"episode_{ep:04d}")
        info[ep] = {"length": T, "frames": frames}
        bad = {c: n for c, n in frames.items() if n != T}
        print(ep, T, frames, "MISMATCH" if bad else "ok", flush=True)
    pd.DataFrame({"path": rows}).to_csv(a.out / "dataset_index.csv", index=False)
    (a.out / "episodes.json").write_text(json.dumps({"dataset": split["dataset"], "revision": split["revision"], "episodes": info}, indent=1))


if __name__ == "__main__":
    main()
