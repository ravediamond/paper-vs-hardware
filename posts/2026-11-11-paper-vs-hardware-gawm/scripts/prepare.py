"""WS-168 step 1a: decode the bowls dataset (LeRobot v3, AV1 mp4) into flat uint8 arrays for the GAWM-S trainer.

  python prepare.py --root <dataset snapshot dir> --out <dir> [--episodes 0,1,172]

Writes (episodes in the order given, default all): arrays.npz (action, state, task_index, ep_of_frame),
episodes.json (orig index, start, length, task text), frames_top.u8 / frames_wrist.u8 (N,256,256,3 uint8, squashed
with cv2.INTER_AREA). The two cameras are `front_top_camera` ("top") and `wrist_camera` ("wrist").
"""
import argparse, glob, json, os
import av
import cv2
import numpy as np
import pandas as pd

CAMS = {"top": "observation.images.front_top_camera", "wrist": "observation.images.wrist_camera"}
SIZE = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", default="")
    a = ap.parse_args()
    info = json.load(open(f"{a.root}/meta/info.json"))
    fps = float(info["fps"])
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.root}/meta/episodes/*/*.parquet"))])
    eps = eps.sort_values("episode_index").set_index("episode_index", drop=False)
    sel = [int(x) for x in a.episodes.split(",")] if a.episodes else [int(x) for x in eps.index]
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.root}/data/*/*.parquet"))]).sort_values("index")
    tasks = pd.read_parquet(f"{a.root}/meta/tasks.parquet")
    task_text = {int(r.task_index): str(i) for i, r in tasks.iterrows()}
    os.makedirs(a.out, exist_ok=True)

    action, state, task_idx, ep_of = [], [], [], []
    meta, start = [], 0
    for k, e in enumerate(sel):
        d = data[data.episode_index == e].sort_values("frame_index")
        n = len(d)
        assert n == int(eps.loc[e, "length"]), (e, n)
        action.append(np.stack(d["action"].to_numpy()).astype(np.float32))
        state.append(np.stack(d["observation.state"].to_numpy()).astype(np.float32))
        ti = int(d["task_index"].iloc[0])
        task_idx.append(np.full(n, ti, np.int32))
        ep_of.append(np.full(n, k, np.int32))
        meta.append({"orig": e, "start": start, "length": n, "task": task_text[ti]})
        start += n
    total = start
    np.savez(f"{a.out}/arrays.npz", action=np.concatenate(action), state=np.concatenate(state),
             task_index=np.concatenate(task_idx), ep_of_frame=np.concatenate(ep_of))
    json.dump({"episodes": meta, "fps": fps, "total": total, "size": SIZE}, open(f"{a.out}/episodes.json", "w"))

    for cam, key in CAMS.items():
        mm = np.lib.format.open_memmap(f"{a.out}/frames_{cam}.npy", mode="w+", dtype=np.uint8, shape=(total, SIZE, SIZE, 3))
        filled = np.zeros(total, bool)
        by_file = {}
        for m in meta:
            r = eps.loc[m["orig"]]
            f = (int(r[f"videos/{key}/chunk_index"]), int(r[f"videos/{key}/file_index"]))
            by_file.setdefault(f, []).append((float(r[f"videos/{key}/from_timestamp"]), m))
        for (ch, fi), lst in by_file.items():
            lst.sort(key=lambda x: x[0])
            path = f"{a.root}/videos/{key}/chunk-{ch:03d}/file-{fi:03d}.mp4"
            with av.open(path) as c:
                s = c.streams.video[0]
                tb = float(s.time_base)
                c.seek(max(0, int((lst[0][0] - 0.5) / tb)), stream=s, backward=True)
                p = 0
                end = lst[-1][0] + lst[-1][1]["length"] / fps
                for fr in c.decode(s):
                    t = float(fr.pts) * tb
                    if t > end + 1.0 / fps:
                        break
                    while p < len(lst) - 1 and t >= lst[p + 1][0] - 0.5 / fps:
                        p += 1
                    ts, m = lst[p]
                    i = int(round((t - ts) * fps))
                    if 0 <= i < m["length"]:
                        img = cv2.resize(fr.to_ndarray(format="rgb24"), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                        mm[m["start"] + i] = img
                        filled[m["start"] + i] = True
        mm.flush()
        print(cam, "filled", int(filled.sum()), "of", total, flush=True)
        assert filled.all(), f"{cam}: {int((~filled).sum())} frames missing"
    print("done", total, "frames,", len(sel), "episodes")


if __name__ == "__main__":
    main()
