"""WS-107 §6.5 (O3, D1): G9c HSV bowl-mask area on the wrist camera, every 6th raw frame (5 fps) of the val episodes.
area = largest-component area of the blue bowl + that of the pink bowl (g9c_judge.objects, thresholds of judge/g9c_config.json, unchanged),
on the 224 x 224 transcoded wrist frames. Coverage of frame t = area_t / area_0 (first frame of the episode); area_0 == 0 -> episode excluded (D1).
  python3 wrist_mask.py --data ~/robot-data/so101-bowls-h264-224 --split ../vla-jepa-world-model/wm_train/split.json --out wrist_mask.csv [--max-episodes N] [--contact-sheet PNG]"""
import argparse, glob, json, sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vla-jepa-world-model" / "judge"))
import g9c_judge as G  # noqa: E402

CAM = "observation.images.wrist_camera"


def episode_frames(root, info, meta, ep, every=6):
    m = meta.loc[ep]
    path = f"{root}/" + info["video_path"].format(video_key=CAM, chunk_index=int(m[f"videos/{CAM}/chunk_index"]), file_index=int(m[f"videos/{CAM}/file_index"]))
    fps = info["fps"]; t0, t1 = float(m[f"videos/{CAM}/from_timestamp"]), float(m[f"videos/{CAM}/to_timestamp"])
    n = int(round((t1 - t0) * fps)); out = {}
    with av.open(path) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            i = int(round((float(fr.pts * s.time_base) - t0) * fps))
            if i < 0: continue
            if i >= n: break
            if i % every == 0: out[i] = fr.to_ndarray(format="bgr24")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=0); ap.add_argument("--contact-sheet")
    a = ap.parse_args()
    root = str(Path(a.data).expanduser()); info = json.load(open(f"{root}/meta/info.json"))
    meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))]).set_index("episode_index")
    cfg = G.load_config(str(Path(G.__file__).with_name("g9c_config.json")))
    eps = json.load(open(a.split))["val_episodes"][: a.max_episodes or None]
    rows, sheet = [], []
    for ep in eps:
        fr = episode_frames(root, info, meta, ep)
        for i in sorted(fr):
            o = G.objects(fr[i], cfg)
            rows.append({"episode": ep, "frame": i, "area_blue": o["blue"][0], "area_pink": o["pink"][0]})
        if a.contact_sheet and len(sheet) < 2:
            sheet.append((ep, fr))
    df = pd.DataFrame(rows); df["area"] = df.area_blue + df.area_pink
    first = df[df.frame == 0].set_index("episode").area
    df["area0"] = df.episode.map(first); df["coverage"] = np.where(df.area0 > 0, df.area / df.area0.clip(lower=1), np.nan)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} frames, {df.episode.nunique()} episodes, episodes with area0 == 0: {sorted(first[first == 0].index.tolist())}")
    if a.contact_sheet:
        import cv2
        tiles = []
        for ep, fr in sheet:
            keys = sorted(fr)[:: max(1, len(fr) // 8)][:8]; strip = []
            for i in keys:
                img = fr[i].copy(); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                for name, col in (("blue", (255, 0, 0)), ("pink", (255, 0, 255))):
                    m = np.zeros(hsv.shape[:2], np.uint8)
                    for lo, hi in cfg["colors"][name]["hsv_ranges"]: m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                    img[m > 0] = (0.4 * img[m > 0] + 0.6 * np.array(col)).astype(np.uint8)
                cov = df[(df.episode == ep) & (df.frame == i)].coverage.iloc[0]
                cv2.putText(img, f"e{ep} f{i} {cov:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1); strip.append(img)
            tiles.append(np.concatenate(strip, 1))
        cv2.imwrite(a.contact_sheet, np.concatenate(tiles, 0))


if __name__ == "__main__":
    main()
