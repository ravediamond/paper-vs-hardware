"""WS-107 ladder rung 5 (F_dino): frozen DINOv2-S (facebook/dinov2-small) CLS token of the top and the wrist frame, val episodes, every 6th raw
frame, 224 x 224 transcoded frames, the Hub model's own image processor (PROTOCOL.md §4). No training.
  python3 fdino.py --data DIR --split split.json --out-dir DIR [--max-episodes N] [--device cpu]  -> fdino_top.npz, fdino_wrist.npz (emb, episode, frame)"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel

from wrist_mask import episode_frames
import wrist_mask

CAMS = {"top": "observation.images.front_top_camera", "wrist": "observation.images.wrist_camera"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--split", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-episodes", type=int, default=0); ap.add_argument("--device", default="cpu"); ap.add_argument("--proc-dir", default="facebook/dinov2-small", help="dir with the Hub preprocessor_config.json")
    a = ap.parse_args()
    root = str(Path(a.data).expanduser()); info = json.load(open(f"{root}/meta/info.json"))
    meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))]).set_index("episode_index")
    eps = json.load(open(a.split))["val_episodes"][: a.max_episodes or None]
    proc = AutoImageProcessor.from_pretrained(a.proc_dir); model = AutoModel.from_pretrained("facebook/dinov2-small").to(a.device).eval()
    for name, cam in CAMS.items():
        wrist_mask.CAM = cam
        embs, epi, frm = [], [], []
        for ep in eps:
            fr = episode_frames(root, info, meta, ep)
            keys = sorted(fr)
            imgs = [np.ascontiguousarray(fr[k][:, :, ::-1]) for k in keys]  # bgr -> rgb
            for s in range(0, len(imgs), 64):
                x = proc(images=imgs[s:s + 64], return_tensors="pt")["pixel_values"].to(a.device)
                with torch.no_grad():
                    embs.append(model(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy())
            epi += [ep] * len(keys); frm += keys
        np.savez(Path(a.out_dir) / f"fdino_{name}.npz", emb=np.concatenate(embs).astype(np.float32), episode=np.array(epi), frame=np.array(frm))
        print(name, len(frm))


if __name__ == "__main__":
    main()
