"""Frozen DINOv2 ViT-S/14 patch tokens for W2 (G1_PROTOCOL.md): every frame of both cameras of the bowls data at 224x224.

Input: the transcoded snapshot (H.264 224x224, ImageNet normalization as the LeWM pipeline). Output per episode and view:
<out>/ep{N:06d}_{pixels|wrist}.npy, float16, shape (frames, 256, 384) (16x16 patches of the last_hidden_state, CLS dropped), or
(frames, 64, 384) with --pool2 (2x2 average pooling; the brief allows it only if memory requires, record when used).
Model: facebook/dinov2-small (Apache-2.0). Position embeddings are interpolated from the 518 px pre-training size.

    python extract_dino_features.py --data <snapshot> --out <dir> [--episodes 4 5 | --split val|train|all] [--pool2]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1b_range as G  # noqa: E402  FrameSet
import probe as P  # noqa: E402  frame_table, CAMS, DEV

MODEL = "facebook/dinov2-small"


@torch.no_grad()
def encode(model, x: torch.Tensor, pool2: bool) -> torch.Tensor:
    h = model(pixel_values=x.to(P.DEV)).last_hidden_state[:, 1:]            # (N, 256, 384)
    if pool2:
        n, _, c = h.shape
        h = h.view(n, 8, 2, 8, 2, c).mean(dim=(2, 4)).reshape(n, 64, c)
    return h.half().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--split", choices=["train", "val", "all"], default="val")
    ap.add_argument("--pool2", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from transformers import AutoModel

    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    eps = args.episodes or {"train": split["train_episodes"], "val": split["val_episodes"],
                            "all": split["train_episodes"] + split["val_episodes"]}[args.split]
    model = AutoModel.from_pretrained(MODEL).to(P.DEV).eval()
    tab = P.frame_table(args.data)
    ds = LeRobotDataset(f"local/{args.data.name}", root=args.data, delta_timestamps={c: [0.0] for c in P.CAMS.values()})
    args.out.mkdir(parents=True, exist_ok=True)
    t0, frames = time.time(), 0
    for ep in eps:
        start, act, _ = tab[ep]
        loader = torch.utils.data.DataLoader(G.FrameSet(ds, np.arange(start, start + len(act))), batch_size=64,
                                             num_workers=args.workers, shuffle=False)
        feats = {v: [] for v in P.CAMS}
        for b in loader:
            for v in P.CAMS:
                feats[v].append(encode(model, b[v], args.pool2))
        for v in P.CAMS:
            np.save(args.out / f"ep{ep:06d}_{v}.npy", torch.cat(feats[v]).numpy())
        frames += len(act)
        print(f"episode {ep}: {len(act)} frames, {frames / (time.time() - t0):.1f} frames/s (both views) so far", flush=True)
    print(f"done: {frames} frames in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
