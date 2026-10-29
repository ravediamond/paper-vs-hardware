"""WS-047 gate 0, step 1: encoder latents h and joint positions z per frame (PROTOCOL.md + amendment 1).

    python gate0_extract.py --data ~/robot-data/so101-bowls-h264-224 --w1 ~/wm_data/weights_final.pt --out <dir>

Frames: every frame of the 25 G1b val episodes, every 3rd frame of the 157 train episodes (split.json of the
leworldmodel post). h_w1 = W1's fused encoder latent z_t (probe.encode_stages, the latent G1b ranks with);
h_w2p = frozen facebook/dinov2-small patch tokens (W2' encoder), mean-pooled per view, views concatenated.
z = observation.state (6 joint positions, raw units). Writes <out>/latents.npz.
Runs in lerobot-jetson-wm:rc with the W1 pydeps (transformers 5.19.0, strict load).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PVH = Path("/pvh/posts/2026-10-23-paper-vs-hardware-leworldmodel/scripts")
sys.path.insert(0, str(PVH))
import eval_action_sensitivity as E  # noqa: E402
import probe as P  # noqa: E402  CAMS, MEAN, STD, DEV, frame_table, encode_stages


def one_frame(x):
    """One timestamp: LeRobotDataset returns (3, H, W), not (1, 3, H, W). Indexing [0] there took the red channel only
    (fixed 2026-10-28, see the correction note)."""
    return x[0] if x.dim() == 4 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--w1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N frames only")
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from transformers import AutoModel

    split = json.loads((PVH / "split.json").read_text())
    tab = P.frame_table(args.data)
    rows = []  # (global index, episode, is_val)
    for ep in split["val_episodes"]:
        start, acts, _ = tab[ep]
        rows += [(start + i, ep, 1) for i in range(len(acts))]
    for ep in split["train_episodes"]:
        start, acts, _ = tab[ep]
        rows += [(start + i, ep, 0) for i in range(0, len(acts), 3)]
    if args.limit:
        rows = rows[: args.limit]
    idx = np.array([r[0] for r in rows])
    w1, _ = E.load(args.w1.parent, P.DEV)
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(P.DEV).eval()
    ds = LeRobotDataset(f"local/{args.data.name}", root=args.data, delta_timestamps={c: [0.0] for c in P.CAMS.values()})

    class Frames(torch.utils.data.Dataset):
        def __len__(self):
            return len(idx)

        def __getitem__(self, i):
            it = ds[int(idx[i])]
            out = {v: ((one_frame(it[c]) - P.MEAN[0]) / P.STD[0]) for v, c in P.CAMS.items()}
            out["state"] = it["observation.state"]
            return out

    dl = torch.utils.data.DataLoader(Frames(), batch_size=args.bs, num_workers=4, shuffle=False)
    hw1, hdp, zs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for k, b in enumerate(dl):
            frames = {v: b[v] for v in P.CAMS}
            _, _, _, z = P.encode_stages(w1, frames)
            hw1.append(z.float().cpu().numpy())
            pooled = [dino(pixel_values=b[v].to(P.DEV)).last_hidden_state[:, 1:].mean(1) for v in P.CAMS]
            hdp.append(torch.cat(pooled, 1).float().cpu().numpy())
            zs.append(b["state"].float().numpy().reshape(len(b["state"]), -1))
            if k % 20 == 0:
                print(f"batch {k}/{len(dl)} {time.time() - t0:.0f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "latents.npz", h_w1=np.concatenate(hw1), h_w2p=np.concatenate(hdp),
                        z=np.concatenate(zs), index=idx, episode=np.array([r[1] for r in rows]),
                        is_val=np.array([r[2] for r in rows]))
    print(f"done: {len(idx)} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
