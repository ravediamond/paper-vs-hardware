"""WS-051 gate 0, step 1: W1 latents for every frame + W1 low-level rollouts on the val anchors (PROTOCOL.md).

    python hwm_extract.py --data ~/robot-data/so101-bowls-h264-224 --w1 ~/wm_data/weights_final.pt --out <dir>

Writes <out>/latents.npz: z (frames, 192) in global-index order, actions (frames, 6) raw, episode, frame, is_val,
a_mean / a_std (train split); and <out>/lowlevel.npz: anchors (global index), pred (anchors, 5, 192) = W1's rollout
read at t + h for h in H, true (anchors, 5, 192). Runs in lerobot-jetson-wm:rc (transformers 5.19.0, strict load).
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
import eval_action_sensitivity as E  # noqa: E402  load, rollout
import probe as P  # noqa: E402  CAMS, MEAN, STD, DEV, frame_table, encode_stages

H = [22, 30, 38, 45, 60]
C, STRIDE = 3, 10


def one_frame(x):
    """One timestamp: LeRobotDataset returns (3, H, W), not (1, 3, H, W). Indexing [0] there took the red channel only
    (fixed 2026-10-28, see the correction note)."""
    return x[0] if x.dim() == 4 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--w1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bs", type=int, default=96)
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    split = json.loads((PVH / "split.json").read_text())
    tab = P.frame_table(args.data)
    eps = sorted(tab)
    starts = {e: tab[e][0] for e in eps}
    n = max(starts[e] + len(tab[e][1]) for e in eps)
    actions = np.zeros((n, 6), np.float32)
    episode = np.full(n, -1)
    frame = np.full(n, -1)
    for e in eps:
        s, a, _ = tab[e]
        actions[s:s + len(a)] = a
        episode[s:s + len(a)] = e
        frame[s:s + len(a)] = np.arange(len(a))
    val_set = set(split["val_episodes"])
    is_val = np.array([e in val_set for e in episode])
    tr = np.isin(episode, split["train_episodes"])
    a_mean, a_std = actions[tr].mean(0), actions[tr].std(0)

    model, _ = E.load(args.w1.parent, P.DEV)
    ds = LeRobotDataset(f"local/{args.data.name}", root=args.data, delta_timestamps={c: [0.0] for c in P.CAMS.values()})

    class Frames(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            it = ds[i]
            return {v: ((one_frame(it[c]) - P.MEAN[0]) / P.STD[0]) for v, c in P.CAMS.items()}

    z = np.zeros((n, 192), np.float32)
    t0 = time.time()
    dl = torch.utils.data.DataLoader(Frames(), batch_size=args.bs, num_workers=4, shuffle=False)
    k0 = 0
    with torch.no_grad():
        for k, b in enumerate(dl):
            _, _, _, zz = P.encode_stages(model, {v: b[v] for v in P.CAMS})
            z[k0:k0 + len(zz)] = zz.float().cpu().numpy()
            k0 += len(zz)
            if k % 50 == 0:
                print(f"encode {k}/{len(dl)} {time.time() - t0:.0f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "latents.npz", z=z, actions=actions, episode=episode, frame=frame, is_val=is_val,
                        a_mean=a_mean, a_std=a_std)

    anchors = []
    for e in split["val_episodes"]:
        s, a, _ = tab[e]
        anchors += [s + t for t in range(C - 1, len(a) - max(H), STRIDE)]
    anchors = np.array(anchors)
    K = max(H)
    pred = np.zeros((len(anchors), len(H), 192), np.float32)
    with torch.no_grad():
        for i in range(0, len(anchors), 64):
            idx = anchors[i:i + 64]
            ctx = torch.tensor(np.stack([z[t - C + 1:t + 1] for t in idx]), device=P.DEV)
            act = np.stack([(actions[t - C + 1:t + K + 1] - a_mean) / a_std for t in idx]).astype(np.float32)
            emb = model.action_encoder(torch.tensor(act, device=P.DEV))
            roll = E.rollout(model, ctx, emb, C, K)  # roll[:, j] = z_{t+1+j}
            pred[i:i + len(idx)] = roll[:, [h - 1 for h in H]].float().cpu().numpy()
    true = np.stack([z[anchors + h] for h in H], 1)
    np.savez_compressed(args.out / "lowlevel.npz", anchors=anchors, pred=pred, true=true, H=np.array(H))
    print(f"done: {n} frames, {len(anchors)} val anchors, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
