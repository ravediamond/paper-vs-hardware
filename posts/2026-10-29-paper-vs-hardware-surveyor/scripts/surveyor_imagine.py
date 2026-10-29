"""WS-050 gate 1, step 1: W1's imagined latent ŵ_r at r + S for every boundary of the 69 rollouts (PROTOCOL.md).

    python surveyor_imagine.py --w1 ~/wm_data/weights_final.pt --latents outputs/reap/latents --out <dir>

Uses the W1-A path exactly: run_w1_offline_screening's EpisodeDataset (actions), its episode ends (reset_start cut),
W1LiveMonitor.imagine_chunk with the 3-frame context and the executed actions. Writes <out>/imagine_S{S}.npz with,
per episode, boundaries r and ŵ (n_r, 192). Runs in lerobot-jetson-wm:rc with W1=1 paths.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/work")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_w1_offline_screening as S  # noqa: E402  EpisodeDataset, download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", type=Path, required=True)
    ap.add_argument("--latents", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from so101_harness.w1_live import W1LiveMonitor

    ds = S.EpisodeDataset(S.download("ravediamond/so101-bowls-policy-rollouts", with_videos=False))
    outcomes = pd.read_csv(ROOT / "data/rollouts/so101-bowls-policy-rollouts_labels.csv")
    mon = W1LiveMonitor.from_checkpoint(args.w1)
    args.out.mkdir(parents=True, exist_ok=True)
    for step in (10, 30):
        res = {}
        for r in outcomes.itertuples():
            ep = int(r.episode_index)
            z = np.load(args.latents / f"ep{ep:03d}.npy")
            a = ds.actions(ep)
            end = len(z) if int(r.reset_start) < 0 else min(int(r.reset_start), len(z))
            bs, ws = [], []
            for b in range(2, end - step, step):
                mon._ctx = torch.from_numpy(z[b - 2:b + 1]).to(mon.device, mon.dtype).unsqueeze(0)
                mon._ctx_actions = a[b - 2:b].astype(np.float32)
                mon.imagine_chunk(None, a[b:b + step], check_steps=(step,))
                bs.append(b)
                ws.append(mon._expected_latents[step].float().cpu().numpy().reshape(-1))
            res[f"b{ep}"] = np.array(bs)
            res[f"w{ep}"] = np.array(ws, np.float32)
            res[f"end{ep}"] = np.array(end)
            print(f"S={step} ep {ep} boundaries {len(bs)}", flush=True)
        np.savez_compressed(args.out / f"imagine_S{step}.npz", **res)


if __name__ == "__main__":
    main()
