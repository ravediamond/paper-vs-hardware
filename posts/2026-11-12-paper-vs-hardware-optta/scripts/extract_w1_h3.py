"""WS-196 H3 extraction: strict-load W1 latents of the 69 labelled rollouts, clean and synthetically darkened.
Runs inside the lerobot-jetson-wm image (W1=1, transformers 5.19.0, strict load) via projects/optta/run_h3_extract.sh.
Same decode / preprocess / encode path as scripts/run_w1_offline_screening.py (W1LiveMonitor._encode, FP16 on CUDA).
Shift (PROTOCOL section 2): on the preprocessed [0,1] frame, x -> clip(x)^1.5 * 0.7, both cameras, before ImageNet norm.
One npz per episode (resumable: finished episodes are skipped), one log line per episode, a heartbeat every 4 batches,
and a watchdog (faulthandler) that kills the process if one episode takes longer than --episode-timeout seconds."""
import argparse, faulthandler, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
GAMMA, BRIGHT = 1.5, 0.7


def darken(t):
    return t.clamp(0, 1).pow(GAMMA) * BRIGHT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", default="/home/jetson/wm_data/weights_final.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", default="", help="comma list; default all 69 labelled")
    ap.add_argument("--max-frames", type=int, default=0, help="test only: stop each episode after N frames")
    ap.add_argument("--episode-timeout", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    import torch
    from so101_harness.episodes import TOP, WRIST, EpisodeDataset, download
    from so101_harness.w1_live import W1LiveMonitor, preprocess
    a.out.mkdir(parents=True, exist_ok=True)
    ds = EpisodeDataset(download("ravediamond/so101-bowls-policy-rollouts"))
    oc = pd.read_csv(ROOT / "data/rollouts/so101-bowls-policy-rollouts_labels.csv")
    oc = oc[oc.outcome.isin(["success", "failure"])].copy()
    on = pd.read_csv(ROOT / "shared/gate/data/2026-10-23_onsets_vlm/onsets.csv").set_index("episode_index").onset_frame
    eps = [int(x) for x in a.episodes.split(",")] if a.episodes else [int(x) for x in oc.episode_index]
    print(f"{time.strftime('%H:%M:%S')} loading W1 (strict) device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    mon = W1LiveMonitor.from_checkpoint(a.w1)
    print(f"{time.strftime('%H:%M:%S')} W1 loaded dtype={mon.dtype} n={len(eps)}", flush=True)
    for n, ep in enumerate(eps):
        f = a.out / f"ep{ep:03d}.npz"
        if f.exists():
            print(f"[{n + 1}/{len(eps)}] ep {ep} done, skip", flush=True); continue
        faulthandler.dump_traceback_later(a.episode_timeout, exit=True)
        t0 = time.perf_counter()
        r = oc[oc.episode_index == ep].iloc[0]
        T = len(ds.states(ep))
        end = T if int(r.reset_start) < 0 else min(int(r.reset_start), T)
        zc = np.full((T, 192), np.nan, np.float32); zd = zc.copy()
        idx, tops, wrists = [], [], []

        def flush():
            if idx:
                zc[idx] = mon._encode(tops, wrists).float().cpu().numpy()
                zd[idx] = mon._encode([darken(x) for x in tops], [darken(x) for x in wrists]).float().cpu().numpy()
                idx.clear(), tops.clear(), wrists.clear()

        nb = 0
        for (i, top), (j, wr) in zip(ds.iter_frames(ep, TOP), ds.iter_frames(ep, WRIST)):
            if i != j:
                raise RuntimeError(f"episode {ep}: camera frames out of step ({i} vs {j})")
            idx.append(i), tops.append(preprocess(top)), wrists.append(preprocess(wr))
            if a.max_frames and i + 1 >= a.max_frames:
                break
            if len(idx) == a.batch:
                flush(); nb += 1
                if nb % 4 == 0:
                    print(f"    ep {ep} frame {i + 1}/{T} {time.perf_counter() - t0:.0f}s", flush=True)
        flush()
        faulthandler.cancel_dump_traceback_later()
        np.savez(f, z_clean=zc, z_dark=zd, actions=ds.actions(ep).astype(np.float32), end=end, T=T, outcome=str(r.outcome),
                 policy=str(r.policy), onset=int(on.get(ep, -1)))
        print(f"[{n + 1}/{len(eps)}] ep {ep} {r.outcome} T={T} end={end} nan_rows={int(np.isnan(zc).any(1).sum())} "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"{time.strftime('%H:%M:%S')} EXTRACTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
