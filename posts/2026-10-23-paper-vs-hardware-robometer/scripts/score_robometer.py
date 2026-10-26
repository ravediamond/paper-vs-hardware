#!/usr/bin/env python3
"""Score LeRobot v3.0 episodes with Robometer-4B (G9a-AB protocol, robometer/G9A_AB_PROTOCOL.md).

Runs inside the robometer repo's env (uv sync) on a GPU box. Per episode: frames of one camera sampled at 1 fps over
the episode's segment of the concatenated video (last frame always included), the episode's task string verbatim,
one forward pass; writes one JSON line per episode (per-frame progress and success probability) to --out, appending
and skipping episodes already there (restartable).

  python score_robometer.py --root DATASET_DIR --view top --out scores_top.jsonl [--only-rollouts] [--limit N]
      [--episodes FILE] [--max-frames N] [--fixed-frames N] [--short-edge PX]

--fixed-frames N (G9b / G9a-own input B): exactly N frames linspace over the episode, first and last included, instead
of 1 fps. --short-edge PX: resize every frame so its shortest edge is PX (G9b trains on 240 px RBM videos).

G9a-AB re-score (2026-10-16): the first run ran out of CUDA memory and never recovered (462 episodes lost). Now: after
an OOM the batch is dropped, memory freed and the episode retried at full length, then once more uniformly subsampled
to --oom-frames (last frame kept); episodes longer than --max-frames are uniformly subsampled up front. Every record
says how many frames were decoded / scored and how many attempts it took; episodes that still fail keep "error".
Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before starting (remote script does).
"""
import argparse, gc, glob, json, os, queue, threading, time, traceback

import av
import numpy as np
import pandas as pd
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator

# local snapshot of robometer/Robometer-4B @ beef63bc914c5c189329d49c6d712d96d632aa34 (remote_g9a_ab.sh downloads it)
MODEL = os.environ.get("RBM_MODEL", "robometer/Robometer-4B")


def episodes(root):
    e = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))])
    return e.sort_values("episode_index").reset_index(drop=True)


def sample_times(t0, t1, fps):
    """1 fps over [t0, t1), plus the episode's last frame."""
    last = t1 - 1.0 / fps
    ts = list(np.arange(t0, last, 1.0))
    if not ts or last - ts[-1] > 1e-6:
        ts.append(last)
    return ts


def subsample(frames, n):
    """Uniformly subsample to n frames, first and last kept."""
    if n <= 0 or len(frames) <= n:
        return frames
    return frames[np.unique(np.round(np.linspace(0, len(frames) - 1, n)).astype(int))]


def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def fixed_times(t0, t1, fps, n):
    """Exactly n timestamps linspace over [t0, last frame], first and last included."""
    return list(np.linspace(t0, t1 - 1.0 / fps, n))


def decode_at(path, times, fps, short_edge=0):
    """Decode the frames nearest to the given timestamps (seconds); returns uint8 (T, H, W, 3)."""
    out = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for t in times:
            target = round(t * fps)
            c.seek(int(max(t - 0.5, 0) / s.time_base), stream=s, backward=True)
            got = None
            for fr in c.decode(s):
                k = round(float(fr.pts * s.time_base) * fps)
                if k >= target:
                    if short_edge:
                        k = short_edge / min(fr.width, fr.height)
                        fr = fr.reformat(width=round(fr.width * k / 2) * 2, height=round(fr.height * k / 2) * 2)
                    got = fr.to_ndarray(format="rgb24")
                    break
            if got is None:
                raise RuntimeError(f"frame at {t:.3f}s not found in {path}")
            out.append(got)
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-rollouts", action="store_true", help="skip policy_type == teleoperated")
    ap.add_argument("--skip-rollouts", action="store_true", help="only policy_type == teleoperated")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--episodes", help="file with one episode_index per line: score only these")
    ap.add_argument("--max-frames", type=int, default=0, help="uniformly subsample longer episodes (0 = no cap)")
    ap.add_argument("--fixed-frames", type=int, default=0, help="exactly N frames linspace over the episode (0 = 1 fps)")
    ap.add_argument("--short-edge", type=int, default=0, help="resize frames to this shortest edge (0 = native)")
    ap.add_argument("--oom-frames", type=int, default=40, help="frames for the last retry after two OOMs")
    a = ap.parse_args()

    info = json.load(open(f"{a.root}/meta/info.json"))
    fps = info["fps"]
    cam = f"observation.images.{a.view}"
    tasks = pd.read_parquet(f"{a.root}/meta/tasks.parquet")
    e = episodes(a.root)
    if "policy_type" in e.columns and a.only_rollouts:
        e = e[e.policy_type != "teleoperated"]
    if "policy_type" in e.columns and a.skip_rollouts:
        e = e[e.policy_type == "teleoperated"]
    if a.episodes:
        e = e[e.episode_index.isin({int(l) for l in open(a.episodes) if l.strip()})]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["episode_index"] for l in open(a.out) if l.strip()}
    e = e[~e.episode_index.isin(done)]
    if a.limit:
        e = e.head(a.limit)
    print(f"{len(e)} episodes to score ({len(done)} already done), view {a.view}", flush=True)

    dev = torch.device("cuda")
    cfg, tok, proc, model = load_model_from_hf(model_path=MODEL, device=dev)
    model.eval()
    collate = setup_batch_collator(proc, tok, cfg, is_eval=True)
    lc = getattr(cfg, "loss", None)
    def forward(frames, task, idx):
        traj = Trajectory(frames=frames, frames_shape=tuple(frames.shape), task=task, id=str(idx),
                          metadata={"subsequence_length": int(frames.shape[0])}, video_embeddings=None)
        batch = collate([ProgressSample(trajectory=traj, sample_type="progress")])
        inp = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch["progress_inputs"].items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # model is bf16, inputs float32
            res = compute_batch_outputs(model, tok, inp, sample_type="progress", is_discrete_mode=discrete, num_bins=bins)
        prog = np.asarray(res.get("progress_pred", [[]])[0], dtype=float)
        succ = np.asarray((res.get("outputs_success") or {}).get("success_probs", [[]])[0], dtype=float)
        return prog, succ  # batch / inp / res go out of scope here, so an OOM leaves no GPU tensors referenced

    discrete = getattr(lc, "progress_loss_type", "l2").lower() == "discrete" if lc else False
    bins = getattr(lc, "progress_discrete_bins", None) or getattr(cfg.model, "progress_discrete_bins", 10)

    q = queue.Queue(maxsize=8)

    def producer():
        for _, r in e.iterrows():
            try:
                path = f"{a.root}/" + info["video_path"].format(
                    video_key=cam, chunk_index=int(r[f"videos/{cam}/chunk_index"]),
                    file_index=int(r[f"videos/{cam}/file_index"]))
                t0, t1 = float(r[f"videos/{cam}/from_timestamp"]), float(r[f"videos/{cam}/to_timestamp"])
                ts = fixed_times(t0, t1, fps, a.fixed_frames) if a.fixed_frames else sample_times(t0, t1, fps)
                q.put((r, decode_at(path, ts, fps, a.short_edge), None))
            except Exception as ex:  # logged, the episode is recorded with an error
                q.put((r, None, repr(ex)))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t_start = time.time(); n = 0; n_fail = 0
    with open(a.out, "a") as f:
        while (item := q.get()) is not None:
            r, frames, err = item
            task = r["tasks"][0] if isinstance(r["tasks"], (list, np.ndarray)) else str(r["tasks"])
            rec = {"episode_index": int(r["episode_index"]), "view": a.view, "task": task, "length": int(r["length"])}
            for k in ("policy_type", "success_class", "success", "policy_repo_id"):
                if k in r:
                    rec[k] = r[k].item() if hasattr(r[k], "item") else r[k]
            if err is None:
                rec["n_frames_decoded"] = int(frames.shape[0])
                frames = subsample(frames, a.max_frames)
                plan = [frames, frames, subsample(frames, a.oom_frames)]  # full, full after freeing memory, fewer frames
                for attempt, fr in enumerate(plan, 1):
                    try:
                        prog, succ = forward(fr, task, rec["episode_index"])
                        rec.update(n_frames=int(fr.shape[0]), frame_hw=list(fr.shape[1:3]), attempts=attempt,
                                   subsampled=bool(fr.shape[0] < rec["n_frames_decoded"]),
                                   progress=[round(x, 4) for x in prog.tolist()], success_prob=[round(x, 4) for x in succ.tolist()],
                                   p_succ=float(succ[-1]) if succ.size else None, last_progress=float(prog[-1]) if prog.size else None)
                        err = None
                        break
                    except torch.OutOfMemoryError as ex:
                        err = repr(ex)
                        print(f"OOM episode {rec['episode_index']} attempt {attempt} ({fr.shape[0]} frames), freeing memory", flush=True)
                    except Exception as ex:
                        traceback.print_exc()
                        err = repr(ex)
                        break
                    finally:
                        free_cuda()
            if err is not None:
                rec["error"] = err[:500]
                n_fail += 1
                print(f"FAILED episode {rec['episode_index']}: {err[:200]}", flush=True)
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1
            if n % 50 == 0 or n == 10:
                el = time.time() - t_start
                print(f"{n}/{len(e)} episodes, {el / n:.2f} s/episode, eta {(len(e) - n) * el / n / 60:.1f} min", flush=True)
    print(f"done: {n} episodes in {(time.time() - t_start) / 60:.1f} min, {n_fail} still failing", flush=True)


if __name__ == "__main__":
    main()
