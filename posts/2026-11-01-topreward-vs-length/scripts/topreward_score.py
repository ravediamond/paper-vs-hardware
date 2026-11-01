#!/usr/bin/env python3
"""TOPReward (2602.19313) failure scores on LeRobot v3.0 episodes — TOPREWARD_PROTOCOL.md (frozen before any score).

    python topreward_score.py --root DATASET --view top --out ab.jsonl [--armnet] [--labels CSV] [--model DIR] [--limit N]

Per episode, one decode of the camera segment at FPS_S, frames resized to SHORT px; then
  * K uniformly spaced prefixes (t_1 = first frame, t_K = last): r+ = log p(" True"), r- = log p(" False") after the
    paper's prompt, scored directly (no chat template, as the paper does for open models);
  * the full episode: y+ / y- = log p(" Yes") / log p(" No") after the Yes/No prompt;
  * the episode truncated at CAP_FRAC x the task's longest episode (LENGTH_PROTOCOL.md's 50 % analysis): Yes/No again.
Each query sees at most MAXF frames (uniform, last kept). Appends one JSON line per episode, skips episodes already
in --out (restartable).
"""
import argparse, glob, json, os, time

import av
import numpy as np
import pandas as pd
import torch

K, FPS_S, SHORT, MAXF, CAP_FRAC = 10, 2.0, 224, 16, 0.5
P_TRUE = ("The above video shows a robot manipulation trajectory that completes the following task: {task}. "
          "Decide whether the above statement is True or not. The answer is:")
P_YN = "Did the robot in the above video complete the following task: {task}? Answer Yes or No. The answer is:"
VID = "<|vision_start|><|video_pad|><|vision_end|>"


def episodes(root):
    e = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))])
    return e.sort_values("episode_index").reset_index(drop=True)


def decode(path, t0, t1, fps_src):
    """Frames at FPS_S over [t0, t1), last frame included, shortest edge SHORT; uint8 (T, H, W, 3), times from t0."""
    last = t1 - 1.0 / fps_src
    ts = list(np.arange(t0, last, 1.0 / FPS_S))
    if not ts or last - ts[-1] > 1e-6:
        ts.append(last)
    out, i = [], 0
    with av.open(path) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"
        c.seek(int(max(t0 - 0.5, 0) / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            if i >= len(ts):
                break
            t = float(fr.pts * s.time_base)
            if round(t * fps_src) >= round(ts[i] * fps_src):
                k = SHORT / min(fr.width, fr.height)
                fr = fr.reformat(width=round(fr.width * k / 2) * 2, height=round(fr.height * k / 2) * 2)
                img = fr.to_ndarray(format="rgb24")
                while i < len(ts) and round(t * fps_src) >= round(ts[i] * fps_src):
                    out.append(img); i += 1
    if len(out) != len(ts):
        raise RuntimeError(f"decoded {len(out)} of {len(ts)} frames from {path}")
    return np.stack(out), np.array(ts) - t0


def sub(frames):
    if len(frames) <= MAXF:
        idx = np.arange(len(frames))
    else:
        idx = np.unique(np.round(np.linspace(0, len(frames) - 1, MAXF)).astype(int))
    if len(idx) % 2:  # Qwen3-VL merges frames in temporal pairs: repeat the first frame to keep the last one
        idx = np.r_[idx[:1], idx]
    return frames[idx]


class Scorer:
    def __init__(self, model):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model)
        self.m = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda").eval()
        tok = self.proc.tokenizer
        self.ids = {}
        for w in (" True", " False", " Yes", " No"):
            ids = tok(w, add_special_tokens=False).input_ids
            assert len(ids) == 1, (w, ids)
            self.ids[w] = ids[0]

    @torch.no_grad()
    def logp(self, frames, text, words):
        v = sub(frames)
        inp = self.proc(text=[VID + " " + text], videos=[v], return_tensors="pt", do_sample_frames=False,
                        video_metadata=[{"fps": FPS_S, "total_num_frames": len(v), "frames_indices": list(range(len(v)))}])
        inp = {k: x.to("cuda") for k, x in inp.items()}
        lp = torch.log_softmax(self.m(**inp).logits[0, -1].float(), -1)
        return [float(lp[self.ids[w]]) for w in words]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--view", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("TOPR_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--armnet", action="store_true", help="policy rollouts only, success / failure / suboptimal kept")
    ap.add_argument("--labels", help="CSV with episode_index,outcome (own rollouts)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    info = json.load(open(f"{a.root}/meta/info.json")); fps = info["fps"]
    cam = next(k for k in info["features"] if k.startswith("observation.images.") and k.endswith(a.view))
    e = episodes(a.root)
    e["task_s"] = e["tasks"].apply(lambda t: str(list(t)[0]) if len(t) else "")
    if a.armnet:
        e = e[e.policy_type != "teleoperated"].copy()
        T = e[e.success_class.isin(["successful", "failure"])].groupby("task_s").length.max()  # = length_analysis.py
    else:
        T = e.groupby("task_s").length.max()
    if a.labels:
        lab = pd.read_csv(a.labels)[["episode_index", "outcome"]]
        e = e.merge(lab, on="episode_index", how="left")
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["episode_index"] for l in open(a.out)}
    sc = Scorer(a.model); n = 0; t_start = time.time()
    with open(a.out, "a") as f:
        for _, r in e.iterrows():
            if int(r.episode_index) in done:
                continue
            if a.limit and n >= a.limit:
                break
            rec = {"episode_index": int(r.episode_index), "task": r.task_s, "length": int(r.length), "view": a.view}
            for k in ("policy_type", "success_class", "outcome"):
                if k in r and not (isinstance(r[k], float) and np.isnan(r[k])):
                    rec[k] = r[k].item() if hasattr(r[k], "item") else r[k]
            try:
                path = f"{a.root}/" + info["video_path"].format(video_key=cam, chunk_index=int(r[f"videos/{cam}/chunk_index"]),
                                                                  file_index=int(r[f"videos/{cam}/file_index"]))
                fr, ts = decode(path, float(r[f"videos/{cam}/from_timestamp"]), float(r[f"videos/{cam}/to_timestamp"]), fps)
                ends = np.unique(np.round(np.linspace(0, len(fr) - 1, K)).astype(int))
                rec["prefix_end_s"] = [float(ts[j]) for j in ends]
                rec["r_true"], rec["r_false"] = zip(*[self_lp for self_lp in
                                                      (sc.logp(fr[: j + 1], P_TRUE.format(task=r.task_s), (" True", " False")) for j in ends)])
                rec["yn_full"] = sc.logp(fr, P_YN.format(task=r.task_s), (" Yes", " No"))
                cap_s = CAP_FRAC * T[r.task_s] / fps
                m = int(np.searchsorted(ts, cap_s, side="right"))
                rec["cap_s"] = float(cap_s); rec["truncated"] = bool(m < len(fr))
                rec["yn_cap"] = sc.logp(fr[: max(m, 1)], P_YN.format(task=r.task_s), (" Yes", " No")) if m < len(fr) else rec["yn_full"]
                rec["n_frames"] = int(len(fr))
            except Exception as ex:  # recorded, never silently dropped
                rec["error"] = repr(ex)[:300]
            f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
            if n % 50 == 0:
                print(f"{n} episodes, {(time.time() - t_start) / n:.2f} s/episode", flush=True)
    print(f"done: {n} episodes in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
