"""G13b's boundary study re-run through vla.cpp's SmolVLA with prefix overwrite (no PyTorch, no GPU autograd).

Same checkpoint as G13b (`smolvla-so101-bowls-020000.gguf` = ravediamond/smolvla_2026-09-24_14-34-40 @ c1d76f6),
same 20 episodes of ravediamond/so101_bowls_all, same boundaries (every s = 25 frames), delays and metric
(`boundary_metrics`, copied from scripts/run_g13b_rtc_offline.py so this runs without torch). Methods: naive (no
prefix) and overwrite (vla.cpp's `prev_actions` = the previous chunk's rows s.. in robot units, `inference_delay` = d).
Noise is paired per boundary (numpy, not G13b's torch generator, so chunks differ from G13b's; the comparison is
within this run). Also checks that the first d rows of an overwrite chunk equal the leftover (prefix_err ~ 0) and
times both methods.

  VLA_LIBRARY=~/workspace/git/vla.cpp/build-orin/libvla.so PYTHONPATH=~/workspace/git/vla.cpp/bindings/python \
  python scripts/check_vlacpp_prefix_overwrite.py --gguf ~/vla-models/smolvla-so101-bowls-020000.gguf \
      --episodes-from shared/gate/data/2026-10-19_g13b_offline/g13b_summary.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so101_harness.episodes import TOP, WRIST, EpisodeDataset, download  # noqa: E402

ARM = slice(0, 5)
IMAGE_KEYS = (TOP, WRIST)  # the policy's input_features order


def boundary_metrics(a_prev: np.ndarray, a_new: np.ndarray, s: int, d: int, recorded: np.ndarray) -> dict[str, float]:
    """Copied from scripts/run_g13b_rtc_offline.py (same definitions)."""
    P, N = a_prev[:, ARM], a_new[:, ARM]
    steps = np.linalg.norm(np.diff(P[: s + d], axis=0), axis=1)
    acc = np.linalg.norm(np.diff(P[: s + d], n=2, axis=0), axis=1)
    med_step = float(np.median(steps)) or 1e-6
    med_acc = float(np.median(acc)) or 1e-6
    seq = np.vstack([P[s + d - 2: s + d], N[d: d + 2]])
    acc_switch = np.linalg.norm(np.diff(seq, n=2, axis=0), axis=1).max()
    rec = recorded[d: d + s, ARM] if len(recorded) >= d + s else recorded[d:, ARM]
    return {
        "prefix_err": float(np.abs(N[:d] - P[s: s + d]).mean()) if d else 0.0,
        "jump_ratio": float(np.linalg.norm(N[d] - P[s + d - 1]) / med_step),
        "accel_ratio": float(acc_switch / med_acc),
        "drift": float(np.abs(N[d: d + len(rec)] - rec).mean()) if len(rec) else float("nan"),
    }


def resize_with_pad(img: np.ndarray, side: int) -> np.ndarray:
    """HWC uint8 -> HWC float32 [0, 1], aspect kept, padding on the top and left (as lerobot.vla_cpp's client)."""
    import cv2

    h, w = img.shape[:2]
    ratio = max(w / side, h / side)
    rh, rw = int(h / ratio), int(w / ratio)
    out = np.zeros((side, side, 3), np.float32)
    out[side - rh:, side - rw:] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--repo-id", default="ravediamond/so101_bowls_all")
    ap.add_argument("--episodes-from", type=Path, required=True, help="g13b_summary.json (its meta.episodes)")
    ap.add_argument("--s", type=int, default=25)
    ap.add_argument("--delays", default="8,12,20")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolVLM2-500M-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import vla_cpp
    from transformers import AutoTokenizer

    meta = json.loads(args.episodes_from.read_text())["meta"]
    delays = [int(x) for x in args.delays.split(",")]
    s = args.s
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = EpisodeDataset(download(args.repo_id, with_videos=True))
    model = vla_cpp.load(args.gguf)
    H, A = int(model.config.n_suffix), int(model.config.max_action_dim)
    rng_base = np.random.default_rng(args.seed)

    def predict(obs, noise, prev=None, d=0):
        task = obs["task"] if obs["task"].endswith("\n") else obs["task"] + "\n"
        ids = tok(task, padding=False, truncation=True, max_length=48)["input_ids"]
        state = np.zeros(int(model.config.max_state_dim), np.float32)
        state[: len(obs["state"])] = obs["state"]
        t0 = time.perf_counter()
        out = model.predict([obs[k] for k in IMAGE_KEYS], ids, state=state, noise=noise,
                            pixel_format=vla_cpp.PIXEL_F32_RGB_01, prev_actions=prev, inference_delay=d)
        return np.asarray(out, np.float64), (time.perf_counter() - t0) * 1e3

    rows: list[dict[str, Any]] = []
    for e, ep in enumerate(meta["episodes"]):
        states, actions, task = ds.states(ep), ds.actions(ep), ds.task(ep)
        cams = {k: {i: resize_with_pad(im, 512) for i, im in ds.iter_frames(ep, k, stride=s)} for k in IMAGE_KEYS}
        T = len(states)
        obs = {i: {"state": states[i], "task": task, **{k: cams[k][i] for k in IMAGE_KEYS}}
               for i in range(0, T, s) if all(i in cams[k] for k in IMAGE_KEYS)}
        T = min(T, max(obs) + 1)
        for t in range(s, T - H, s):
            rng = np.random.default_rng([args.seed, e, t])
            prev, _ = predict(obs[t - s], rng.standard_normal(H * A).astype(np.float32))
            noise = rng.standard_normal(H * A).astype(np.float32)
            naive, ms_naive = predict(obs[t], noise)
            for d in delays:
                rows.append({"episode": ep, "t": t, "d": d, "method": "naive", "ms": ms_naive,
                             **boundary_metrics(prev, naive, s, d, actions[t:])})
                new, ms = predict(obs[t], noise, prev[s:], d)
                rows.append({"episode": ep, "t": t, "d": d, "method": "overwrite", "ms": ms,
                             **boundary_metrics(prev, new, s, d, actions[t:])})
        print(f"episode {ep} ({e + 1}/{len(meta['episodes'])}): {sum(r['episode'] == ep for r in rows)} rows", flush=True)

    summary: dict[str, Any] = {}
    for m in ("naive", "overwrite"):
        summary[m] = {}
        for d in delays:
            sel = [r for r in rows if r["method"] == m and r["d"] == d]
            summary[m][str(d)] = {k: round(float(np.nanmedian([r[k] for r in sel])), 4)
                                  for k in ("prefix_err", "jump_ratio", "accel_ratio", "drift", "ms")}
            summary[m][str(d)]["boundaries"] = len(sel)
    cut = {str(d): round(1 - summary["overwrite"][str(d)]["jump_ratio"] / summary["naive"][str(d)]["jump_ratio"], 3)
           for d in delays}
    max_prefix_err = max(r["prefix_err"] for r in rows if r["method"] == "overwrite")
    res = {"meta": {"gguf": args.gguf, "source": args.repo_id, "episodes": meta["episodes"], "s": s, "delays": delays,
                    "seed": args.seed}, "summary": summary, "jump_cut_vs_naive": cut,
           "overwrite_max_prefix_err_deg": round(max_prefix_err, 5)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "vlacpp_overwrite_summary.json").write_text(json.dumps(res, indent=2))
    with open(args.out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
