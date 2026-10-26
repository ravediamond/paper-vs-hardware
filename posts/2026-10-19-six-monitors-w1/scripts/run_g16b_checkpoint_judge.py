"""G16-B, offline: is a checkpoint VLM judge worth keeping? (protocol: V1.0_REVIEW.md section 6, added 2026-10-19)

One judge call per checkpoint instead of sliding windows:
- post-grasp: 1 s after the gripper command first falls below 10 deg having been above 20 deg (proprioception only,
  available live); question: is the gripper holding the blue bowl lifted off the mat?
- at rest: the last frame before reset_start (or the last frame); question: is the blue bowl inside the pink bowl?
Input: top + wrist at t - 1 s and t. Binary JSON answer (no confidence: the endpoint gives no logprobs and the
verbalised confidence saturated). An episode is flagged if either checkpoint says no; an unparseable answer does not
flag (counted as a parse error).
Bar: <= 2/42 successes flagged, >= 6/8 failures flagged (the 50 episodes of the tau fit), median latency <= 5 s.
The 25 failures of all 69 episodes are reported as a second column.

The judge is any OpenAI-compatible endpoint (key in VLM_API_KEY); `reasoning_effort` is passed through (Gemini 3.x:
"none" keeps thinking short; it cannot be switched off). Answers are cached in <out>/cache, so reruns are free.

  python scripts/run_g16b_checkpoint_judge.py --repo-id ravediamond/so101-bowls-policy-rollouts \
      --outcomes data/rollouts/so101-bowls-policy-rollouts_labels.csv --model gemini-3.8-flash \
      --url https://generativelanguage.googleapis.com/v1beta/openai --out shared/gate/data/<date>_g16b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from so101_harness.episodes import TOP, WRIST, EpisodeDataset, download  # noqa: E402

TASK = "Put the blue bowl in the pink bowl."
G16_POLICIES_EXCLUDED = ("turbovla_a", "turbovla_c", "turbovla_trimmed", "turbovla_v2")  # as calibrate_judge.py
OPEN_DEG, CLOSED_DEG = 20.0, 10.0
BAR = {"success_flagged_max": 2, "failure_flagged_min": 6, "latency_s_median_max": 5.0}

HEAD = """You supervise a SO-101 robot arm doing the task "{task}".
The image has two columns, 1 s apart (left: 1 s ago, right: now); the top row is the overhead camera, the bottom row
the wrist camera.
"""
QUESTIONS = {
    "post_grasp": "The gripper has just closed. Is the gripper now holding the BLUE bowl, lifted off the mat (not the "
                  "pink bowl, not empty, not just touching it)?",
    "at_rest": "The arm has finished its attempt and is at rest. Is the blue bowl inside the pink bowl? When the blue "
               "bowl sits inside the pink bowl it can hide the pink bowl from the overhead camera.",
}
TAIL = '\nAnswer with JSON only: {"answer": "yes" | "no", "why": "one short sentence"}'


def checkpoints(actions: np.ndarray, T: int, fps: float) -> dict[str, int]:
    """Frame index of each checkpoint; post_grasp absent if the gripper command never opens then closes."""
    g = actions[:T, 5]
    out = {"at_rest": T - 1}
    opened = np.flatnonzero(g > OPEN_DEG)
    if len(opened):
        closed = np.flatnonzero((np.arange(T) > opened[0]) & (g < CLOSED_DEG))
        if len(closed):
            out["post_grasp"] = min(int(closed[0] + round(fps)), T - 1)
    return out


def parse_answer(text: str) -> str | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        a = str(json.loads(m.group(0))["answer"]).strip().lower() if m else None
    except (ValueError, KeyError, TypeError):
        return None
    return a if a in ("yes", "no") else None


class Judge:
    def __init__(self, url: str, model: str, cache: Path, reasoning_effort: str | None, gap_s: float,
                 max_calls: int) -> None:
        self.url, self.model, self.cache = url.rstrip("/") + "/chat/completions", model, cache
        self.reasoning_effort, self.gap_s, self.max_calls = reasoning_effort, gap_s, max_calls
        self.cache.mkdir(parents=True, exist_ok=True)
        self.calls, self.latencies, self.usage = 0, [], {"prompt_tokens": 0, "total_tokens": 0}
        self._last = 0.0

    def ask(self, img_b64: str, prompt: str) -> str:
        key = self.cache / (hashlib.sha256((img_b64 + prompt + self.model).encode()).hexdigest()[:24] + ".json")
        if key.exists():
            return json.loads(key.read_text())["text"]
        if self.calls >= self.max_calls:
            raise RuntimeError(f"call budget ({self.max_calls}) spent")
        body = {"model": self.model, "max_tokens": 2048, "temperature": 0, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt}]}]}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        for attempt in range(6):
            time.sleep(max(0.0, self.gap_s - (time.time() - self._last)))
            self._last = t0 = time.time()
            req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST", headers={
                "content-type": "application/json", "authorization": f"Bearer {os.environ.get('VLM_API_KEY', '')}"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.loads(r.read().decode())
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
                time.sleep(min(120, 10 * 2 ** attempt))
                continue
            except (urllib.error.URLError, TimeoutError):
                time.sleep(min(120, 10 * 2 ** attempt))
                continue
            self.calls += 1
            self.latencies.append(time.time() - t0)
            for k in self.usage:
                self.usage[k] += int(d.get("usage", {}).get(k, 0))
            text = d["choices"][0]["message"]["content"] or ""
            key.write_text(json.dumps({"text": text, "latency_s": self.latencies[-1]}))
            return text
        raise RuntimeError("no answer after 6 attempts")


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s = [r for r in rows if r["outcome"] == "success"]
    f = [r for r in rows if r["outcome"] == "failure"]
    return {"success": len(s), "success_flagged": sum(r["flagged"] for r in s),
            "failure": len(f), "failure_flagged": sum(r["flagged"] for r in f),
            "failure_flagged_by": {cp: sum(r.get(f"{cp}_answer") == "no" for r in f) for cp in QUESTIONS}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path)
    src.add_argument("--repo-id")
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--url", required=True, help="OpenAI-compatible base URL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--gap-s", type=float, default=1.0, help="minimum seconds between calls")
    ap.add_argument("--max-calls", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from agent.supervision.vlm_judge import jpeg_b64, montage

    ds = EpisodeDataset(args.root if args.root else download(args.repo_id, with_videos=True))
    outcomes = pd.read_csv(args.outcomes)
    outcomes = outcomes[outcomes.outcome.isin(["success", "failure"])]
    judge = Judge(args.url, args.model, args.out / "cache", args.reasoning_effort, args.gap_s, args.max_calls)
    rows = []
    for r in outcomes.itertuples():
        ep = int(r.episode_index)
        actions = ds.actions(ep)
        T = len(actions) if int(r.reset_start) < 0 else min(int(r.reset_start), len(actions))
        cps = checkpoints(actions, T, ds.fps)
        need = {i for t in cps.values() for i in (max(t - int(round(ds.fps)), 0), t)}
        tops = {i: im for i, im in ds.iter_frames(ep, TOP, size=(320, 240)) if i in need}
        wrists = {i: im for i, im in ds.iter_frames(ep, WRIST, size=(320, 240)) if i in need}
        row: dict[str, Any] = {"episode_index": ep, "policy": r.policy, "outcome": r.outcome,
                               "in_g16_set": r.policy not in G16_POLICIES_EXCLUDED}
        for cp, t in cps.items():
            t0 = max(t - int(round(ds.fps)), 0)
            if not all(i in tops and i in wrists for i in (t0, t)):
                row[f"{cp}_answer"] = "no_frames"
                continue
            img = jpeg_b64(montage([(tops[t0], wrists[t0]), (tops[t], wrists[t])]))
            text = judge.ask(img, HEAD.format(task=TASK) + QUESTIONS[cp] + TAIL)
            row[f"{cp}_frame"], row[f"{cp}_answer"], row[f"{cp}_raw"] = t, parse_answer(text) or "parse_error", text
        row["flagged"] = any(row.get(f"{cp}_answer") == "no" for cp in QUESTIONS)
        rows.append(row)
        print(f"  ep {ep:3d} {r.outcome:7s} {r.policy:18s} " + "  ".join(
            f"{cp}={row.get(f'{cp}_answer', '-')}" for cp in QUESTIONS), flush=True)

    g16 = score([r for r in rows if r["in_g16_set"]])
    lat = [json.loads(p.read_text())["latency_s"] for p in (args.out / "cache").glob("*.json")]
    med = float(np.median(lat)) if lat else None
    passed = (g16["success_flagged"] <= BAR["success_flagged_max"] and g16["failure_flagged"] >= BAR["failure_flagged_min"]
              and med is not None and med <= BAR["latency_s_median_max"])
    res = {"bar": BAR, "model": args.model, "reasoning_effort": args.reasoning_effort, "g16_set": g16,
           "all_69": score(rows),
           "parse_errors": sum(r.get(f"{cp}_answer") == "parse_error" for r in rows for cp in QUESTIONS),
           "no_post_grasp_checkpoint": sum("post_grasp_answer" not in r for r in rows),
           "latency_s_median": med, "latency_s_p90": float(np.percentile(lat, 90)) if lat else None,
           "calls_this_run": judge.calls, "tokens_this_run": judge.usage,
           "verdict": "PASS (checkpoint judge worth a robot test)" if passed else "FAIL (drop the VLM judge from v1.0)"}
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "g16b_episodes.csv", index=False)
    (args.out / "g16b_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
