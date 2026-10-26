"""Fit the judge's acceptance threshold tau on our own rollouts (JevAgentHarness 2610.21412: fit per decision).

For each recorded episode, montages (top + wrist at t - 2 s, t - 1 s, t) every `--stride-s` seconds go to the VLM
judge with the telemetry the live screener would send. An episode's score is its highest FAIL confidence (FAIL
verdicts with a repair other than NONE; 0 otherwise). tau = the smallest threshold at which at most 1 in 20
successful episodes would have been interrupted (G16's false-intervention bar). Reported with it: the share of
failed episodes the judge would have caught at that tau (on our recordings, failures are not staged, so this is
only indicative).

The τ only holds for the backend, model and prompt it was fit with: run G16 with the same `--judge` and
`--judge-model`. The default backend is headless Claude Code (`claude -p`, the machine's Claude Code login, no API
key). The median latency is reported: G16 waits for the judge when the policy is more than 2 chunks ahead.

Costs: one judge call per window. Responses are cached (`<out>/judge_cache`), so reruns are free. The script prints
the number of calls before making them and stops at `--max-calls`; check it against the daily budget.

  python scripts/calibrate_judge.py --repo-id ravediamond/so101-bowls-policy-rollouts \
      --outcomes data/rollouts/so101-bowls-policy-rollouts_labels.csv --judge anthropic --out shared/gate/data/<date>_judge_tau
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from so101_harness.episodes import TOP, WRIST, EpisodeDataset, download  # noqa: E402

FALSE_INTERVENTION_MAX = 1 / 20


def windows(ds: EpisodeDataset, ep: int, stride_s: float, reset_start: int = -1):
    """Yields (frame_index, [(top, wrist) at t-2s, t-1s, t], telemetry)."""
    fps = int(round(ds.fps))
    states, actions = ds.states(ep), ds.actions(ep)
    T = len(states) if reset_start < 0 else min(reset_start, len(states))
    tops = {i: im for i, im in ds.iter_frames(ep, TOP, size=(320, 240)) if i < T}
    wrists = {i: im for i, im in ds.iter_frames(ep, WRIST, size=(320, 240)) if i < T}
    step = max(int(stride_s * fps), 1)
    for t in range(2 * fps, T, step):
        idx = [t - 2 * fps, t - fps, t]
        if not all(i in tops and i in wrists for i in idx):
            continue
        cmd, present = actions[t], states[t]
        tel = {"stage": "unknown", "gripper_deg": round(float(present[5]), 1),
               "gripper_cmd_deg": round(float(cmd[5]), 1),
               "max_joint_error_deg": round(float(np.max(np.abs(cmd[:5] - present[:5]))), 1)}
        yield t, [(tops[i], wrists[i]) for i in idx], tel


def fail_score(v: dict[str, Any]) -> float:
    return float(v["confidence"]) if v["verdict"] == "FAIL" and v["repair_primitive"] != "NONE" else 0.0


def fit_tau(success_scores: list[float], fail_scores: list[float]) -> dict[str, Any]:
    cands = sorted(set([0.0, 1.0] + list(success_scores) + list(fail_scores)))
    n_s = max(len(success_scores), 1)
    chosen = 1.01
    for tau in cands:  # smallest tau meeting the false-intervention bar
        if sum(s >= tau for s in success_scores) / n_s <= FALSE_INTERVENTION_MAX and tau > 0:
            chosen = tau
            break
    caught = sum(f >= chosen for f in fail_scores)
    return {"tau": round(chosen, 4), "false_intervention_rate": round(sum(s >= chosen for s in success_scores) / n_s, 4),
            "caught_failed": caught, "failed_episodes": len(fail_scores),
            "success_episodes": len(success_scores)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path)
    src.add_argument("--repo-id")
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--task", default="Put the blue bowl in the pink bowl.")
    ap.add_argument("--judge", choices=["claude-code", "anthropic", "openai"], default="claude-code")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-url", default=None)
    ap.add_argument("--stride-s", type=float, default=3.0)
    ap.add_argument("--max-calls", type=int, default=400)
    ap.add_argument("--exclude-policies", default="turbovla_a,turbovla_c,turbovla_trimmed,turbovla_v2",
                    help="policies not used in G16 (their failures are stalls the judge would not see live)")
    ap.add_argument("--yes", action="store_true", help="do not ask before spending calls")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from agent.supervision.vlm_judge import VLMJudge, make_backend

    ds = EpisodeDataset(args.root if args.root else download(args.repo_id, with_videos=True))
    outcomes = pd.read_csv(args.outcomes)
    outcomes = outcomes[~outcomes.policy.isin(args.exclude_policies.split(","))]
    outcomes = outcomes[outcomes.outcome.isin(["success", "failure"])]
    est = int(sum(max(0, (int(r.length) - 2 * ds.fps) // (args.stride_s * ds.fps)) + 1 for r in outcomes.itertuples()))
    print(f"{len(outcomes)} episodes, about {est} judge calls (cap {args.max_calls})")
    if not args.yes and input("go? [y/N] ").strip().lower() != "y":
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    backend = make_backend(args.judge, args.judge_model, args.judge_url)
    judge = VLMJudge(backend, args.task, tau=1.0, cache_dir=args.out / "judge_cache", max_calls=args.max_calls)
    per_ep = []
    for r in outcomes.itertuples():
        best, verdicts = 0.0, []
        try:
            for t, frames, tel in windows(ds, int(r.episode_index), args.stride_s, int(r.reset_start)):
                v = judge.judge(frames, tel)
                verdicts.append({"t": t, **{k: v[k] for k in ("verdict", "confidence", "failure_mode",
                                                             "repair_primitive", "target_blob")}})
                best = max(best, fail_score(v))
        except RuntimeError as exc:  # call budget spent
            print(f"stopped: {exc}")
            break
        per_ep.append({"episode": int(r.episode_index), "policy": r.policy, "outcome": r.outcome, "score": best,
                       "windows": verdicts})
    succ = [e["score"] for e in per_ep if e["outcome"] == "success"]
    fail = [e["score"] for e in per_ep if e["outcome"] == "failure"]
    res = fit_tau(succ, fail) | {"calls": judge.calls, "stride_s": args.stride_s, "judge": args.judge,
                                 "judge_model": args.judge_model, "parse_errors": sum(1 for x in judge.log if "parse_error" in x),
                                 "cost_usd": round(getattr(backend, "cost_usd", 0.0), 3),
                                 "latency_ms_median": float(np.median([x["latency_ms"] for x in judge.log
                                                                       if x["latency_ms"] > 50])) if judge.log else None}
    (args.out / "judge_tau.json").write_text(json.dumps(res, indent=2))
    (args.out / "judge_windows.json").write_text(json.dumps(per_ep))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
