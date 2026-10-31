"""G15: does the stage tracker advance at the right moment? Offline, on labelled rollouts.

Stage machine as `PACEVerifier` runs it live: `grasp` is evaluated until it fires, then `place`. A query every
`--query-every` frames (default 3, i.e. 10 Hz at 30 fps); a stage fires when >= k of the last w queries pass
(w = 5, k = 3, HyMeS). Three modes for each query: proprio (gripper + shoulder), masks (G9c-3 top-camera blobs),
both (proprio OR masks).

Evidence (same predicates as agent/memory/pace_verifier.py):
- grasp, proprio: closed_on_nothing < gripper < open and shoulder_lift > lifted_shoulder;
- grasp, masks:   blue blob centre moved >= min_shift_px from its first position in the episode;
- place, proprio: gripper >= place_open;
- place, masks:   blue blob centre within max_dist_px of the pink blob centre.

Ground truth: video labels from scripts/g15_label_transitions.py (frame of each event, -1 = did not happen).
Scoring per event (grasp and place), fixed before running:
- premature: fires before t* - 5 frames, or fires when the event did not happen;
- missed: the event happened but the stage fires after t* + 30 frames (1 s) or never;
- premature rate = premature / all labelled events; missed rate = missed / events that happened.
Bar (V1.0_REVIEW.md section 6): mode `both` on the test half: premature <= 2 %, missed <= 10 %.

Calibration: thresholds fit per mode on a 50 % dev split (stratified by policy and outcome, seed 0) by grid search:
lowest missed rate among settings with premature <= 2 % on dev (if none, lowest premature). Scored once on the
other 50 %.

Usage:
  python scripts/run_g15_pace_offline.py --repo-id ravediamond/so101-bowls-policy-rollouts \
      --labels data/labels/g15_transitions.csv --outcomes data/rollouts/so101-bowls-policy-rollouts_labels.csv \
      --out shared/gate/data/<date>_g15
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so101_harness.episodes import TOP, EpisodeDataset, download  # noqa: E402

PREMATURE_SLACK = 5
LATE_TOLERANCE = 30
BAR = {"premature_max": 0.02, "missed_max": 0.10}
MODES = ("proprio", "masks", "both")

GRID = {
    "closed_on_nothing": [2.0, 5.0, 8.0, 12.0],
    "open": [25.0, 35.0, 45.0],
    "lifted_shoulder": [-90.0, -70.0, -50.0, -30.0],
    "min_shift_px": [10.0, 20.0, 35.0, 50.0],
    "place_open": [15.0, 25.0, 35.0],
    "max_dist_px": [20.0, 30.0, 42.0, 60.0],
}
PROPRIO_KEYS = ("closed_on_nothing", "open", "lifted_shoulder", "place_open")
MASK_KEYS = ("min_shift_px", "max_dist_px")


@dataclass
class Features:
    """Per query: the frame index and what the predicates need."""

    episode: int
    frames: np.ndarray       # (Q,) frame index of each query
    gripper: np.ndarray      # (Q,)
    shoulder: np.ndarray     # (Q,)
    blue: np.ndarray         # (Q, 2), NaN when the blob is not found
    pink: np.ndarray         # (Q, 2)


def extract_features(ds: EpisodeDataset, ep: int, query_every: int = 3) -> Features:
    from so101_harness.judge_g9c3 import DeterministicTopJudge

    judge = DeterministicTopJudge()
    states = ds.states(ep)
    fr, blue, pink = [], [], []
    for idx, img in ds.iter_frames(ep, TOP, stride=query_every):
        if idx >= len(states):
            break
        fr.append(idx)
        for out, color in ((blue, "blue"), (pink, "pink")):
            b = judge.extract_blob(img, color)
            out.append(b["center_xy"] if b else (np.nan, np.nan))
    fr = np.array(fr, dtype=int)
    return Features(ep, fr, states[fr, 5], states[fr, 1], np.array(blue, float), np.array(pink, float))


def evidence(f: Features, p: dict[str, float], mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(grasp_pass, place_pass) booleans per query for one mode and parameter set."""
    g_prop = (f.gripper > p["closed_on_nothing"]) & (f.gripper < p["open"]) & (f.shoulder > p["lifted_shoulder"])
    first = f.blue[~np.isnan(f.blue[:, 0])]
    if len(first):
        shift = np.hypot(*(f.blue - first[0]).T)
        g_vis = np.nan_to_num(shift, nan=0.0) >= p["min_shift_px"]
    else:
        g_vis = np.zeros(len(f.frames), bool)
    p_prop = f.gripper >= p["place_open"]
    dist = np.hypot(*(f.blue - f.pink).T)
    p_vis = np.nan_to_num(dist, nan=np.inf) <= p["max_dist_px"]
    if mode == "proprio":
        return g_prop, p_prop
    if mode == "masks":
        return g_vis, p_vis
    return g_prop | g_vis, p_prop | p_vis


def run_stage_machine(grasp: np.ndarray, place: np.ndarray, frames: np.ndarray, w: int = 5,
                      k: int = 3) -> tuple[int | None, int | None]:
    """Frame at which grasp and then place fire (k of the last w queries), as PACEVerifier does."""
    fired: dict[str, int | None] = {"grasp": None, "place": None}
    stage, window = "grasp", []
    for q in range(len(frames)):
        window.append(bool((grasp if stage == "grasp" else place)[q]))
        window = window[-w:]
        if sum(window) >= k:
            fired[stage] = int(frames[q])
            if stage == "place":
                break
            stage, window = "place", []
    return fired["grasp"], fired["place"]


def score_event(fire: int | None, truth: int) -> str:
    if truth < 0:
        return "premature" if fire is not None else "correct"
    if fire is None or fire > truth + LATE_TOLERANCE:
        return "missed"
    if fire < truth - PREMATURE_SLACK:
        return "premature"
    return "on_time"


def evaluate(feats: dict[int, Features], labels: pd.DataFrame, params: dict[str, float], mode: str,
             episodes: list[int]) -> dict[str, Any]:
    outcomes = []
    for ep in episodes:
        f = feats[ep]
        row = labels.loc[ep]
        g, p = evidence(f, params, mode)
        fg, fp = run_stage_machine(g, p, f.frames)
        outcomes.append((ep, "grasp", score_event(fg, int(row.grasp_frame)), fg, int(row.grasp_frame)))
        outcomes.append((ep, "place", score_event(fp, int(row.place_frame)), fp, int(row.place_frame)))
    events = len(outcomes)
    present = sum(1 for o in outcomes if o[4] >= 0)
    prem = sum(1 for o in outcomes if o[2] == "premature")
    miss = sum(1 for o in outcomes if o[2] == "missed")
    return {"events": events, "present": present, "premature": prem, "missed": miss,
            "premature_rate": round(prem / events, 4) if events else None,
            "missed_rate": round(miss / present, 4) if present else None,
            "per_event": [{"episode": e, "event": ev, "result": r, "fired": f, "truth": t}
                          for e, ev, r, f, t in outcomes]}


def calibrate(feats, labels, mode: str, dev: list[int], grid: dict[str, list[float]] = GRID) -> dict[str, float]:
    keys = PROPRIO_KEYS if mode == "proprio" else MASK_KEYS if mode == "masks" else PROPRIO_KEYS + MASK_KEYS
    fixed = {k: v[0] for k, v in grid.items()}
    best, best_key = None, None
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = dict(fixed, **dict(zip(keys, combo)))
        if p["closed_on_nothing"] >= p["open"]:
            continue
        r = evaluate(feats, labels, p, mode, dev)
        over = max(0.0, (r["premature_rate"] or 0) - BAR["premature_max"])
        key = (over, r["missed_rate"] if r["missed_rate"] is not None else 1.0, r["premature_rate"] or 0)
        if best_key is None or key < best_key:
            best, best_key = p, key
    assert best is not None
    return best


def split(outcomes: pd.DataFrame, episodes: list[int], seed: int = 0) -> tuple[list[int], list[int]]:
    """50/50, stratified by policy and outcome."""
    rng = np.random.default_rng(seed)
    dev, test = [], []
    sub = outcomes[outcomes.episode_index.isin(episodes)]
    extra_to_dev = True  # odd-sized groups alternate their extra episode between the halves
    for _, g in sub.groupby(["policy", "outcome"]):
        eps = list(g.episode_index.astype(int))
        rng.shuffle(eps)
        half = len(eps) // 2
        if len(eps) % 2:
            half += int(extra_to_dev)
            extra_to_dev = not extra_to_dev
        dev += eps[:half]
        test += eps[half:]
    missing = sorted(set(episodes) - set(sub.episode_index.astype(int)))
    if missing:
        raise ValueError(f"episodes without a policy/outcome row: {missing}")
    return sorted(dev), sorted(test)


def run(feats: dict[int, Features], labels: pd.DataFrame, outcomes: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    episodes = sorted(set(feats) & set(labels.index))
    dev, test = split(outcomes, episodes, seed)
    res: dict[str, Any] = {"dev_episodes": dev, "test_episodes": test, "bar": BAR,
                           "scoring": {"premature_slack_frames": PREMATURE_SLACK, "late_tolerance_frames": LATE_TOLERANCE},
                           "modes": {}}
    for mode in MODES:
        p = calibrate(feats, labels, mode, dev)
        res["modes"][mode] = {"params": p, "dev": {k: v for k, v in evaluate(feats, labels, p, mode, dev).items()
                                                    if k != "per_event"},
                              "test": evaluate(feats, labels, p, mode, test)}
    t = res["modes"]["both"]["test"]
    ok = (t["premature_rate"] is not None and t["premature_rate"] <= BAR["premature_max"]
          and t["missed_rate"] is not None and t["missed_rate"] <= BAR["missed_max"])
    res["verdict"] = "PASS" if ok else "FAIL (keep stage changes on the judge at rest only)"
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path)
    src.add_argument("--repo-id")
    ap.add_argument("--labels", type=Path, required=True, help="output of g15_label_transitions.py")
    ap.add_argument("--outcomes", type=Path, required=True, help="per-episode policy/outcome CSV")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--query-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ds = EpisodeDataset(args.root if args.root else download(args.repo_id, with_videos=True))
    labels = pd.read_csv(args.labels).set_index("episode_index")
    outcomes = pd.read_csv(args.outcomes)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "features"
    cache.mkdir(exist_ok=True)
    feats: dict[int, Features] = {}
    for ep in labels.index:
        path = cache / f"ep{int(ep):03d}_q{args.query_every}.npz"
        if path.exists():
            z = np.load(path)
            feats[int(ep)] = Features(int(ep), z["frames"], z["gripper"], z["shoulder"], z["blue"], z["pink"])
        else:
            f = extract_features(ds, int(ep), args.query_every)
            np.savez(path, **{k: v for k, v in asdict(f).items() if k != "episode"})
            feats[int(ep)] = f
    res = run(feats, labels, outcomes, args.seed)
    res["query_every"] = args.query_every
    res["source"] = args.repo_id or str(args.root)
    (args.out / "g15_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({m: {"params": v["params"], "test": {k: v["test"][k] for k in
                                                          ("premature_rate", "missed_rate", "events")}}
                      for m, v in res["modes"].items()} | {"verdict": res["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
