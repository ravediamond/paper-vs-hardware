"""W1-A and W1-B, offline screenings of W1 (protocol: V1.0_REVIEW.md section 6, added 2026-10-19).

W1-A: W1 as a mid-rollout failure screener. W1 as it runs live (`W1LiveMonitor`): an anchor every 25 frames, context =
the anchor and the 2 frames before it with their recorded actions, imagined over the recorded actions to k = 25. Per
anchor at k in {15, 20, 25}: L2 surprise ||z - z_hat|| and cosine drift 1 - cos(z_hat_k - z_a, z_k - z_a). Episode
score = max over anchors of the mean over the three k. tau = smallest threshold with <= 1/20 successes flagged on the
50 episodes of the judge tau fit (in-sample). Bar: one signal flags >= 5/8 failures. Also the live reflex as
registered (stall 2.2 at k <= 5, missed grasp 10.0 at k >= 15).

W1-B: a latent stage tracker. Prototypes = mean latent at the labelled grasp / place frames of G15's dev half; grasp
fires when ||z - p_grasp|| < tau_g for 3 of the last 5 queries (every 3rd frame); then place fires when
||z - p_place|| < tau_p for 3 of 5 and the gripper reading has opened by >= delta since grasp fired. Fit on dev with
G15's rule, scored once on G15's test half with G15's scoring. Bar: premature <= 2 %, missed <= 10 %.

Run in the W1 image (strict load needs transformers 5.19.0):
  W1=1 shared/gate/jetson_run.sh --python scripts/run_w1_offline_screening.py \
      --w1 <W1 weights_final.pt> --out outputs/w1_screening
Latents of every frame are cached in <out>/latents (rerun is cheap).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calibrate_judge import fit_tau  # noqa: E402
from so101_harness.episodes import TOP, WRIST, EpisodeDataset, download  # noqa: E402

STRIDE, HORIZON, CHECK_K = 25, 25, (15, 20, 25)
REFLEX = {"stall_k_max": 5, "stall_l2": 2.2, "slip_k_min": 15, "slip_l2": 10.0}
LIVE_K = (3, 10, 15, 20, 25)  # the steps W1LiveMonitor checks live
G16_POLICIES_EXCLUDED = ("turbovla_a", "turbovla_c", "turbovla_trimmed", "turbovla_v2")
QUERY_EVERY, W, K = 3, 5, 3
PREMATURE_SLACK, LATE_TOLERANCE = 5, 30
BAR_A = {"failures_flagged_min": 5, "false_intervention_max": 1 / 20}
BAR_B = {"premature_max": 0.02, "missed_max": 0.10}


# ---------------------------------------------------------------------------------------------------------------
def encode_episode(mon, ds: EpisodeDataset, ep: int, batch: int = 64) -> np.ndarray:
    """(T, 192) W1 latents of every frame (NaN rows where a camera frame is missing)."""
    import torch
    from so101_harness.w1_live import preprocess

    T = len(ds.states(ep))
    z = np.full((T, 192), np.nan, np.float32)
    idx, tops, wrists = [], [], []

    def flush():
        if idx:
            z[idx] = mon._encode(tops, wrists).float().cpu().numpy()
            idx.clear(), tops.clear(), wrists.clear()

    for (i, top), (j, wrist) in zip(ds.iter_frames(ep, TOP), ds.iter_frames(ep, WRIST)):
        if i != j:
            raise RuntimeError(f"episode {ep}: camera frames out of step ({i} vs {j})")
        idx.append(i), tops.append(preprocess(top)), wrists.append(preprocess(wrist))
        if len(idx) == batch:
            flush()
    flush()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return z


def anchor_curves(mon, z: np.ndarray, actions: np.ndarray, end: int) -> list[dict[str, Any]]:
    """Per anchor: L2 surprise and cosine drift at every k in 1..HORIZON (W1LiveMonitor.imagine_chunk)."""
    import torch

    out = []
    for a in range(2, end - HORIZON, STRIDE):
        ctx = z[a - 2: a + 1]
        if np.isnan(ctx).any() or np.isnan(z[a + 1: a + HORIZON + 1]).any():
            continue
        mon._ctx = torch.from_numpy(ctx).to(mon.device, mon.dtype).unsqueeze(0)
        mon._ctx_actions = actions[a - 2: a].astype(np.float32)
        mon.imagine_chunk(None, actions[a: a + HORIZON], check_steps=tuple(range(1, HORIZON + 1)))
        l2, cos = {}, {}
        for k, zh in mon._expected_latents.items():
            zh = zh.float().cpu().numpy()
            real = z[a + k]
            l2[k] = float(np.linalg.norm(real - zh))
            u, v = zh - z[a], real - z[a]
            cos[k] = float(1 - u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
        out.append({"anchor": a, "l2": l2, "cos": cos})
    return out


def w1a(curves: dict[int, list[dict[str, Any]]], outcomes: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for r in outcomes.itertuples():
        cs = curves.get(int(r.episode_index), [])
        per = [{"anchor": c["anchor"], "l2": np.mean([c["l2"][k] for k in CHECK_K]),
                "cos": np.mean([c["cos"][k] for k in CHECK_K]),
                "reflex": any((k <= REFLEX["stall_k_max"] and c["l2"][k] > REFLEX["stall_l2"]) or
                              (k >= REFLEX["slip_k_min"] and c["l2"][k] > REFLEX["slip_l2"]) for k in LIVE_K)}
               for c in cs]
        rows.append({"episode_index": int(r.episode_index), "policy": r.policy, "outcome": r.outcome,
                     "in_g16_set": r.policy not in G16_POLICIES_EXCLUDED, "anchors": len(per),
                     "end": int(r.end), "score_l2": max((p["l2"] for p in per), default=0.0),
                     "score_cos": max((p["cos"] for p in per), default=0.0),
                     "reflex": any(p["reflex"] for p in per), "per_anchor": per})
    res: dict[str, Any] = {}
    for name, sel in (("g16_set", [x for x in rows if x["in_g16_set"]]), ("all_69", rows)):
        s = [x for x in sel if x["outcome"] == "success"]
        f = [x for x in sel if x["outcome"] == "failure"]
        block: dict[str, Any] = {"reflex_live_rule": {"success_flagged": sum(x["reflex"] for x in s), "success": len(s),
                                                      "failure_flagged": sum(x["reflex"] for x in f), "failure": len(f)}}
        for sig in ("l2", "cos"):
            fit = fit_tau([x[f"score_{sig}"] for x in s], [x[f"score_{sig}"] for x in f])
            lead = []
            for x in f:
                hit = [p["anchor"] for p in x["per_anchor"] if p[sig] >= fit["tau"]]
                if hit:
                    lead.append(x["end"] - (hit[0] + HORIZON))
            ys = [0] * len(s) + [1] * len(f)
            sc = [x[f"score_{sig}"] for x in s] + [x[f"score_{sig}"] for x in f]
            block[sig] = {**fit, "auroc": auroc(ys, sc),
                          "frames_before_end_when_caught": sorted(lead)}
        res[name] = block
    g = res["g16_set"]
    passed = any(g[sig]["caught_failed"] >= BAR_A["failures_flagged_min"] and
                 g[sig]["false_intervention_rate"] <= BAR_A["false_intervention_max"] for sig in ("l2", "cos"))
    res["verdict"] = "PASS (W1 screener goes into G16-lite's trials)" if passed else "FAIL (W1 stays a reflex only)"
    res["episodes"] = [{k: v for k, v in x.items() if k != "per_anchor"} for x in rows]
    return res


def auroc(y: list[int], s: list[float]) -> float | None:
    pos = [b for a, b in zip(y, s) if a]
    neg = [b for a, b in zip(y, s) if not a]
    if not pos or not neg:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


# ---------------------------------------------------------------------------------------------------------------
def track(zq: np.ndarray, frames: np.ndarray, grip: np.ndarray, protos: dict[str, np.ndarray], tau_g: float,
          tau_p: float, delta: float) -> tuple[int | None, int | None]:
    """Stage machine on the queried frames: returns the frames where grasp and place fired (None = never)."""
    dg = np.linalg.norm(zq - protos["grasp"], axis=1)
    dp = np.linalg.norm(zq - protos["place"], axis=1)
    votes: list[bool] = []
    g_fire = p_fire = None
    g_val = None
    for i, f in enumerate(frames):
        if g_fire is None:
            votes.append(bool(dg[i] < tau_g))
            if sum(votes[-W:]) >= K:
                g_fire, g_val, votes = int(f), grip[i], []
        else:
            votes.append(bool(dp[i] < tau_p and grip[i] - g_val >= delta))
            if sum(votes[-W:]) >= K:
                p_fire = int(f)
                break
    return g_fire, p_fire


def score_event(fired: int | None, truth: int) -> str:
    if truth < 0:
        return "premature" if fired is not None else "correct"
    if fired is not None and fired < truth - PREMATURE_SLACK:
        return "premature"
    if fired is None or fired > truth + LATE_TOLERANCE:
        return "missed"
    return "on_time"


def evaluate(eps: list[int], data: dict[int, Any], labels: pd.DataFrame, protos, tau_g, tau_p, delta,
             detail: bool = False) -> dict[str, Any]:
    res = {"events": 0, "present": 0, "premature": 0, "missed": 0}
    per = []
    for ep in eps:
        zq, frames, grip = data[ep]
        g, p = track(zq, frames, grip, protos, tau_g, tau_p, delta)
        for ev, fired in (("grasp", g), ("place", p)):
            truth = int(labels.loc[ep, f"{ev}_frame"])
            r = score_event(fired, truth)
            res["events"] += 1
            res["present"] += truth >= 0
            res["premature"] += r == "premature"
            res["missed"] += r == "missed"
            per.append({"episode": ep, "event": ev, "result": r, "fired": fired, "truth": truth})
    res["premature_rate"] = round(res["premature"] / max(res["events"], 1), 4)
    res["missed_rate"] = round(res["missed"] / max(res["present"], 1), 4)
    if detail:
        res["per_event"] = per
    return res


def w1b(latents: dict[int, np.ndarray], grips: dict[int, np.ndarray], labels: pd.DataFrame, dev: list[int],
        test: list[int], ends: dict[int, int]) -> dict[str, Any]:
    protos = {}
    for ev in ("grasp", "place"):
        zs = [latents[e][int(labels.loc[e, f"{ev}_frame"])] for e in dev if int(labels.loc[e, f"{ev}_frame"]) >= 0]
        protos[ev] = np.nanmean(np.stack(zs), axis=0)
    data = {}
    for e in dev + test:
        frames = np.arange(0, ends[e], QUERY_EVERY)
        frames = frames[~np.isnan(latents[e][frames]).any(axis=1)]
        data[e] = (latents[e][frames], frames, grips[e][frames])
    dist = {ev: np.concatenate([np.linalg.norm(data[e][0] - protos[ev], axis=1) for e in dev]) for ev in protos}
    grid_g = np.percentile(dist["grasp"], np.arange(5, 100, 5))
    grid_p = np.percentile(dist["place"], np.arange(5, 100, 5))
    best = None
    for tg, tp, dl in itertools.product(grid_g, grid_p, (2.0, 4.0, 6.0, 8.0, 12.0)):
        r = evaluate(dev, data, labels, protos, tg, tp, dl)
        ok = r["premature_rate"] <= BAR_B["premature_max"]
        key = (0 if ok else 1, r["missed_rate"] if ok else r["premature_rate"], r["missed_rate"])
        if best is None or key < best[0]:
            best = (key, (float(tg), float(tp), dl), r)
    (tg, tp, dl), dev_res = best[1], best[2]
    test_res = evaluate(test, data, labels, protos, tg, tp, dl, detail=True)
    passed = test_res["premature_rate"] <= BAR_B["premature_max"] and test_res["missed_rate"] <= BAR_B["missed_max"]
    return {"params": {"tau_grasp": round(tg, 4), "tau_place": round(tp, 4), "delta_deg": dl},
            "dev": dev_res, "test": test_res, "dev_met_premature_bar": best[0][0] == 0,
            "verdict": "PASS (joins G15-v2's fresh-data test)" if passed else "FAIL"}


# ---------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w1", required=True, help="weights_final.pt")
    ap.add_argument("--repo-id", default="ravediamond/so101-bowls-policy-rollouts")
    ap.add_argument("--outcomes", type=Path, default=ROOT / "data/rollouts/so101-bowls-policy-rollouts_labels.csv")
    ap.add_argument("--labels", type=Path, default=ROOT / "shared/gate/data/2026-10-19_g15/transition_labels.csv")
    ap.add_argument("--g15-summary", type=Path, default=ROOT / "shared/gate/data/2026-10-19_g15/g15_summary.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from so101_harness.w1_live import W1LiveMonitor

    ds = EpisodeDataset(download(args.repo_id, with_videos=True))
    outcomes = pd.read_csv(args.outcomes)
    outcomes = outcomes[outcomes.outcome.isin(["success", "failure"])].copy()
    outcomes["end"] = [len(ds.states(int(r.episode_index))) if int(r.reset_start) < 0
                       else min(int(r.reset_start), len(ds.states(int(r.episode_index)))) for r in outcomes.itertuples()]
    mon = W1LiveMonitor.from_checkpoint(args.w1)
    lat_dir = args.out / "latents"
    lat_dir.mkdir(parents=True, exist_ok=True)

    latents, curves, ms_enc = {}, {}, []
    for r in outcomes.itertuples():
        ep = int(r.episode_index)
        f = lat_dir / f"ep{ep:03d}.npy"
        if f.exists():
            latents[ep] = np.load(f)
        else:
            t0 = time.perf_counter()
            latents[ep] = encode_episode(mon, ds, ep)
            ms_enc.append((time.perf_counter() - t0) * 1e3 / len(latents[ep]))
            np.save(f, latents[ep])
        curves[ep] = anchor_curves(mon, latents[ep], ds.actions(ep), int(r.end))
        print(f"  ep {ep:3d} {r.outcome:7s} {r.policy:18s} anchors {len(curves[ep])}", flush=True)

    res_a = w1a(curves, outcomes)
    labels = pd.read_csv(args.labels).set_index("episode_index")
    split = json.loads(args.g15_summary.read_text())
    grips = {int(r.episode_index): ds.states(int(r.episode_index))[:, 5] for r in outcomes.itertuples()}
    ends = {int(r.episode_index): int(r.end) for r in outcomes.itertuples()}
    res_b = w1b(latents, grips, labels, split["dev_episodes"], split["test_episodes"], ends)

    meta = {"w1": args.w1, "source": args.repo_id, "stride": STRIDE, "horizon": HORIZON, "check_k": CHECK_K,
            "encode_ms_per_frame_batched": round(float(np.median(ms_enc)), 2) if ms_enc else None,
            "dtype": str(mon.dtype), "device": str(mon.device)}
    (args.out / "w1a_summary.json").write_text(json.dumps({"meta": meta, "bar": BAR_A, **res_a}, indent=2,
                                                          default=float))
    (args.out / "w1b_summary.json").write_text(json.dumps({"meta": meta, "bar": BAR_B, **res_b}, indent=2,
                                                          default=float))
    print(json.dumps({"W1-A": {k: res_a[k] for k in ("g16_set", "verdict")},
                      "W1-B": {k: v for k, v in res_b.items() if k != "test"} | {
                          "test": {k: v for k, v in res_b["test"].items() if k != "per_event"}}}, indent=2,
                     default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
