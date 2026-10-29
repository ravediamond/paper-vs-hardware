"""WS-049 gate 0: Adaptive Action Chunking (Liang et al., arXiv 2604.04161) on our SmolVLA, offline (PROTOCOL.md).

At each decision frame p = t - 25 of the G13b grid (same 20 episodes, same policy, same seeds), draw N = 20 SmolVLA
chunks with independent noise, compute the per-timestep entropy (5 arm joints: Gaussian differential entropy of the
5x5 sample covariance; gripper: binary entropy of closed / open at g_open), the running mean E_bar_h and
h* = max(argmax_h (E_bar_{h+1} - E_bar_h), xi). Sample 0 uses G13b's `noise_prev`, so it is the chunk the fixed arm
executes. Then four arms per delay d: fixed (switch at p + 25) and AAC (switch at p + clip(h*, xi, 50 - d - 2)), each
naive and with the prefix overwrite (beta 5). Metric: G13b's jump_ratio.

Writes <out>/rows.jsonl (one row per decision: h*, the entropy curve, the phase, and the jump per arm and d) and
<out>/gate0.json (A1, A2, A3 with the bars). Resumable: episodes already in rows.jsonl are skipped.

    shared/gate/jetson_run.sh --python projects/aac-chunking/scripts/aac_gate0.py \\
        --policy-path ravediamond/smolvla_2026-09-24_14-34-40 --revision c1d76f6 --repo-id ravediamond/so101_bowls_all \\
        --device cuda --out outputs/aac_gate0 [--preflight]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_g13b_rtc_offline as G  # noqa: E402  LeRobotSmolVLA, boundary_metrics, choose_episodes

N_SAMPLES, XI, S_FIXED, BETA = 20, 2, 25, 5.0
DELAYS = (8, 12, 20)
G_OPEN = 10.0  # judge/g9c2_rule.json: gripper command below this = closed
CMD_OPEN, CMD_CLOSED = 20.0, 10.0  # W1-C1 grasp anchor (scripts/run_w1c_offline.py first_close)
GRASP_WINDOW = 15
ARM = slice(0, 5)


def first_close(cmd: np.ndarray) -> int | None:
    armed = False
    for i, v in enumerate(np.asarray(cmd, float)):
        if armed and v < CMD_CLOSED:
            return i
        if v > CMD_OPEN:
            armed = True
    return None


def release_after(cmd: np.ndarray, grasp: int) -> int | None:
    """First frame after the grasp where the command opens again (above 20, the W1-C1 arming level)."""
    nxt = np.flatnonzero(np.asarray(cmd, float)[grasp + 1:] > CMD_OPEN)
    return int(grasp + 1 + nxt[0]) if len(nxt) else None


def entropy_curve(samples: np.ndarray) -> np.ndarray:
    """samples (N, H, A) in robot units -> E_i (H,): Gaussian differential entropy of the 5 arm joints + gripper
    binary entropy (natural log)."""
    n, h, _ = samples.shape
    out = np.zeros(h)
    for i in range(h):
        x = samples[:, i, ARM]
        cov = np.cov(x, rowvar=False) + 1e-6 * np.eye(x.shape[1])
        sign, logdet = np.linalg.slogdet(cov)
        cont = 0.5 * (x.shape[1] * np.log(2 * np.pi * np.e) + logdet)
        p = float(np.mean(samples[:, i, 5] < G_OPEN))
        disc = 0.0 if p in (0.0, 1.0) else -(p * np.log(p) + (1 - p) * np.log(1 - p))
        out[i] = cont + disc
    return out


def h_star(e: np.ndarray, xi: int = XI) -> int:
    """h* = max(argmax_{h in 1..H-1} (E_bar_{h+1} - E_bar_h), xi), E_bar_h = mean of E_1..E_h."""
    ebar = np.cumsum(e) / np.arange(1, len(e) + 1)
    diffs = ebar[1:] - ebar[:-1]  # diffs[k] = E_bar_{k+2} - E_bar_{k+1}, i.e. h = k + 1
    return int(max(int(np.argmax(diffs)) + 1, xi))


def phase(p: int, grasp: int | None, release: int | None) -> str:
    if grasp is None:
        return "no_grasp"
    if abs(p - grasp) <= GRASP_WINDOW:
        return "grasp"
    if release is not None and grasp + GRASP_WINDOW < p < release:
        return "transport"
    return "other"


class Sampler(G.LeRobotSmolVLA):
    """N noise samples from one observation in one batched call (same preprocessing as `predict`)."""

    def batch_of(self, obs: dict[str, Any], n: int) -> dict[str, Any]:
        b = {k: torch.from_numpy(obs[k]).permute(2, 0, 1).float().unsqueeze(0) / 255.0 for k in self.image_keys}
        b["observation.state"] = torch.as_tensor(obs["state"], dtype=torch.float32).unsqueeze(0)
        b["task"] = [obs["task"]]
        b = self.pre(b)
        return {k: (v.repeat(n, *[1] * (v.dim() - 1)) if torch.is_tensor(v) else v * n if isinstance(v, list) else v)
                for k, v in b.items()}

    def sample_n(self, obs: dict[str, Any], noises: torch.Tensor) -> torch.Tensor:
        self._set("naive")
        with torch.no_grad():
            out = self.policy.predict_action_chunk(self.batch_of(obs, len(noises)), noise=noises.to(self.device))
        return out.detach().float().cpu()

    def to_robot_n(self, chunks: torch.Tensor) -> np.ndarray:
        return np.stack([self.to_robot(c) for c in chunks])


def noise_seed(seed: int, e: int, t: int) -> int:
    return seed * 1_000_003 + e * 10_007 + t  # G13b's scheme: noise_prev of boundary t, noise of t at t + 1


def load_frames(root: Path, ep: int, frames: set[int], image_keys, image_map) -> dict[str, Any]:
    from so101_harness.episodes import EpisodeDataset

    ds = EpisodeDataset(root)
    states, actions, task = ds.states(ep), ds.actions(ep), ds.task(ep)
    cams = {k: {i: img for i, img in ds.iter_frames(ep, image_map.get(k, k)) if i in frames} for k in image_keys}
    obs = {i: {"state": states[i], "task": task, **{k: cams[k][i] for k in image_keys}}
           for i in sorted(frames) if all(i in cams[k] for k in image_keys)}
    return {"obs": obs, "states": states, "actions": actions, "T": len(states)}


def episode_rows(m: Sampler, root: Path, ep_idx: int, e: int, seed: int, image_map) -> list[dict[str, Any]]:
    from so101_harness.episodes import EpisodeDataset

    ds = EpisodeDataset(root)
    actions = ds.actions(ep_idx)
    T, H = len(actions), m.horizon
    grasp = first_close(actions[:, 5])
    release = release_after(actions[:, 5], grasp) if grasp is not None else None
    ts = list(range(S_FIXED, T - H, S_FIXED))
    first = load_frames(root, ep_idx, {t - S_FIXED for t in ts} | set(ts), m.image_keys, image_map)
    plan = {}
    for t in ts:
        p = t - S_FIXED
        noises = torch.cat([m.noise(noise_seed(seed, e, t))] +
                           [m.noise(noise_seed(seed, e, t) * 31 + 7919 * k) for k in range(1, N_SAMPLES)])
        t0 = time.perf_counter()
        chunks = m.sample_n(first["obs"][p], noises)
        ms_n = (time.perf_counter() - t0) * 1e3
        robot = m.to_robot_n(chunks)
        e_curve = entropy_curve(robot)
        hs = h_star(e_curve)
        plan[t] = (p, chunks[0], robot[0], e_curve, hs, ms_n)
    extra = {plan[t][0] + int(np.clip(plan[t][4], XI, H - d - 2)) for t in ts for d in DELAYS}
    more = load_frames(root, ep_idx, extra - set(first["obs"]), m.image_keys, image_map)
    obs = {**first["obs"], **more["obs"]}
    rows = []
    for t in ts:
        p, prev, prev_robot, e_curve, hs, ms_n = plan[t]
        noise = m.noise(noise_seed(seed, e, t + 1))
        row = {"episode": e, "episode_index": ep_idx, "p": p, "t": t, "h_star": hs, "phase": phase(p, grasp, release),
               "grasp": grasp, "release": release, "entropy": np.round(e_curve, 4).tolist(), "ms_n20": round(ms_n, 1),
               "jump": {}}
        naive_cache: dict[int, np.ndarray] = {}
        for d in DELAYS:
            for arm, s in (("fixed", S_FIXED), ("aac", int(np.clip(hs, XI, H - d - 2)))):
                q = p + s
                if q not in obs:
                    continue
                if s not in naive_cache:
                    naive_cache[s] = m.to_robot(m.predict(obs[q], noise, "naive", 0, None))
                ow = m.to_robot(m.predict(obs[q], noise, "overwrite", d, prev[s:]))
                for meth, new in (("naive", naive_cache[s]), ("overwrite", ow)):
                    met = G.boundary_metrics(prev_robot, new, s, d, actions[q:])
                    row["jump"][f"{arm}+{meth}|{d}"] = round(met["jump_ratio"], 4)
                row.setdefault("s", {})[f"{arm}|{d}"] = s
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = lambda ph: [r["h_star"] for r in rows if r["phase"] == ph]  # noqa: E731
    grasp, transport = by("grasp"), by("transport")
    a1 = {"median_h_grasp": float(np.median(grasp)) if grasp else None, "n_grasp": len(grasp),
          "median_h_transport": float(np.median(transport)) if transport else None, "n_transport": len(transport),
          "median_h_all": float(np.median([r["h_star"] for r in rows])), "n_all": len(rows)}
    a1["met"] = bool(grasp and transport and a1["median_h_transport"] >= a1["median_h_grasp"] + 2)
    a2 = {}
    for d in DELAYS:
        for k in ("fixed+naive", "fixed+overwrite", "aac+naive", "aac+overwrite"):
            v = [r["jump"][f"{k}|{d}"] for r in rows if f"{k}|{d}" in r["jump"]]
            a2[f"{k}|{d}"] = {"median": round(float(np.median(v)), 4), "n": len(v)}
    a3 = {"aac+overwrite_d8": a2["aac+overwrite|8"]["median"], "fixed+overwrite_d8": a2["fixed+overwrite|8"]["median"]}
    a3["met"] = bool(a3["aac+overwrite_d8"] <= a3["fixed+overwrite_d8"])
    return {"A1": a1, "A2_descriptive": a2, "A3": a3,
            "ms_n20_median": float(np.median([r["ms_n20"] for r in rows]))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preflight", action="store_true", help="batched vs single sample 0, timing, then stop")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from so101_harness.episodes import download

    G.check_hook()
    path = a.policy_path
    if a.revision:
        from huggingface_hub import snapshot_download

        path = snapshot_download(a.policy_path, revision=a.revision)
    root = download(a.repo_id, with_videos=True)
    chosen = G.choose_episodes(root, a.episodes, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    m = Sampler(path, a.device, BETA)
    meta = {"policy": a.policy_path, "revision": a.revision, "data": a.repo_id, "episodes": chosen, "seed": a.seed,
            "N": N_SAMPLES, "xi": XI, "s_fixed": S_FIXED, "beta": BETA, "delays": list(DELAYS), "g_open": G_OPEN}
    if a.preflight:
        ep = load_frames(root, chosen[0], {0, 25}, m.image_keys, {})
        nz = torch.cat([m.noise(noise_seed(a.seed, 0, 25))] + [m.noise(k) for k in range(1, N_SAMPLES)])
        t0 = time.perf_counter(); batched = m.sample_n(ep["obs"][0], nz); tb = time.perf_counter() - t0  # noqa: E702
        t0 = time.perf_counter(); single = m.predict(ep["obs"][0], nz[:1], "naive", 0, None); ts = time.perf_counter() - t0  # noqa: E702,E501
        diff = float((m.to_robot(batched[0]) - m.to_robot(single)).__abs__().max())
        res = {"meta": meta, "batched_ms_n20": round(tb * 1e3, 1), "single_ms": round(ts * 1e3, 1),
               "sample0_batched_vs_single_max_deg": round(diff, 5),
               "h_star_example": h_star(entropy_curve(m.to_robot_n(batched)))}
        (a.out / "preflight.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return 0
    rows_path = a.out / "rows.jsonl"
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {r["episode"] for r in rows}
    t_start = time.monotonic()
    for e, ep_idx in enumerate(chosen):
        if e in done:
            continue
        new = episode_rows(m, root, ep_idx, e, a.seed, {})
        with rows_path.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in new)
        rows += new
        el = time.monotonic() - t_start
        print(f"episode {e + 1}/{len(chosen)}: {len(new)} decisions, {el / 60:.1f} min", flush=True)
    res = {"meta": meta, **summarize(rows)}
    (a.out / "gate0.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("A1", "A3", "ms_n20_median")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
