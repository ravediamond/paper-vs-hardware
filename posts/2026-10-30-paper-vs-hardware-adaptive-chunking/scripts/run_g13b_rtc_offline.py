"""G13b, offline half: at chunk boundaries under async execution, how much of exact RTC guidance does a
gradient-free correction recover, and at what cost? (protocol: projects/vlacpp/README.md section 7)

Open loop on recorded SO-101 episodes with our SmolVLA checkpoint. At each boundary t (every s frames): the previous
chunk A_prev comes from the observation at t - s; the new chunk from the observation at t, with the leftover
Y = A_prev[s:] (model space, as LeRobot's RTC engine passes it) and an inference delay d. Same noise for every
method at a boundary (paired). The arm does not follow the policy, so this measures boundary consistency, not success.

Methods (each a processor plugged into SmolVLA's Euler loop, LeRobot's `rtc_processor.denoise_step` hook):
  naive     RTC off
  identity  LeRobot's guided mode as shipped: correction = weighted error (no Jacobian through the denoiser)
  exact     RTC as published: correction = VJP of x1_hat = x_t - t v(x_t) with the weighted error
  overwrite prefix overwrite of the predicted clean actions (x1_hat <- W Y + (1 - W) x1_hat), no gradient
Guidance weight as LeRobot/RTC: min(beta, (1 - tau) / tau * 1 / r^2), tau = 1 - t; prefix weights with the EXP schedule
(RTC Eq. 5), beta in {5 (paper), 10 (LeRobot default)}.

Metrics per boundary (robot units, degrees, five arm joints):
  prefix_err   mean |A_new[:d] - Y[:d]|
  jump_ratio   |A_new[d] - A_prev[s + d - 1]| / median within-chunk step of A_prev
  accel_ratio  peak second difference across the switch / median within-chunk second difference
  drift        mean |A_new[d:d+s] - recorded actions[t+d : t+d+s]|
  ms           time per chunk on the device used
Decision (bar fixed in projects/vlacpp/README.md section 7):
  propose engine parts 1+2 if the best gradient-free method cuts the median jump by >= 50 % vs naive, is within 20 %
  of exact, and raises drift by <= 10 %; also propose part 3 (exact) if exact's median jump is > 20 % below the best
  gradient-free one with drift no worse, at d = 8 and 12; propose nothing if no method beats naive by 25 %.

Usage (the Orin, a Mac with MPS or a GPU; needs a LeRobot with the RTC hook, the checkpoint and the dataset):
  python scripts/run_g13b_rtc_offline.py --policy-path ravediamond/smolvla_2026-09-24_14-34-40 --revision c1d76f6 \
      --repo-id ravediamond/so101_bowls_all --device cuda --out shared/gate/data/<date>_g13b_offline --preflight
  (same without --preflight for the study; it appends rows per episode and resumes after a stop)
--preflight is G13b's timing half: ms per chunk for each method on this device, against d <= 20 steps (667 ms).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARM = slice(0, 5)
METHODS = ("naive", "identity", "exact", "overwrite")


# ---------------------------------------------------------------------------------------------------------------
# Processors (LeRobot NOISE_AT_ONE convention: x_t = t * noise + (1 - t) * actions, v = noise - actions)
# ---------------------------------------------------------------------------------------------------------------
def prefix_weights(d: int, end: int, total: int) -> torch.Tensor:
    """RTC Eq. 5 (LeRobot's EXP schedule): 1 on the first d rows, exponential decay up to `end`, 0 after."""
    w = torch.zeros(total)
    d = min(d, end)
    w[:d] = 1.0
    n = end - d
    if n > 0:
        c = torch.linspace(1, 0, n + 2)[1:-1]
        w[d:end] = c * torch.expm1(c) / (math.e - 1)
    return w


def guidance_weight(t: float, beta: float) -> float:
    tau = 1.0 - float(t)
    if tau <= 0:
        return 0.0
    if tau >= 1:
        return beta
    inv_r2 = ((1 - tau) ** 2 + tau**2) / (1 - tau) ** 2
    return min(beta, (1 - tau) / tau * inv_r2)


class GuidedProcessor:
    """Drop-in for LeRobot's `RTCProcessor.denoise_step` (same signature), one method per instance."""

    def __init__(self, method: str, beta: float = 5.0) -> None:
        if method not in ("identity", "exact", "overwrite"):
            raise ValueError(method)
        self.method, self.beta = method, beta
        self.rtc_config = type("Cfg", (), {"execution_horizon": None, "debug": False})()

    def _pad(self, prev: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        if prev.dim() == 2:
            prev = prev.unsqueeze(0)
        out = torch.zeros_like(like)
        out[:, : prev.shape[1], : prev.shape[2]] = prev.to(like)
        return out

    def denoise_step(self, x_t, prev_chunk_left_over, inference_delay, time, original_denoise_step_partial,
                     execution_horizon=None):
        t = float(time.flatten()[0]) if torch.is_tensor(time) else float(time)
        if prev_chunk_left_over is None:
            return original_denoise_step_partial(x_t)
        squeeze = x_t.dim() == 2
        x = x_t.unsqueeze(0) if squeeze else x_t
        H = x.shape[1]
        end = min(execution_horizon or prev_chunk_left_over.shape[-2], prev_chunk_left_over.shape[-2])
        W = prefix_weights(int(inference_delay), end, H).to(x).view(1, H, 1)
        Y = self._pad(prev_chunk_left_over, x)
        if self.method == "exact":
            with torch.enable_grad():
                xr = x.detach().clone().requires_grad_(True)
                v = original_denoise_step_partial(xr)
                x1 = xr - t * v
                err = ((Y - x1) * W).detach()
                (corr,) = torch.autograd.grad(x1, xr, err)
            v = v.detach()
            out = v - guidance_weight(t, self.beta) * corr
        else:
            with torch.no_grad():
                v = original_denoise_step_partial(x)
            x1 = x - t * v
            err = (Y - x1) * W
            if self.method == "identity":
                out = v - guidance_weight(t, self.beta) * err
            else:  # overwrite: x1_hat <- x1_hat + W (Y - x1_hat), v' = (x_t - x1_hat') / t
                out = v - err / max(t, 1e-6)
        return out.squeeze(0) if squeeze else out

    def track(self, *a, **k) -> None:  # LeRobot calls this when debug tracking is on
        return None

    def is_debug_enabled(self) -> bool:  # euler_integrate asks before tracking
        return False


def euler_noise_at_one(velocity: Callable[[torch.Tensor, float], torch.Tensor], noise: torch.Tensor, steps: int,
                       processor: GuidedProcessor | None = None, d: int = 0, leftover: torch.Tensor | None = None,
                       horizon_end: int | None = None) -> torch.Tensor:
    """The loop LeRobot's euler_integrate runs (t from 1 to 0, x <- x + dt v), for stand-in models in tests."""
    x, dt = noise.clone(), -1.0 / steps
    for k in range(steps):
        t = 1.0 + k * dt
        if processor is None:
            v = velocity(x, t)
        else:
            v = processor.denoise_step(x, leftover, d, torch.tensor(t), lambda xx: velocity(xx, t), horizon_end)
        x = x + dt * v
    return x


# ---------------------------------------------------------------------------------------------------------------
# Models: what the study needs from a policy
# ---------------------------------------------------------------------------------------------------------------
class ChunkModel:
    """predict(obs, noise, method, d, leftover) -> model-space chunk (H, A); to_robot(chunk) -> degrees."""

    horizon: int
    action_dim: int

    def predict(self, obs: dict[str, Any], noise: torch.Tensor, method: str, d: int,
                leftover: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError

    def to_robot(self, chunk: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    def noise(self, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randn(1, self.horizon, self.action_dim, generator=g)


class LeRobotSmolVLA(ChunkModel):
    """SmolVLA through LeRobot, with the study's processors swapped into its RTC hook."""

    def __init__(self, path: str, device: str = "cpu", beta: float = 5.0, image_keys=None) -> None:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(path).to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=path)
        cfg = self.policy.config
        cfg.rtc_config = RTCConfig(execution_horizon=cfg.chunk_size)
        self.policy.init_rtc_processor()
        self.horizon = cfg.chunk_size
        self.action_dim = cfg.max_action_dim
        self.real_dim = cfg.output_features["action"].shape[0]
        self.image_keys = image_keys or [k for k in cfg.input_features if k.startswith("observation.images.")]
        self.processors = {m: GuidedProcessor(m, beta) for m in ("identity", "exact", "overwrite")}

    def _set(self, method: str) -> None:
        enabled = method != "naive"
        self.policy.config.rtc_config.enabled = enabled
        if enabled:
            self.policy.model.rtc_processor = self.processors[method]

    def predict(self, obs, noise, method, d, leftover):
        batch = {k: torch.from_numpy(obs[k]).permute(2, 0, 1).float().unsqueeze(0) / 255.0 for k in self.image_keys}
        batch["observation.state"] = torch.as_tensor(obs["state"], dtype=torch.float32).unsqueeze(0)
        batch["task"] = [obs["task"]]
        batch = self.pre(batch)
        self._set(method)
        kw = {}
        if method != "naive" and leftover is not None:
            kw = {"inference_delay": d, "prev_chunk_left_over": leftover.to(self.device),
                  "execution_horizon": leftover.shape[-2]}
        with torch.no_grad() if method != "exact" else torch.enable_grad():
            out = self.policy.predict_action_chunk(batch, noise=noise.to(self.device), **kw)
        return out[0].detach().float().cpu()

    def to_robot(self, chunk):
        return np.asarray(self.post(chunk.unsqueeze(0).to(self.device))[0].detach().cpu().numpy(), dtype=np.float64)[
            :, : self.real_dim]


# ---------------------------------------------------------------------------------------------------------------
# Metrics and the decision
# ---------------------------------------------------------------------------------------------------------------
def boundary_metrics(a_prev: np.ndarray, a_new: np.ndarray, s: int, d: int, recorded: np.ndarray) -> dict[str, float]:
    P, N = a_prev[:, ARM], a_new[:, ARM]
    steps = np.linalg.norm(np.diff(P[: s + d], axis=0), axis=1)
    acc = np.linalg.norm(np.diff(P[: s + d], n=2, axis=0), axis=1)
    med_step = float(np.median(steps)) or 1e-6
    med_acc = float(np.median(acc)) or 1e-6
    seq = np.vstack([P[s + d - 2: s + d], N[d: d + 2]])  # last two executed of prev, first two of new
    acc_switch = np.linalg.norm(np.diff(seq, n=2, axis=0), axis=1).max()
    rec = recorded[d: d + s, ARM] if len(recorded) >= d + s else recorded[d:, ARM]
    return {
        "prefix_err": float(np.abs(N[:d] - P[s: s + d]).mean()) if d else 0.0,
        "jump_ratio": float(np.linalg.norm(N[d] - P[s + d - 1]) / med_step),
        "accel_ratio": float(acc_switch / med_acc),
        "drift": float(np.abs(N[d: d + len(rec)] - rec).mean()) if len(rec) else float("nan"),
    }


def decide(summary: dict[str, dict[str, dict[str, float]]], delays: list[int]) -> dict[str, Any]:
    """summary[method][str(d)] = {"jump_ratio": median, "drift": median, ...}; beta already chosen."""
    out: dict[str, Any] = {"per_delay": {}}
    propose_12, propose_3, any_beats = [], [], []
    for d in delays:
        k = str(d)
        naive, exact = summary["naive"][k], summary["exact"][k]
        free = min(("identity", "overwrite"), key=lambda m: summary[m][k]["jump_ratio"])
        f = summary[free][k]
        cut = 1 - f["jump_ratio"] / naive["jump_ratio"]
        within = f["jump_ratio"] <= 1.2 * exact["jump_ratio"]
        drift_ok = f["drift"] <= 1.1 * naive["drift"]
        exact_better = exact["jump_ratio"] < 0.8 * f["jump_ratio"] and exact["drift"] <= f["drift"]
        beats = max(1 - summary[m][k]["jump_ratio"] / naive["jump_ratio"] for m in ("identity", "overwrite", "exact"))
        out["per_delay"][k] = {"best_gradient_free": free, "jump_cut_vs_naive": round(cut, 3),
                               "within_20pct_of_exact": within, "drift_ok": drift_ok, "exact_20pct_better": exact_better,
                               "best_cut_any": round(beats, 3)}
        propose_12.append(cut >= 0.5 and within and drift_ok)
        if d in (8, 12):
            propose_3.append(exact_better)
        any_beats.append(beats >= 0.25)
    if not any(any_beats):
        out["decision"] = "propose nothing: no method beats naive by 25 %"
    else:
        parts = []
        if all(propose_12):
            parts.append("1+2 (engine inputs + gradient-free correction)")
        if propose_3 and all(propose_3):
            parts.append("3 (exact guidance)")
        out["decision"] = "propose " + (" and ".join(parts) if parts else "1 only (inputs); gradient-free not enough")
    return out


# ---------------------------------------------------------------------------------------------------------------
def episode_rows(model: ChunkModel, ep: dict[str, Any], e: int, s: int, delays: list[int], seed: int = 0,
                 methods=METHODS) -> list[dict[str, Any]]:
    """Rows for every boundary of one episode. ep: {"obs": per-frame obs (list, or dict keyed by frame index with
    at least the multiples of s), "T": frames (default len(obs)), "actions": (T, A) recorded}. Naive does not
    depend on d: computed once per boundary."""
    H = model.horizon
    T = int(ep.get("T", len(ep["obs"])))
    rows = []
    for t in range(s, T - H, s):
        noise_prev = model.noise(seed * 1_000_003 + e * 10_007 + t)
        prev = model.predict(ep["obs"][t - s], noise_prev, "naive", 0, None)
        prev_robot = model.to_robot(prev)
        noise = model.noise(seed * 1_000_003 + e * 10_007 + t + 1)
        naive = None
        for d in delays:
            for m in methods:
                if m == "naive" and naive is not None:
                    new, ms = naive
                else:
                    t0 = time.perf_counter()
                    new = model.predict(ep["obs"][t], noise, m, d, None if m == "naive" else prev[s:])
                    ms = (time.perf_counter() - t0) * 1e3
                    if m == "naive":
                        naive = (new, ms)
                met = boundary_metrics(prev_robot, model.to_robot(new), s, d, ep["actions"][t:])
                rows.append({"episode": e, "t": t, "d": d, "method": m, "ms": ms, **met})
    return rows


def run_study(model: ChunkModel, episodes: list[dict[str, Any]], s: int, delays: list[int], seed: int = 0,
              methods=METHODS) -> dict[str, Any]:
    rows = [r for e, ep in enumerate(episodes) for r in episode_rows(model, ep, e, s, delays, seed, methods)]
    return summarize_rows(rows, delays, methods)


def summarize_rows(rows: list[dict[str, Any]], delays: list[int], methods=METHODS) -> dict[str, Any]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for m in methods:
        summary[m] = {}
        for d in delays:
            sel = [r for r in rows if r["method"] == m and r["d"] == d]
            summary[m][str(d)] = {k: round(float(np.nanmedian([r[k] for r in sel])), 4)
                                  for k in ("prefix_err", "jump_ratio", "accel_ratio", "drift", "ms")}
            summary[m][str(d)]["boundaries"] = len(sel)
    return {"rows": rows, "summary": summary, "decision": decide(summary, delays) if set(METHODS) <= set(methods) else None}


def choose_episodes(root: Path, n: int, seed: int) -> list[int]:
    from so101_harness.episodes import EpisodeDataset

    eps = EpisodeDataset(root).episode_indices()
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(eps, size=min(n, len(eps)), replace=False).tolist())


def load_episode(root: Path, ep: int, s: int, image_keys: list[str], image_map: dict[str, str] | None = None
                 ) -> dict[str, Any]:
    """Only the frames at multiples of s are kept (the boundaries and their t - s): a full episode at 640x480 with
    two cameras is ~1.6 GB."""
    from so101_harness.episodes import EpisodeDataset

    ds = EpisodeDataset(root)
    image_map = image_map or {}
    states, actions, task = ds.states(ep), ds.actions(ep), ds.task(ep)
    cams = {k: dict(ds.iter_frames(ep, image_map.get(k, k), stride=s)) for k in image_keys}
    T = len(states)
    obs = {i: {"state": states[i], "task": task, **{k: cams[k][i] for k in image_keys}}
           for i in range(0, T, s) if all(i in cams[k] for k in image_keys)}
    T = min(T, max(obs) + 1) if obs else 0
    return {"episode_index": ep, "obs": obs, "T": T, "actions": actions[:T]}


def check_hook() -> None:
    """The study plugs into LeRobot's RTC hook in SmolVLA's denoising loop; fail early if this LeRobot lacks it."""
    import inspect

    from lerobot.policies.smolvla import modeling_smolvla as m

    src = inspect.getsource(m)
    missing = [k for k in ("rtc_processor", "denoise_step", "init_rtc_processor") if k not in src]
    if missing:
        raise SystemExit(f"this LeRobot ({m.__file__}) has no RTC hook in SmolVLA ({missing}); use a LeRobot "
                         "with policies/rtc (>= 0.4), e.g. IMAGE=lerobot-jetson:rc LEROBOT_SRC=image on the Jetson")


def preflight(model: "LeRobotSmolVLA", ep: dict[str, Any], s: int, delays: list[int], repeats: int = 5
              ) -> dict[str, Any]:
    """G13b's timing half: ms per chunk for each method on this device (median of `repeats`, after one warm-up)."""
    t = s
    prev = model.predict(ep["obs"][0], model.noise(1), "naive", 0, None)
    out: dict[str, Any] = {"device": model.device}
    for m in METHODS:
        for d in (delays if m != "naive" else delays[:1]):
            ms = []
            for i in range(repeats + 1):
                if model.device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.predict(ep["obs"][t], model.noise(2), m, d, None if m == "naive" else prev[s:])
                if model.device.startswith("cuda"):
                    torch.cuda.synchronize()
                if i:
                    ms.append((time.perf_counter() - t0) * 1e3)
            out[m if m == "naive" else f"{m}_d{d}"] = round(float(np.median(ms)), 1)
    if model.device.startswith("cuda"):
        out["max_memory_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    budget_ms = 20 / 30 * 1e3  # d <= 20 steps at 30 Hz (V1.0_REVIEW §6)
    out["exact_within_budget"] = out.get(f"exact_d{delays[0]}", 1e9) <= budget_ms
    out["budget_ms"] = round(budget_ms, 1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path)
    src.add_argument("--repo-id")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--s", type=int, default=25, help="execution horizon s (chunk boundary every s frames)")
    ap.add_argument("--delays", default="8,12,20")
    ap.add_argument("--betas", default="5,10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu, mps, cuda (the Orin: cuda)")
    ap.add_argument("--revision", default=None, help="hub revision of --policy-path (e.g. c1d76f6, the vla.cpp one)")
    ap.add_argument("--image-map", default="", help="policy_key=dataset_key,... when the camera names differ")
    ap.add_argument("--preflight", action="store_true", help="check the hook and time each method, then stop")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from so101_harness.episodes import download

    check_hook()
    policy_path = args.policy_path
    if args.revision:
        from huggingface_hub import snapshot_download

        policy_path = snapshot_download(args.policy_path, revision=args.revision)
    image_map = dict(kv.split("=", 1) for kv in args.image_map.split(",") if kv)
    root = args.root if args.root else download(args.repo_id, with_videos=True)
    delays = [int(x) for x in args.delays.split(",")]
    chosen = choose_episodes(root, args.episodes, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"policy_path": args.policy_path, "revision": args.revision, "source": args.repo_id or str(args.root),
            "s": args.s, "delays": delays, "episodes": chosen, "device": args.device, "seed": args.seed}
    betas = [float(b) for b in args.betas.split(",")]

    if args.preflight:
        model = LeRobotSmolVLA(policy_path, args.device, betas[0])
        ep = load_episode(root, chosen[0], args.s, model.image_keys, image_map)
        res = {"meta": meta, "beta": betas[0], **preflight(model, ep, args.s, delays)}
        (args.out / "g13b_preflight.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return 0

    results = {}
    for beta in betas:
        model = LeRobotSmolVLA(policy_path, args.device, beta)
        rows_path = args.out / f"rows_beta{beta:g}.jsonl"
        rows = [json.loads(l) for l in rows_path.read_text().splitlines()] if rows_path.exists() else []
        done = {r["episode"] for r in rows}
        t_start = time.monotonic()
        for e, ep_idx in enumerate(chosen):
            if e in done:
                continue
            ep = load_episode(root, ep_idx, args.s, model.image_keys, image_map)
            new = episode_rows(model, ep, e, args.s, delays, args.seed)
            with rows_path.open("a") as f:
                f.writelines(json.dumps(r) + "\n" for r in new)
            rows += new
            el = time.monotonic() - t_start
            left = len(chosen) - len({r["episode"] for r in rows})
            print(f"beta {beta:g}: episode {e + 1}/{len(chosen)} ({len(new)} rows), "
                  f"{el / 60:.1f} min so far, ~{el / max(e + 1 - len(done), 1) * left / 60:.0f} min left", flush=True)
        res = summarize_rows(rows, delays)
        results[f"beta={beta:g}"] = {"summary": res["summary"], "decision": res["decision"]}
    (args.out / "g13b_summary.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print(json.dumps({k: v["decision"] for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
