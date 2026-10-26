"""Steps B + C: sample diversity, oracle headroom and baselines, plus the noise-scale / step sweep.

ALL NUMBERS ARE OPTIMISTIC (SEEN DURING TRAINING): the VLA was trained on every episode of the dataset.

Observations: M = 50 from 50 different episodes, 10 per phase (start, approach, grasp, transport, release; phases
from the gripper trace, see common.phase_windows), frame drawn uniformly in the phase window, seed SEED. Each needs
60 frames left in its episode (expert chunk at t and at t + 30 for the scale reference).

Per observation and setting, N = 8 chunks are sampled with seeds SEED + i_obs (same noise across settings, so a
setting only rescales or re-integrates it). The backbone runs once per observation; the head is batched over N.

Metrics (chunk distance d = mean over the 30 steps of the L2 norm over the joints; "all" 6 joints, "arm" the 5 arm
joints, "gripper" alone):
  spread         mean pairwise d between the 8 samples
  joint_std      per-joint std across samples, averaged over the chunk
  scale_ref      d(expert chunk at t, expert chunk at t + 30): how far the arm moves in one chunk
  mean_err       mean over samples of d(sample, expert)      = a random pick
  sample0_err    d(sample 0, expert)                          = the plain VLA (what runs on the robot)
  oracle_err     min over samples                             = best-of-8 with a perfect reranker
  meanchunk_err  d(mean of the 8 chunks, expert)
  consensus_err  d(medoid chunk, expert); medoid = sample with the smallest summed d to the others (no world model)
  null           8 chunks = mean chunk + Gaussian noise with the samples' per-step, per-joint std, 32 draws;
                 null_ratio = its oracle_err / mean_err: what best-of-8 gives from spread alone
Aggregates: ratios of sums over observations (oracle_ratio = sum oracle_err / sum mean_err, etc.);
gain_above_null = null_ratio - oracle_ratio; consensus_capture = (mean - consensus) / (mean - oracle).

Settings (step C): noise x1.0 / x1.5 / x2.0 with 4 Euler steps, and x1.0 with 2 and 1 steps.

    IMAGE=lerobot-jetson:rc LEROBOT_SRC=image shared/gate/jetson_run.sh --python projects/vla-jepa-world-model/prechecks/sample_diversity.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

M_PER_PHASE = 10
N = 8
NULL_DRAWS = 32
PHASES = ["start", "approach", "grasp", "transport", "release"]
SETTINGS = [  # name, noise scale, Euler steps
    ("base", 1.0, 4),
    ("noise_x1.5", 1.5, 4),
    ("noise_x2.0", 2.0, 4),
    ("steps_2", 1.0, 2),
    ("steps_1", 1.0, 1),
]
DIMS = {"all": slice(None), "arm": C.ARM, "gripper": slice(C.GRIPPER, C.GRIPPER + 1)}
OUT = C.OUT / "sample_diversity"


def select_observations(df) -> list[dict]:
    rng = np.random.default_rng(C.SEED)
    episodes = sorted(df.episode_index.unique())
    per_ep = {}
    for e in episodes:
        g = df[df.episode_index == e]
        win = C.phase_windows(np.stack(g.action.values)[:, C.GRIPPER])
        if win:
            per_ep[e] = (g, win)
    used, picks = set(), []
    for phase in PHASES:
        n = 0
        for e in rng.permutation(list(per_ep)):
            if n == M_PER_PHASE:
                break
            g, win = per_ep[e]
            if e in used or phase not in win:
                continue
            lo, hi = win[phase][0], min(win[phase][1], len(g) - 2 * C.CHUNK)
            if hi <= lo:
                continue
            t = int(rng.integers(lo, hi))
            actions = np.stack(g.action.values)
            picks.append({"phase": phase, "episode": int(e), "frame": t, "index": int(g["index"].iloc[t]),
                          "expert": actions[t : t + C.CHUNK], "expert_next": actions[t + C.CHUNK : t + 2 * C.CHUNK]})
            used.add(e)
            n += 1
        assert n == M_PER_PHASE, f"only {n} observations for phase {phase}"
    return picks


def metrics(samples: np.ndarray, expert: np.ndarray, expert_next: np.ndarray, rng) -> dict:
    out = {}
    mean_chunk = samples.mean(0)
    std = samples.std(0)
    pair = C.chunk_dist(samples[:, None], samples[None, :])
    medoid = int(pair.sum(1).argmin())
    nulls = [mean_chunk + std * rng.standard_normal(samples.shape) for _ in range(NULL_DRAWS)]
    for name, d in DIMS.items():
        errs = C.chunk_dist(samples, expert[None], d)
        null_errs = np.array([C.chunk_dist(z, expert[None], d) for z in nulls])
        iu = np.triu_indices(len(samples), 1)
        pair_d = C.chunk_dist(samples[:, None], samples[None, :], d)[iu]
        null_pair = np.mean([C.chunk_dist(z[:, None], z[None, :], d)[iu].mean() for z in nulls])
        out[name] = {
            "spread": float(pair_d.mean()),
            "null_spread": float(null_pair),
            "scale_ref": float(C.chunk_dist(expert, expert_next, d)),
            "mean_err": float(errs.mean()),
            "sample0_err": float(errs[0]),
            "oracle_err": float(errs.min()),
            "meanchunk_err": float(C.chunk_dist(mean_chunk, expert, d)),
            "consensus_err": float(errs[medoid]),
            "null_mean_err": float(null_errs.mean()),
            "null_oracle_err": float(null_errs.min(1).mean()),
        }
    out["joint_std"] = std.mean(0).tolist()
    return out


def aggregate(rows: list[dict], dims: str) -> dict:
    s = {k: float(np.sum([r[dims][k] for r in rows])) for k in rows[0][dims]}
    n = len(rows)
    oracle_ratio = s["oracle_err"] / s["mean_err"]
    null_ratio = s["null_oracle_err"] / s["null_mean_err"]
    gap = s["mean_err"] - s["oracle_err"]
    return {
        "n": n,
        **{k: v / n for k, v in s.items()},
        "oracle_ratio": oracle_ratio,
        "null_ratio": null_ratio,
        "gain_above_null": null_ratio - oracle_ratio,
        "consensus_ratio": s["consensus_err"] / s["mean_err"],
        "meanchunk_ratio": s["meanchunk_err"] / s["mean_err"],
        "sample0_ratio": s["sample0_err"] / s["mean_err"],
        "consensus_capture": (s["mean_err"] - s["consensus_err"]) / gap if gap > 0 else float("nan"),
        "meanchunk_capture": (s["mean_err"] - s["meanchunk_err"]) / gap if gap > 0 else float("nan"),
        "joint_std": np.mean([r["joint_std"] for r in rows], 0).tolist(),
    }


def table(summary: dict, setting: str, unit: str, dims: str) -> list[str]:
    head = ("| phase | n | spread | scale_ref | mean_err | sample0 | oracle | meanchunk | consensus | "
            "oracle/mean | null | gain>null | consensus capture |")
    lines = [head, "|" + "---|" * 13]
    for phase in [*PHASES, "all"]:
        a = summary[setting][unit][dims][phase]
        lines.append(
            f"| {phase} | {a['n']} | {a['spread']:.3f} | {a['scale_ref']:.3f} | {a['mean_err']:.3f} | "
            f"{a['sample0_err']:.3f} | {a['oracle_err']:.3f} | {a['meanchunk_err']:.3f} | {a['consensus_err']:.3f} | "
            f"**{a['oracle_ratio']:.2f}** | {a['null_ratio']:.2f} | {a['gain_above_null']:+.2f} | "
            f"{a['consensus_capture']:.0%} |"
        )
    return lines


def plots(picks, raw_samples, summary, amin, amax) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(PHASES) + 1)
    labels = [*PHASES, "all"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, dims in zip(axes, DIMS):
        a = summary["base"]["norm"][dims]
        for k, (key, lab) in enumerate([("oracle_ratio", "oracle (best of 8)"), ("null_ratio", "null (spread only)"),
                                        ("consensus_ratio", "consensus (medoid)"),
                                        ("meanchunk_ratio", "mean chunk")]):
            ax.bar(x + (k - 1.5) * 0.2, [a[p][key] for p in labels], 0.2, label=lab)
        ax.axhline(1, color="k", lw=0.8)
        ax.axhline(0.7, color="r", lw=0.8, ls="--")
        ax.set_xticks(x, labels, rotation=30)
        ax.set_title(f"error / mean_err, {dims} joints (base)")
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("ratio to a random pick (lower = more headroom)")
    fig.suptitle("Oracle gain by phase - OPTIMISTIC (seen during training)")
    fig.tight_layout()
    fig.savefig(OUT / "oracle_gain_by_phase.png", dpi=110)

    deg = C.norm_to_deg_scale(amin, amax)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [5, 1.4]})
    for k, phase in enumerate(labels):
        std = np.array(summary["base"]["norm"]["all"][phase]["joint_std"]) * deg
        axes[0].bar(np.arange(5) + (k - 2.5) * 0.14, std[:5], 0.14, label=phase)
        axes[1].bar(k, std[5], 0.7)
    axes[0].set_xticks(range(5), C.JOINTS[:5])
    axes[0].set_ylabel("std across the 8 samples (degrees)")
    axes[0].legend(fontsize=8)
    axes[1].set_xticks(range(len(labels)), labels, rotation=60)
    axes[1].set_title("gripper (0-100)")
    fig.suptitle("Sample spread per joint (base, raw units)")
    fig.tight_layout()
    fig.savefig(OUT / "spread_per_joint.png", dpi=110)

    show = [next(i for i, p in enumerate(picks) if p["phase"] == ph) for ph in ("approach", "grasp", "transport")]
    fig, axes = plt.subplots(len(show), 6, figsize=(18, 3.2 * len(show)))
    for row, i in enumerate(show):
        for j in range(6):
            ax = axes[row, j]
            for s in raw_samples[i]:
                ax.plot(s[:, j], color="tab:blue", lw=0.8, alpha=0.7)
            ax.plot(picks[i]["expert"][:, j], color="k", lw=2, label="expert")
            ax.set_title(f"{picks[i]['phase']} ep{picks[i]['episode']} t{picks[i]['frame']}: {C.JOINTS[j]}", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("8 sampled chunks (blue) vs expert (black), raw units, base setting")
    fig.tight_layout()
    fig.savefig(OUT / "overlays.png", dpi=100)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [s[0] for s in SETTINGS]
    a = [summary[n]["norm"]["all"]["all"] for n in names]
    axes[0].plot(names, [v["spread"] for v in a], "o-", label="spread")
    axes[0].plot(names, [v["mean_err"] for v in a], "o-", label="mean_err")
    axes[0].plot(names, [v["oracle_err"] for v in a], "o-", label="oracle_err")
    axes[0].plot(names, [v["consensus_err"] for v in a], "o-", label="consensus_err")
    axes[0].set_ylabel("normalized units")
    axes[0].legend(fontsize=8)
    axes[1].plot(names, [v["oracle_ratio"] for v in a], "o-", label="oracle/mean")
    axes[1].plot(names, [v["null_ratio"] for v in a], "o-", label="null")
    axes[1].axhline(0.7, color="r", ls="--", lw=0.8)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Step C sweep (all joints, all phases) - OPTIMISTIC")
    fig.tight_layout()
    fig.savefig(OUT / "sweep.png", dpi=110)


def main() -> None:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    C.seed_everything()
    OUT.mkdir(parents=True, exist_ok=True)
    df, _ = C.load_tables()
    picks = select_observations(df)
    policy, pre, post, cfg, path = C.load_policy()
    amin, amax = C.action_minmax(path)
    ds = LeRobotDataset(C.DATASET_REPO)
    rng = np.random.default_rng(C.SEED)

    results = {s[0]: {"norm": [], "raw": []} for s in SETTINGS}
    base_raw_samples = []
    for i, p in enumerate(picks):
        item = ds[p["index"]]
        assert int(item["episode_index"]) == p["episode"] and int(item["frame_index"]) == p["frame"]
        obs = {k: v for k, v in item.items() if k.startswith("observation.")} | {"task": item["task"]}
        tokens, state = C.encode(policy, pre(obs))
        exp_n = C.to_norm(p["expert"], amin, amax)
        nxt_n = C.to_norm(p["expert_next"], amin, amax)
        for name, scale, steps in SETTINGS:
            noise = C.draw_noise(policy, N, seed=C.SEED + i, scale=scale)
            with torch.no_grad():
                s_n = np.clip(C.sample_head(policy, tokens, state, noise, steps).cpu().numpy(), -1, 1)
            s_raw = C.to_raw(s_n, amin, amax)
            results[name]["norm"].append(metrics(s_n, exp_n, nxt_n, rng) | {"phase": p["phase"]})
            results[name]["raw"].append(metrics(s_raw, p["expert"], p["expert_next"], rng) | {"phase": p["phase"]})
            if name == "base":
                base_raw_samples.append(s_raw)
        print(f"{i + 1}/{len(picks)} {p['phase']:9s} ep{p['episode']:3d} t{p['frame']:3d} "
              f"oracle/mean {results['base']['norm'][-1]['all']['oracle_err'] / results['base']['norm'][-1]['all']['mean_err']:.2f}",
              flush=True)

    summary = {
        name: {unit: {dims: {ph: aggregate([r for r in rows if ph == "all" or r["phase"] == ph], dims)
                             for ph in [*PHASES, "all"]} for dims in DIMS}
               for unit, rows in results[name].items()}
        for name, *_ in SETTINGS
    }
    (OUT / "metrics.json").write_text(json.dumps(
        {"observations": [{k: v for k, v in p.items() if k not in ("expert", "expert_next")} for p in picks],
         "per_observation": results, "summary": summary}, indent=1))

    md = ["# Steps B + C: sample diversity - OPTIMISTIC (seen during training)", ""]
    for name, scale, steps in SETTINGS:
        md += [f"## {name} (noise x{scale}, {steps} Euler steps)", ""]
        for dims in DIMS:
            md += [f"### {dims} joints, normalized units", "", *table(summary, name, "norm", dims), ""]
        if name == "base":
            md += ["### all joints, raw units (degrees; gripper 0-100)", "", *table(summary, name, "raw", "all"), ""]
            md += ["### arm joints, raw units (degrees)", "", *table(summary, name, "raw", "arm"), ""]
            deg = C.norm_to_deg_scale(amin, amax)
            md += ["### per-joint std across samples, raw units", "", "| phase | " + " | ".join(C.JOINTS) + " |",
                   "|" + "---|" * 7]
            for ph in [*PHASES, "all"]:
                std = np.array(summary["base"]["norm"]["all"][ph]["joint_std"]) * deg
                md.append(f"| {ph} | " + " | ".join(f"{v:.2f}" for v in std) + " |")
            md.append("")
    (OUT / "summary.md").write_text("\n".join(md))
    plots(picks, base_raw_samples, summary, amin, amax)
    C.fix_owner(OUT)
    print("\n".join(md))


if __name__ == "__main__":
    main()
