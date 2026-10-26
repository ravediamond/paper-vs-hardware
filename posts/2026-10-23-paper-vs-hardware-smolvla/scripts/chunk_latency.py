"""
Does the latency tail of a chunked policy come from its chunk boundaries?

Reads the gate ticks files (`select_action_ms` per control tick) and, per policy and machine:
  - flags spikes (> 10x the median call) and checks whether they fall on a chunk boundary (call 0, H, 2H, ...: the
    action queue is empty, so select_action runs the model instead of popping a queued action);
  - reports the numbers that describe a chunked policy better than p50/p99: time per chunk inference, chunk
    inferences per second, and the share of wall time the sync control loop spends blocked on inference;
  - draws per-call latency over the first ticks of one trial with the chunk boundaries marked.

    uv run python shared/gate/chunk_latency.py            # table + shared/gate/docs/figures/chunk_latency.png
"""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]  # repo root
RUNS = {"M1": "shared/gate/data/2026-09-29_m1", "Orin NX": "shared/gate/data/2026-09-29_orin_nx"}
POLICIES = {"act": ("ACT", 100, "#2a78d6"), "smolvla": ("SmolVLA", 50, "#eb6834")}  # name, chunk, categorical slot
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TICKS_SHOWN = 300


def load(run: str, pol: str) -> list[dict]:
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(REPO / run / "ticks" / f"{pol}_*.json")))]


def summarize(trials: list[dict], chunk: int) -> dict:
    spikes, rest, on_boundary, blocked, wall = [], [], 0, 0.0, 0.0
    for t in trials:
        s = np.array(t["select_action_ms"])
        idx = np.where(s > 10 * np.median(s))[0]
        on_boundary += int(np.sum(idx % chunk == 0))
        spikes += s[idx].tolist()
        rest += np.delete(s, idx).tolist()
        ts = np.array(t["tick_ts"])
        wall += ts[-1] - ts[0]
        blocked += s[idx].sum() / 1e3
    return {
        "spikes": len(spikes),
        "on_boundary_pct": 100 * on_boundary / len(spikes),
        "chunk_ms": float(np.median(spikes)),
        "pop_ms": float(np.median(rest)),
        "p99_ms": float(np.percentile(spikes + rest, 99)),
        "chunks_per_s": len(spikes) / wall,
        "blocked_pct": 100 * blocked / wall,
    }


def main() -> None:
    print(f"{'policy':8s} {'machine':8s} {'chunk':>5s} {'on boundary':>11s} {'chunk inference':>15s} {'queue pop':>9s} "
          f"{'p99':>7s} {'chunks/s':>8s} {'loop blocked':>12s}")
    rows = {}
    for pol, (name, chunk, _) in POLICIES.items():
        for machine, run in RUNS.items():
            r = rows[(pol, machine)] = summarize(load(run, pol), chunk)
            print(f"{name:8s} {machine:8s} {chunk:5d} {r['on_boundary_pct']:10.0f}% {r['chunk_ms']:12.0f} ms "
                  f"{r['pop_ms']:6.2f} ms {r['p99_ms']:5.0f} ms {r['chunks_per_s']:8.2f} {r['blocked_pct']:11.0f}%")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(len(POLICIES), len(RUNS), figsize=(10, 5.6), sharex=True, facecolor=SURFACE)
    for row, (pol, (name, chunk, color)) in enumerate(POLICIES.items()):
        for col, (machine, run) in enumerate(RUNS.items()):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            s = np.array(load(run, pol)[0]["select_action_ms"])[:TICKS_SHOWN]
            for b in range(0, len(s), chunk):  # chunk boundaries: queue empty -> model runs
                ax.axvline(b, color=GRID, linewidth=1, zorder=0)
            ax.plot(np.arange(len(s)), s, color=color, linewidth=1.5, zorder=2)
            spikes = np.where(s > 10 * np.median(s))[0]
            ax.scatter(spikes, s[spikes], s=36, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.set_yscale("log")
            ax.set_ylim(0.5, 2000)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            r = rows[(pol, machine)]
            ax.set_title(f"{name} · {machine}", loc="left", fontsize=10, color=INK)
            ax.text(1.0, 1.02, f"chunk {chunk} · inference {r['chunk_ms']:.0f} ms · queue pop {r['pop_ms']:.1f} ms",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
            if col == 0:
                ax.set_ylabel("select_action (ms, log)")
            if row == len(POLICIES) - 1:
                ax.set_xlabel("control tick (first trial)")
    fig.suptitle("Latency spikes are the chunk inferences: every spike sits on a chunk boundary (vertical lines)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = REPO / "shared" / "gate" / "docs" / "figures" / "chunk_latency.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("figure:", out.relative_to(REPO))


if __name__ == "__main__":
    main()
