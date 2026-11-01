#!/usr/bin/env python3
"""WS-077, LENGTH_PROTOCOL.md (frozen 1a69f30): length as the null model on ArmnetBench, and CIs on every count.

    python length_analysis.py --meta ~/robot-data/armnetbench_meta      # (a), (b), (d); (c) only with --scores
Writes results/length_armnetbench.json and results/counts_ci.json next to this file.
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
B, SEED = 2000, 0
TS = (0.25, 0.50, 0.75)


def auroc(y, s):
    """P(score of a success > score of a failure), ties 1/2 (Mann-Whitney)."""
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = pd.Series(s).rank().values
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified(df, score, by="policy_type"):
    """Pair-weighted mean of per-stratum AUROCs (weights successes x failures)."""
    num = den = 0.0
    for _, g in df.groupby(by):
        a = auroc(g.y, g[score]); w = g.y.sum() * (~g.y).sum()
        if a is not None:
            num += a * w; den += w
    return num / den if den else None


def boot(df, fn, strat="policy_type"):
    rng = np.random.default_rng(SEED); groups = [g for _, g in df.groupby(strat)]; v = []
    for _ in range(B):
        s = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
        x = fn(s)
        if x is not None:
            v.append(x)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def boot_count(k, n):
    rng = np.random.default_rng(SEED); x = np.r_[np.ones(k), np.zeros(n - k)]
    v = [x[rng.integers(0, n, n)].mean() for _ in range(B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--meta", required=True); ap.add_argument("--scores")
    a = ap.parse_args()
    d = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{a.meta}/meta/episodes/*/*.parquet")))
    d["task_s"] = d["tasks"].apply(lambda t: str(list(t)[0]) if len(t) else "")
    d = d[d.success_class.isin(["successful", "failure"]) & (d.policy_type != "teleoperated")].copy()  # G9a-AB: policy rollouts only
    d["y"] = d.success_class == "successful"
    d["neg_len"] = -d.length.astype(float)
    T = d.groupby("task_s").length.max(); d["timeout"] = d.task_s.map(T)
    out = {"n": len(d), "n_success": int(d.y.sum()), "n_failure": int((~d.y).sum()), "timeout_source": "longest episode of the task (no timeout in the metadata)"}
    # (a)
    out["a_pooled"] = {"auroc": auroc(d.y, d.neg_len), "ci95": boot(d, lambda s: auroc(s.y, s.neg_len))}
    out["a_within_policy_stratified"] = {"auroc": stratified(d, "neg_len"), "ci95": boot(d, lambda s: stratified(s, "neg_len"))}
    out["a_per_policy"] = {p: {"n": len(g), "n_success": int(g.y.sum()), "auroc": auroc(g.y, g.neg_len),
                               "ci95": boot(g, lambda s: auroc(s.y, s.neg_len), strat="task_s")} for p, g in d.groupby("policy_type")}
    out["a_per_task"] = {t: {"n": len(g), "n_success": int(g.y.sum()), "auroc": auroc(g.y, g.neg_len)} for t, g in d.groupby("task_s")}
    cells = {}
    for (t, p), g in d.groupby(["task_s", "policy_type"]):
        if g.y.sum() >= 5 and (~g.y).sum() >= 5:
            cells[f"{t} | {p}"] = {"n": len(g), "auroc": auroc(g.y, g.neg_len)}
    out["a_cells_5x5"] = cells
    # (b)
    out["b_timeout_share"] = {"failures_hit": float((d[~d.y].length >= 0.98 * d[~d.y].timeout).mean()),
                              "successes_hit": float((d[d.y].length >= 0.98 * d[d.y].timeout).mean())}
    out["b_truncated"] = {}
    for t in TS:
        col = f"cap{t}"; d[col] = -np.minimum(d.length, t * d["timeout"])
        out["b_truncated"][str(t)] = {"pooled": auroc(d.y, d[col]), "pooled_ci95": boot(d, lambda s: auroc(s.y, s[col])),
                                      "within_policy": stratified(d, col), "within_policy_ci95": boot(d, lambda s: stratified(s, col)),
                                      "still_running_failures": float((d[~d.y].length > t * d[~d.y].timeout).mean()),
                                      "still_running_successes": float((d[d.y].length > t * d[d.y].timeout).mean())}
    # (c)
    out["c_robometer"] = "not run: per-episode scores not on the Orin (old Mac archive); re-score ~$6 on vast is the owner's call" \
        if not a.scores else "see scores file"
    # reading rules (LENGTH_PROTOCOL.md)
    out["reading"] = {"within_policy_null": out["a_within_policy_stratified"]["ci95"][0] > 0.60,
                      "more_than_timeout_t050": out["b_truncated"]["0.5"]["within_policy_ci95"][0] > 0.60}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/length_armnetbench.json").write_text(json.dumps(out, indent=1))
    # (d) counts of the memo's evidence table (journal/EXPERIMENTS.md rows)
    counts = [("G9a-own", "length AUROC — not a count", None, None),
              ("Judge τ: failures caught", "", 0, 8), ("Judge τ: successes interrupted", "", 1, 42),
              ("G16-B: failures flagged", "", 8, 8), ("G16-B: successes flagged", "", 41, 42),
              ("W1-A L2 surprise: failures caught", "", 2, 8), ("W1-A L2 surprise: false interventions", "", 2, 42),
              ("W1-A cosine drift: failures caught", "", 0, 8), ("W1-A live reflex: successes flagged", "", 42, 42),
              ("G-flora: failures flagged", "", 8, 8), ("G-flora: successes flagged", "", 2, 42),
              ("G-flora-abl Qwen3.5-27B: failures", "", 7, 8), ("G-flora-abl Qwen3.5-27B: successes", "", 2, 42),
              ("G-flora-abl Qwen3.5-9B: successes flagged", "", 16, 42),
              ("G-flora-abl Gemini 3.8 Flash: failures", "", 8, 8), ("G-flora-abl Gemini 3.8 Flash: successes", "", 3, 42),
              ("REAP v1 logistic: failures", "", 6, 8), ("REAP v1 logistic: false interventions", "", 7, 42),
              ("REAP v1 elapsed time: failures", "", 0, 8),
              ("REAP v2 logistic: failures", "", 3, 8), ("REAP v2 logistic: false interventions", "", 5, 42),
              ("REAP v2 MLP: failures", "", 6, 8), ("REAP v2 MLP: false interventions", "", 13, 42),
              ("REAP v2 GRU floor: failures", "", 2, 8), ("REAP v2 GRU floor: false interventions", "", 3, 42),
              ("G9c-3: false positives", "", 0, 25), ("G9c-3: false negatives", "", 5, 62)]
    cc = {name: {"k": k, "n": n, "wilson95": wilson(k, n), "bootstrap95": boot_count(k, n)} for name, _, k, n in counts if n}
    (HERE / "results/counts_ci.json").write_text(json.dumps(cc, indent=1))
    print(json.dumps({k: out[k] for k in ("n", "n_success", "a_pooled", "a_within_policy_stratified", "b_timeout_share", "reading")}, indent=1))
    print(json.dumps({t: {k: v for k, v in x.items()} for t, x in out["b_truncated"].items()}, indent=1))
    print({p: round(v["auroc"], 3) if v["auroc"] is not None else None for p, v in out["a_per_policy"].items()})


if __name__ == "__main__":
    main()
