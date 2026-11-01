#!/usr/bin/env python3
"""TOPREWARD_PROTOCOL.md analyses on the score files (written before any 8B score).

    python3 topreward_analysis.py --ab ab.jsonl --own own.jsonl --meta ~/robot-data/armnetbench_meta \
        --own-labels ~/robot-data/so101-bowls-policy-rollouts/meta/rollout_labels.csv
Writes ../results/topreward.json.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from length_analysis import auroc, stratified, B, SEED  # noqa: E402  (LENGTH_PROTOCOL.md functions)


def z(x):
    x = np.asarray(x, float); return (x - x.mean()) / x.std()


def scores(path):
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    err = d[d.get("error").notna()] if "error" in d else d.iloc[:0]
    d = d[d.get("error").isna()] if "error" in d else d
    m = d.apply(lambda r: np.array(r.r_true) - np.array(r.r_false), axis=1)
    d = d.assign(s_top=m.apply(lambda v: v[-3:].mean() - v[:3].mean()),
                 s_yn=d.yn_full.apply(lambda v: v[0] - v[1]), s_cap=d.yn_cap.apply(lambda v: v[0] - v[1]))
    return d, len(err)


def paired(df, a, b, strat):
    rng = np.random.default_rng(SEED); groups = [g for _, g in df.groupby(strat)]; v = []
    for _ in range(B):
        s = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
        x, y = stratified(s, a, strat), stratified(s, b, strat)
        if x is not None and y is not None:
            v.append(x - y)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot1(df, fn, strat):
    rng = np.random.default_rng(SEED); groups = [g for _, g in df.groupby(strat)]; v = []
    for _ in range(B):
        x = fn(pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups]))
        if x is not None:
            v.append(x)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def reading(ci):
    return "beats" if ci[0] > 0 else ("below" if ci[1] < 0 else "matches")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ab"); ap.add_argument("--own"); ap.add_argument("--meta")
    ap.add_argument("--own-labels"); a = ap.parse_args(); out = {"protocol": "TOPREWARD_PROTOCOL.md"}
    if a.ab:
        d, nerr = scores(a.ab)
        d = d[d.success_class.isin(["successful", "failure"]) & (d.policy_type != "teleoperated")].copy()
        d["y"] = d.success_class == "successful"; d["neg_len"] = -d.length.astype(float)
        T = d.groupby("task").length.max(); d["neg_cap"] = -np.minimum(d.length, 0.5 * d.task.map(T))
        d["s_succ"] = z(d.s_top) + z(d.s_yn)
        r = {"n": len(d), "n_success": int(d.y.sum()), "errors_all_scored": nerr}
        for s in ("s_succ", "s_top", "s_yn", "s_cap", "neg_len", "neg_cap"):
            r[s] = {"pooled": auroc(d.y, d[s]), "pooled_ci95": boot1(d, lambda x: auroc(x.y, x[s]), "policy_type"),
                    "within": stratified(d, s, "policy_type"), "within_ci95": boot1(d, lambda x: stratified(x, s, "policy_type"), "policy_type")}
        r["per_policy"] = {p: auroc(g.y, g.s_succ) for p, g in d.groupby("policy_type")}
        r["per_task"] = {t: auroc(g.y, g.s_succ) for t, g in d.groupby("task")}
        r["delta_within_vs_length_ci95"] = paired(d, "s_succ", "neg_len", "policy_type")
        r["delta_within_cap_vs_length_cap_ci95"] = paired(d, "s_cap", "neg_cap", "policy_type")
        r["reading_full"] = reading(r["delta_within_vs_length_ci95"]); r["reading_cap50"] = reading(r["delta_within_cap_vs_length_cap_ci95"])
        out["armnetbench"] = r
    if a.own:
        d, nerr = scores(a.own)
        lab = pd.read_csv(a.own_labels)[["episode_index", "policy", "outcome"]]
        d = d.merge(lab, on="episode_index"); d = d[d.outcome.isin(["success", "failure"])].copy()
        d["y"] = d.outcome == "success"; d["neg_len"] = -d.length.astype(float); d["s_succ"] = z(d.s_top) + z(d.s_yn)
        d["pol_rate"] = d.policy.map(d.groupby("policy").y.mean()); d["one"] = 1
        r = {"n": len(d), "n_success": int(d.y.sum()), "errors": nerr}
        for s in ("s_succ", "s_top", "s_yn", "neg_len", "pol_rate"):
            r[s] = {"auroc": auroc(d.y, d[s]), "ci95": boot1(d, lambda x: auroc(x.y, x[s]), "outcome")}
        nt = d[~d.policy.str.startswith("turbovla")]
        r["non_turbovla"] = {"n": len(nt), "n_failure": int((~nt.y).sum()), "auroc": auroc(nt.y, nt.s_succ)}
        out["own"] = r
    p = HERE.parent / "results/topreward.json"; p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
