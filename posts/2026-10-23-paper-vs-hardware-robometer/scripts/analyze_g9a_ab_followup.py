#!/usr/bin/env python3
"""G9a-AB follow-up (exploratory, after the results were read): length confound and scoring errors.

No new scores: reads the pushed per-episode scores (ab_top.jsonl, ab_front.jsonl) and the ArmnetBench episode
metadata. numpy / pandas / scipy / scikit-learn.

  python analyze_g9a_ab_followup.py --out-dir ~/robot-data/g9a-ab-results/out --json results/g9a_ab/followup.json
"""
import argparse, json

import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

FPS = 20
B = 2000


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(fn, idx_groups, rng):
    """Episode bootstrap of a statistic fn(index_array)."""
    n = len(idx_groups); vals = []
    for _ in range(B):
        v = fn(idx_groups[rng.integers(0, n, n)])
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def load(path):
    d = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
    d["task_s"] = d["task"].astype(str)
    d["errored"] = d["p_succ"].isna()
    return d


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def cv_auroc(X, y, groups_seed, n_rep=20):
    """Out-of-fold AUROC of a logistic model, 5-fold stratified, repeated n_rep times; returns per-repeat AUROCs."""
    out = []
    for rep in range(n_rep):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=groups_seed + rep).split(X, y):
            m = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
            oof[te] = m.decision_function(X[te])
        out.append(auroc(y, oof))
    return np.array(out)


def length_confound(d, rng):
    b = d[~d.errored & d.success_class.isin(["successful", "failure"])].copy()
    b["y"] = (b.success_class == "successful").astype(int)
    y = b.y.values.astype(bool)
    res = {"n": len(b), "n_success": int(y.sum()), "n_failure": int((~y).sum()),
           "robometer_auroc": auroc(y, b.p_succ), "length_auroc": auroc(y, -b.length)}

    # per-task max length = the rollout time limit (failures that time out end there)
    tmax = d.groupby("task_s").length.max()
    b["task_max"] = b.task_s.map(tmax)
    b["at_max"] = b.length >= b.task_max - FPS  # within 1 s of the task's longest rollout
    res["task_max_length_frames"] = {t: int(v) for t, v in tmax.items()}
    res["length_by_task_class"] = {
        t: {c: {"median": float(g[g.success_class == c].length.median()), "n": int((g.success_class == c).sum()),
                "n_at_max": int(g[(g.success_class == c)].at_max.sum())} for c in ["successful", "failure"]}
        for t, g in b.groupby("task_s")}

    # (a) quintiles of length: global, and within task (length rank inside its task)
    qa = {}
    for name, key in [("global", b.length.rank(method="first")),
                      ("within_task", b.groupby("task_s").length.rank(method="first", pct=True))]:
        q = pd.qcut(key, 5, labels=False)
        rows = []
        for k in range(5):
            g = b[q == k]
            yy = g.y.values.astype(bool)
            rows.append({"quintile": k + 1, "length_range_s": [float(g.length.min() / FPS), float(g.length.max() / FPS)],
                         "n": len(g), "n_success": int(yy.sum()), "n_failure": int((~yy).sum()),
                         "robometer_auroc": auroc(yy, g.p_succ), "length_auroc_inside": auroc(yy, -g.length)})
        vals = [r["robometer_auroc"] for r in rows if r["robometer_auroc"] is not None]
        # also: quintile-stratified AUROC = pairs compared only inside the same quintile (weighted by pairs)
        num = den = 0.0
        for r, k in zip(rows, range(5)):
            if r["robometer_auroc"] is not None:
                w = r["n_success"] * r["n_failure"]; num += w * r["robometer_auroc"]; den += w
        qa[name] = {"rows": rows, "median_auroc": float(np.median(vals)) if vals else None,
                    "pair_weighted_auroc": num / den if den else None}
    res["a_quintiles"] = qa

    # (b) failures that did not time out vs all successes
    nf = b[(b.y == 1) | ((b.y == 0) & ~b.at_max)]
    yy = nf.y.values.astype(bool)
    idx = np.arange(len(nf))
    res["b_non_timeout_failures"] = {
        "n_failure_not_at_max": int((~yy).sum()), "n_failure_at_max": int(((b.y == 0) & b.at_max).sum()),
        "n_success": int(yy.sum()), "n_success_at_max": int(((b.y == 1) & b.at_max).sum()),
        "robometer_auroc": auroc(yy, nf.p_succ),
        "robometer_ci95": boot_ci(lambda i: auroc(yy[i], nf.p_succ.values[i]), idx, rng),
        "length_auroc": auroc(yy, -nf.length)}

    # (c) cross-validated logistic: length vs length + Robometer (also with task fixed effects)
    L = np.log(b.length.values)[:, None]
    R = logit(b.p_succ.values)[:, None]
    T = pd.get_dummies(b.task_s).values.astype(float)
    Lt = (b.length / b.task_max).values[:, None]
    yv = b.y.values
    models = {"length": L, "length+robometer": np.hstack([L, R]), "robometer": R,
              "length+task": np.hstack([L, Lt, T]), "length+task+robometer": np.hstack([L, Lt, T, R])}
    cv = {k: cv_auroc(v, yv, 0) for k, v in models.items()}
    res["c_logistic_cv"] = {k: {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())} for k, v in cv.items()}
    res["c_gain_over_length"] = float((cv["length+robometer"] - cv["length"]).mean())
    res["c_gain_over_length_task"] = float((cv["length+task+robometer"] - cv["length+task"]).mean())
    # bootstrap CI of the gain (in-sample fit per resample is optimistic; use OOF scores from one CV repeat)
    oof = {}
    for k in ["length", "length+robometer", "length+task", "length+task+robometer"]:
        o = np.zeros(len(yv))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(models[k], yv):
            o[te] = LogisticRegression(max_iter=2000).fit(models[k][tr], yv[tr]).decision_function(models[k][te])
        oof[k] = o
    yb = yv.astype(bool); idx = np.arange(len(yv))
    res["c_gain_ci95"] = boot_ci(lambda i: (lambda a, c: None if a is None or c is None else c - a)(
        auroc(yb[i], oof["length"][i]), auroc(yb[i], oof["length+robometer"][i])), idx, rng)
    res["c_gain_task_ci95"] = boot_ci(lambda i: (lambda a, c: None if a is None or c is None else c - a)(
        auroc(yb[i], oof["length+task"][i]), auroc(yb[i], oof["length+task+robometer"][i])), idx, rng)

    # (d) exploratory, closest to "decide at subtask end": matched-time comparison. Each success (ends at its
    # success time t) vs each failure that is still running at t, scored by the failure's success prob at the
    # 1 fps frame nearest t. Approximation: the failure frame's prob comes from a forward pass that saw the whole
    # episode (if Robometer's per-frame head is not causal, later frames leak into it).
    s_eps = b[b.y == 1]; f_eps = b[b.y == 0]
    wins = ties = pairs = 0
    per_task = {}
    for t, gs in s_eps.groupby("task_s"):
        gf = f_eps[f_eps.task_s == t]
        w_t = p_t = 0.0
        for _, s in gs.iterrows():
            ts = s.length / FPS
            for _, f in gf.iterrows():
                if f.length / FPS < ts:
                    continue
                sp = f.success_prob
                k = min(int(round(ts)), len(sp) - 1)
                if s.p_succ > sp[k]: w_t += 1
                elif s.p_succ == sp[k]: w_t += 0.5
                p_t += 1
        if p_t:
            per_task[t] = {"pairs": int(p_t), "auroc": w_t / p_t}
            wins += w_t; pairs += p_t
    res["d_matched_time_auroc"] = {"pooled_within_task": wins / pairs if pairs else None, "pairs": int(pairs), "per_task": per_task}
    return res


def errors(d):
    d = d.copy()
    d["n_frames_expected"] = np.floor(d.length / FPS).astype(int) + 1
    res = {"n": len(d), "n_errored": int(d.errored.sum()),
           "error_types": d[d.errored].error.str.extract(r"^(\w+)")[0].value_counts().to_dict()}
    tab = lambda col: pd.crosstab(d[col], d.errored).rename(columns={False: "scored", True: "errored"}).assign(
        frac_errored=lambda x: (x.errored / (x.errored + x.scored)).round(3)).to_dict(orient="index")
    res["by_outcome"] = tab("success_class"); res["by_policy"] = tab("policy_type"); res["by_task"] = tab("task_s")
    res["length_s"] = {k: {"median": float(g.length.median() / FPS), "min": float(g.length.min() / FPS),
                           "max": float(g.length.max() / FPS)} for k, g in d.groupby(d.errored.map({True: "errored", False: "scored"}))}
    # frame-count threshold: max frames ever scored, min frames that errored
    sc = d[~d.errored]; er = d[d.errored]
    res["frames"] = {"max_scored_n_frames": int(sc.n_frames.max()), "min_errored_n_frames_expected": int(er.n_frames_expected.min()),
                     "frac_errored_by_n_frames": d.groupby(pd.cut(d.n_frames_expected, [0, 15, 20, 25, 28, 30, 32, 35, 40, 60, 200]),
                                                           observed=True).errored.agg(["mean", "size"]).round(3).rename(index=str).to_dict(orient="index")}
    # does the protocol's binary contrast change? length baseline on all vs scored episodes
    bb = d[d.success_class.isin(["successful", "failure"])]
    y = (bb.success_class == "successful").values
    res["length_auroc_all_binary"] = auroc(y, -bb.length)
    res["n_binary_all"] = len(bb)
    res["n_binary_errored"] = {c: int((bb.errored & (bb.success_class == c)).sum()) for c in ["successful", "failure"]}
    # cascade check: an OOM leaves the allocator fragmented; are short episodes erroring right after an OOM?
    d2 = d.sort_values("episode_index").reset_index(drop=True)
    prev_err = d2.errored.shift(1, fill_value=False)
    short = d2.n_frames_expected <= d2[~d2.errored].n_frames.quantile(0.5)
    res["cascade"] = {"short_eps_err_rate_after_error": float(d2[short & prev_err].errored.mean()) if (short & prev_err).any() else None,
                      "short_eps_err_rate_after_ok": float(d2[short & ~prev_err].errored.mean())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {}
    for v in ["top", "front"]:
        d = load(f"{a.out_dir}/ab_{v}.jsonl")
        out[v] = {"length_confound": length_confound(d, rng), "errors": errors(d)}
    top = load(f"{a.out_dir}/ab_top.jsonl"); front = load(f"{a.out_dir}/ab_front.jsonl")
    m = top[["episode_index", "errored"]].merge(front[["episode_index", "errored"]], on="episode_index", suffixes=("_top", "_front"))
    out["errored_same_episodes_top_front"] = int((m.errored_top & m.errored_front).sum())
    out["errored_top_only"] = int((m.errored_top & ~m.errored_front).sum())
    out["errored_front_only"] = int((~m.errored_top & m.errored_front).sum())
    lc = out["top"]["length_confound"]
    out["rule"] = {"a_median_quintile_auroc_ge_0.75": (lc["a_quintiles"]["global"]["median_auroc"] or 0) >= 0.75,
                   "c_gain_ge_0.03": lc["c_gain_over_length"] >= 0.03}
    out["robometer_beyond_length"] = bool(any(out["rule"].values()))
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
