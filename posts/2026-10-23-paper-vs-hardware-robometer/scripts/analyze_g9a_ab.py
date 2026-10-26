#!/usr/bin/env python3
"""G9a-AB analysis, exactly the metrics of G9A_AB_PROTOCOL.md. numpy/scipy/pandas only.

  python analyze_g9a_ab.py --out-dir OUT   (reads OUT/ab_top.jsonl, ab_front.jsonl, ab_top_teleop.jsonl, k1_top.jsonl)
Writes OUT/results_g9a_ab.json, OUT/scores_ab.csv, OUT/labels_k1000dai.csv, OUT/summary.md.
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

RNG = np.random.default_rng(0)
# deviation 3 (recorded before any result was read): training-data check of the pinned checkpoint
CONTAMINATION = {
    "status": "clean",
    "checkpoint": "robometer/Robometer-4B @ beef63bc914c5c189329d49c6d712d96d632aa34 (last commit 2026-03-10)",
    "evidence": [
        "checkpoint config.yaml: data.train_datasets = [rbm-1m-id], eval_datasets = [mw]",
        "robometer @ 352d160: dataset_category.py:305 rbm-1m-id does not list ArmnetBench",
        "robometer commit f5af481 (2026-06-28) adds ArmnetBench as a separate alias armnetbench_v01 (dataset_category.py:91-94), preprocess.yaml:23, dataset_success_cutoff.txt:66, name_mapping.py:96-97",
        "ArmnetBench repos created 2026-06-28 (villekuosmanen/armnetbench_robometer_v01), 2026-07-03 (armnet/armnetbench_v01_lerobot_so101), 2026-08-16 (jesbu1/armnetbench_v01_tiled_so101): all after the checkpoint",
    ],
    "caveat": "later Robometer checkpoints may include ArmnetBench (it is registered for training); this result holds for beef63b only",
}
B = 2000


def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_ci(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float); a = auroc(y, s)
    if a is None:
        return None, None, None
    bs = []
    for _ in range(B):
        i = RNG.integers(0, len(y), len(y)); v = auroc(y[i], s[i])
        if v is not None:
            bs.append(v)
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def calib(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    bins = np.clip((p * 10).astype(int), 0, 9); tab = []; ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            tab.append({"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "n": int(m.sum()), "mean_p": float(p[m].mean()), "frac_success": float(y[m].mean())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    pred = p >= 0.5; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else None; rec = tp / (tp + fn) if tp + fn else None
    return {"ece": float(ece), "brier": float(((p - y) ** 2).mean()), "accuracy_at_0.5": float((pred == (y == 1)).mean()),
            "precision": prec, "recall": rec, "f1": (2 * prec * rec / (prec + rec)) if prec and rec else None, "reliability": tab}


def load(path):
    if not os.path.exists(path):
        return None
    d = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
    d["task_s"] = d["task"].astype(str)
    return d


def view_metrics(d):
    errs = int(d["p_succ"].isna().sum()) if "p_succ" in d else len(d)
    d = d.dropna(subset=["p_succ"])
    b = d[d.success_class.isin(["successful", "failure"])]
    y = (b.success_class == "successful").values; s = b.p_succ.values
    a, lo, hi = auroc_ci(y, s)
    res = {"n_scored": len(d), "n_errors": errs, "n_binary": len(b), "pooled_auroc": a, "pooled_ci95": [lo, hi]}
    pp = {}
    for pol, g in b.groupby("policy_type"):
        aa, l, h = auroc_ci((g.success_class == "successful").values, g.p_succ.values)
        pp[pol] = {"n": len(g), "n_success": int((g.success_class == "successful").sum()), "auroc": aa, "ci95": [l, h]}
    for v in pp.values():
        v["n_failure"] = v["n"] - v["n_success"]
        v["eligible"] = v["n_success"] >= 10 and v["n_failure"] >= 10   # deviation 4: >= 10 of each class
    res["within_policy"] = pp
    elig = {p: v for p, v in pp.items() if v["eligible"]}
    res["within_policy_excluded"] = sorted(set(pp) - set(elig))
    vals = [v["auroc"] for v in elig.values() if v["auroc"] is not None]
    res["within_policy_mean"] = float(np.mean(vals)) if vals else None
    res["n_eligible_policies"] = len(elig)
    res["n_policies_auroc_ge_0.8"] = int(sum(v >= 0.8 for v in vals))
    need = min(5, len(elig))
    res["within_policy_rule"] = (f"at least 5 of {len(elig)} eligible policies" if len(elig) >= 5
                                 else f"all {len(elig)} eligible policies (fewer than 5 eligible)")
    res["within_policy_pass"] = bool(len(elig) > 0 and res["n_policies_auroc_ge_0.8"] >= need)
    # deviation 5: k1000dai threshold = max balanced accuracy on this binary set
    thr = np.unique(s); best = (None, -1.0)
    for t in thr:
        pred = s >= t; tpr = (pred & y).sum() / max(y.sum(), 1); tnr = (~pred & ~y).sum() / max((~y).sum(), 1)
        if (tpr + tnr) / 2 > best[1]: best = (float(t), float((tpr + tnr) / 2))
    res["threshold_max_balanced_accuracy"] = {"threshold": best[0], "balanced_accuracy": best[1]}
    res["per_task"] = {t: {"n": len(g), "auroc": auroc((g.success_class == "successful").values, g.p_succ.values)} for t, g in b.groupby("task_s")}
    res["per_task_policy"] = {f"{t} | {p}": auroc((g.success_class == "successful").values, g.p_succ.values) for (t, p), g in b.groupby(["task_s", "policy_type"])}
    # suboptimal
    cls = {"failure": 0, "suboptimal": 1, "successful": 2}
    t3 = d[d.success_class.isin(cls)]
    sub = {}
    for a_, b_ in [("successful", "suboptimal"), ("suboptimal", "failure")]:
        g = d[d.success_class.isin([a_, b_])]
        sub[f"auroc_{a_}_vs_{b_}"] = auroc((g.success_class == a_).values, g.p_succ.values)
    rho = spearmanr(t3.success_class.map(cls), t3.p_succ)
    sub["spearman_ordinal"] = float(rho.statistic) if hasattr(rho, "statistic") else float(rho[0])
    sub["mean_p_succ_by_class"] = t3.groupby("success_class").p_succ.mean().to_dict()
    res["suboptimal"] = sub
    res["calibration"] = calib(y, s)
    # baselines (same binary episodes)
    rate = b.groupby("policy_type").success_class.apply(lambda x: (x == "successful").mean())
    res["baselines"] = {"policy_identity_auroc": auroc(y, b.policy_type.map(rate).values),
                        "episode_length_auroc": auroc(y, -b.length.values),
                        "task_identity_auroc": auroc(y, b.task_s.map(b.groupby("task_s").success_class.apply(lambda x: (x == "successful").mean())).values),
                        "last_progress_auroc": auroc(y, b.last_progress.values),
                        "max_success_prob_auroc": auroc(y, b.success_prob.map(lambda v: max(v) if len(v) else np.nan).values)}
    res["pooled_pass"] = bool(a is not None and a >= 0.85)
    res["pass"] = bool(res["pooled_pass"] and res["within_policy_pass"])
    return res, d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", required=True); o = ap.parse_args().out_dir
    out = {"protocol": "G9A_AB_PROTOCOL.md", "primary_view": "top"}
    frames = []
    for v in ["top", "front"]:
        d = load(f"{o}/ab_{v}.jsonl")
        if d is not None:
            out[v], dd = view_metrics(d); frames.append(dd.assign(view=v))
    if frames:
        pd.concat(frames)[["view", "episode_index", "policy_type", "success_class", "task_s", "length", "n_frames", "p_succ", "last_progress"]].to_csv(f"{o}/scores_ab.csv", index=False)
    t = load(f"{o}/ab_top_teleop.jsonl")
    if t is not None:
        t = t.dropna(subset=["p_succ"]); out["teleop_top"] = {"n": len(t), "mean_p_succ": float(t.p_succ.mean()), "frac_ge_0.5": float((t.p_succ >= 0.5).mean())}
    out["contamination"] = CONTAMINATION
    k = load(f"{o}/k1_top.jsonl")
    if k is not None:
        t_star = out.get("top", {}).get("threshold_max_balanced_accuracy", {}).get("threshold")
        k["label_at_0.5"] = np.where(k.p_succ.isna(), "", np.where(k.p_succ >= 0.5, "success", "failure"))
        k["label_at_t_star"] = np.where(k.p_succ.isna() | (t_star is None), "", np.where(k.p_succ >= (t_star if t_star is not None else 2), "success", "failure"))
        usable = bool(out.get("top", {}).get("pass") and CONTAMINATION["status"] == "clean")
        k["labels_usable"] = usable
        k[["episode_index", "task", "length", "n_frames", "p_succ", "last_progress", "label_at_0.5", "label_at_t_star", "labels_usable"] + (["error"] if "error" in k else [])].to_csv(f"{o}/labels_k1000dai.csv", index=False)
        kk = k.dropna(subset=["p_succ"])
        out["k1000dai"] = {"n": len(k), "n_scored": len(kk), "t_star_from_armnetbench_top": t_star,
                           "frac_success_at_0.5": float((kk.p_succ >= 0.5).mean()),
                           "frac_success_at_t_star": float((kk.p_succ >= t_star).mean()) if t_star is not None else None,
                           "mean_p_succ": float(kk.p_succ.mean()), "labels_usable": usable,
                           "usable_rule": "G9a-AB pass on the primary view AND contamination check clean"}
    if "top" in out:
        out["decision"] = "usable zero-shot (pass)" if out["top"]["pass"] else "not usable zero-shot (fail)"
    json.dump(out, open(f"{o}/results_g9a_ab.json", "w"), indent=1, default=float)
    lines = ["# G9a-AB summary", ""]
    for v in ["top", "front"]:
        if v in out:
            r = out[v]
            lines += [f"## view {v}{' (primary)' if v == 'top' else ''}",
                      f"- pooled AUROC {r['pooled_auroc']:.3f} [{r['pooled_ci95'][0]:.3f}, {r['pooled_ci95'][1]:.3f}] on {r['n_binary']} episodes ({r['n_errors']} scoring errors)",
                      f"- within-policy AUROC: " + ", ".join(f"{p} {x['auroc']:.2f}" if x['auroc'] is not None else f"{p} n/a" for p, x in r["within_policy"].items()) + f"; >= 0.8 for {r['n_policies_auroc_ge_0.8']}/7",
                      f"- baselines: policy identity {r['baselines']['policy_identity_auroc']:.3f}, episode length {r['baselines']['episode_length_auroc']:.3f}",
                      f"- calibration: ECE {r['calibration']['ece']:.3f}, Brier {r['calibration']['brier']:.3f}, accuracy at 0.5 {r['calibration']['accuracy_at_0.5']:.3f}",
                      f"- within-policy rule: {r['within_policy_rule']} (excluded: {', '.join(r['within_policy_excluded']) or 'none'}); pass {r['within_policy_pass']}",
                      f"- threshold with max balanced accuracy: {r['threshold_max_balanced_accuracy']}",
                      f"- pass: {r['pass']}", ""]
    lines.append(f"Contamination: {out['contamination']['status']} ({out['contamination']['checkpoint']})")
    if "k1000dai" in out:
        lines.append(f"k1000dai: labels usable = {out['k1000dai']['labels_usable']}; success fraction at 0.5 = {out['k1000dai']['frac_success_at_0.5']:.2f}, at t* = {out['k1000dai']['frac_success_at_t_star']}")
    if "decision" in out:
        lines.append(f"**Decision (protocol rule, primary view): {out['decision']}**")
    open(f"{o}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
