#!/usr/bin/env python3
"""G9a-AB complete set: first-run top-view scores + the re-score of the 462 OOM episodes (remote_g9a_ab_rescore.sh).

Runs the pre-registered metrics (analyze_g9a_ab.view_metrics, unchanged) on the partial set (first run) and the
complete set (first run, errored records replaced by their re-score), plus the length follow-up checks (a)/(b) and
the per-task numbers of the two cable tasks.

  python analyze_g9a_ab_complete.py --out-dir ~/robot-data/g9a-ab-results/out \
      --rescore ~/robot-data/g9a-ab-results/rescore/ab_top_rescore.jsonl --json results/g9a_ab/complete.json
"""
import argparse, json, os, sys

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules (run with python -I)
import analyze_g9a_ab as A  # noqa: E402
import analyze_g9a_ab_followup as F  # noqa: E402

CABLE = ["Remove the power cable from cable holder", "Push the Displayport cable into the cable holder on the white block"]


def summary(res):
    keep = ["n_scored", "n_errors", "n_binary", "pooled_auroc", "pooled_ci95", "within_policy_mean", "n_policies_auroc_ge_0.8",
            "within_policy_rule", "within_policy_pass", "pooled_pass", "pass", "threshold_max_balanced_accuracy", "per_task",
            "suboptimal", "baselines"]
    out = {k: res[k] for k in keep}
    out["within_policy"] = {p: {"auroc": v["auroc"], "ci95": v["ci95"], "n_success": v["n_success"], "n_failure": v["n_failure"]}
                            for p, v in res["within_policy"].items()}
    out["calibration"] = {k: v for k, v in res["calibration"].items() if k != "reliability"}
    return out


def cable(d, t_star):
    d = d.dropna(subset=["p_succ"])
    out = {}
    for t in CABLE:
        g = d[d.task_s == t]
        y = (g.success_class == "successful").values
        b = g[g.success_class.isin(["successful", "failure"])]
        yb = (b.success_class == "successful").values
        out[t] = {"n_scored": len(g), "n_success": int(y.sum()), "n_failure": int((g.success_class == "failure").sum()),
                  "auroc": A.auroc(yb, b.p_succ.values),
                  "mean_p_succ": g.groupby("success_class").p_succ.mean().to_dict(),
                  "frac_ge_t_star": g.groupby("success_class").p_succ.apply(lambda s: float((s >= t_star).mean())).to_dict(),
                  "within_policy": {p: A.auroc((gg.success_class == "successful").values, gg.p_succ.values)
                                    for p, gg in b.groupby("policy_type")}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True); ap.add_argument("--rescore", required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args()
    part = A.load(f"{a.out_dir}/ab_top.jsonl")
    rs = A.load(a.rescore)
    rs_ok = rs.dropna(subset=["p_succ"])
    comp = pd.concat([part[~part.episode_index.isin(rs_ok.episode_index)], rs_ok]).sort_values("episode_index").reset_index(drop=True)
    assert comp.episode_index.is_unique and len(comp) == len(part)

    out = {"rescore": {"n": len(rs), "n_scored": len(rs_ok), "n_still_failing": int(rs.p_succ.isna().sum()),
                       "still_failing": rs.reindex(columns=["episode_index", "task_s", "length", "error", "p_succ"])[rs.p_succ.isna()].to_dict(orient="records"),
                       "n_subsampled": int((rs_ok.get("subsampled", pd.Series(dtype=object)) == True).sum()),  # noqa: E712
                       "attempts": rs_ok.get("attempts", pd.Series(dtype=float)).value_counts().to_dict(),
                       "n_frames_max": int(rs_ok.n_frames.max())}}
    for name, d in [("partial", part), ("complete", comp)]:
        A.RNG = np.random.default_rng(0)  # same bootstrap stream as the first analysis (top view first)
        res, dd = A.view_metrics(d)
        out[name] = summary(res)
        out[name]["cable_tasks"] = cable(dd, res["threshold_max_balanced_accuracy"]["threshold"])
        lc = F.length_confound(d.assign(errored=d.p_succ.isna()), np.random.default_rng(0))
        out[name]["length_followup"] = {"a_quintiles": lc["a_quintiles"], "b_non_timeout_failures": lc["b_non_timeout_failures"],
                                        "c_gain_over_length": lc["c_gain_over_length"], "c_gain_ci95": lc["c_gain_ci95"],
                                        "length_auroc": lc["length_auroc"], "robometer_auroc": lc["robometer_auroc"]}
    c = out["complete"]
    out["decision_complete"] = "usable zero-shot (pass)" if c["pass"] else "not usable zero-shot (fail)"
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    for name in ["partial", "complete"]:
        r = out[name]; lf = r["length_followup"]
        print(f"== {name}: pooled {r['pooled_auroc']:.3f} {np.round(r['pooled_ci95'], 3).tolist()} on {r['n_binary']} binary, "
              f"{r['n_errors']} errors | within-policy " + ", ".join(f"{p} {v['auroc']:.2f}" for p, v in r["within_policy"].items())
              + f" | >=0.8: {r['n_policies_auroc_ge_0.8']} | pass {r['pass']} | t* {r['threshold_max_balanced_accuracy']}")
        print(f"   length {r['baselines']['episode_length_auroc']:.3f} | (a) median {lf['a_quintiles']['global']['median_auroc']:.3f} "
              f"within-task {lf['a_quintiles']['within_task']['median_auroc']:.3f} | (b) {lf['b_non_timeout_failures']['robometer_auroc']:.3f} "
              f"| (c) gain {lf['c_gain_over_length']:+.3f}")
        print("   per task: " + "; ".join(f"{t[:28]} {v['auroc']:.2f}" if v["auroc"] is not None else f"{t[:28]} n/a" for t, v in r["per_task"].items()))
    print(json.dumps(out["rescore"], indent=1, default=float)[:1500])
    print(json.dumps(out["complete"]["cable_tasks"], indent=1, default=float))
    print("decision (complete set):", out["decision_complete"])


if __name__ == "__main__":
    main()
