"""Gate A analysis: criteria 1 and 2 of G0-RJ with 95% CIs (bootstrap over episodes), tables and plots from gateA_*.json."""
import glob
import json
import os
import sys

import numpy as np

ARMS = ("model_true", "model_shuffled_cons", "model_shuffled_truestate", "copy_last")


def load(path):
    d = json.load(open(path))
    eps = np.array([c["episode"] for c in d["per_clip"]])
    # per clip: arm -> mean over prefixes -> [9]; ctx2 = prefix t=1
    per = {k: np.array([np.mean(c["l1"][k], axis=0) for c in d["per_clip"]]) for k in ARMS}
    ctx2 = {k: np.array([c["l1"][k][1] for c in d["per_clip"]]) for k in ARMS}
    return d, eps, per, ctx2


def boot(eps, f, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(eps)
    idx = {e: np.where(eps == e)[0] for e in u}
    vals = []
    for _ in range(n):
        s = rng.choice(u, size=len(u), replace=True)
        vals.append(f(np.concatenate([idx[e] for e in s])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main(res_dir, out_json):
    rows, out = [], {}
    for path in sorted(glob.glob(os.path.join(res_dir, "gateA_*_setting*.json"))):
        d, eps, per, ctx2 = load(path)
        key = f"{d['size']}_{d['step_k']}k_setting{d['setting']}"
        m19 = lambda arm, ix=slice(None): per[arm][ix].mean()  # noqa: E731
        red_copy = lambda ix: 1 - per["model_true"][ix].mean() / per["copy_last"][ix].mean()  # noqa: E731
        red_shuf = lambda ix: 1 - per["model_true"][ix].mean() / per["model_shuffled_cons"][ix].mean()  # noqa: E731
        red_shuf_ts = lambda ix: 1 - per["model_true"][ix].mean() / per["model_shuffled_truestate"][ix].mean()  # noqa: E731
        c2 = lambda ix: 1 - ctx2["model_true"][ix].mean() / ctx2["copy_last"][ix].mean()  # noqa: E731
        s9 = lambda ix: 1 - per["model_true"][ix, 8].mean() / per["copy_last"][ix, 8].mean()  # noqa: E731
        o = {"clips": len(eps), "episodes": int(len(np.unique(eps))),
             "mean_1_9": {k: float(per[k].mean()) for k in ARMS},
             "per_horizon": {k: per[k].mean(0).tolist() for k in ARMS},
             "crit1_reduction_vs_copy_last": {"value": float(red_copy(slice(None))), "ci95": boot(eps, red_copy)},
             "crit2_reduction_vs_shuffled_consistent": {"value": float(red_shuf(slice(None))), "ci95": boot(eps, red_shuf)},
             "secondary_reduction_vs_shuffled_true_states": {"value": float(red_shuf_ts(slice(None))), "ci95": boot(eps, red_shuf_ts)},
             "secondary_step9_reduction_vs_copy_last": {"value": float(s9(slice(None))), "ci95": boot(eps, s9)},
             "secondary_ctx2_reduction_vs_copy_last": {"value": float(c2(slice(None))), "ci95": boot(eps, c2)}}
        out[key] = o
        rows.append((key, o))
    json.dump(out, open(out_json, "w"), indent=1)
    print(f"{'run':28} {'n':>4} {'true':>7} {'shufC':>7} {'copy':>7} | crit1 (vs copy) | crit2 (vs shuf) | step9 | ctx2")
    for k, o in rows:
        f = lambda n: f"{o[n]['value'] * 100:5.1f}% [{o[n]['ci95'][0] * 100:4.1f},{o[n]['ci95'][1] * 100:4.1f}]"  # noqa: E731
        m = o["mean_1_9"]
        print(f"{k:28} {o['clips']:4d} {m['model_true']:7.4f} {m['model_shuffled_cons']:7.4f} {m['copy_last']:7.4f} | {f('crit1_reduction_vs_copy_last')} | "
              f"{f('crit2_reduction_vs_shuffled_consistent')} | {f('secondary_step9_reduction_vs_copy_last')} | {f('secondary_ctx2_reduction_vs_copy_last')}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, setting in zip(axs, "ab"):
            for k, o in rows:
                if k.endswith(f"setting{setting}"):
                    ax.plot(range(1, 10), o["per_horizon"]["model_true"], marker="o", label=k.rsplit("_", 1)[0] + " true")
            k0, o0 = next((k, o) for k, o in rows if k.endswith(f"setting{setting}"))
            ax.plot(range(1, 10), o0["per_horizon"]["copy_last"], "k--", label="copy-last")
            ax.plot(range(1, 10), o0["per_horizon"]["model_shuffled_cons"], "r:", label=f"shuffled ({k0.rsplit('_', 1)[0]})")
            ax.set_title(f"camera setting {setting}" + (" (primary)" if setting == "a" else " (ablation)"))
            ax.set_xlabel("rollout step j")
        axs[0].set_ylabel("per-token L1")
        axs[0].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(out_json), "gateA_l1_per_horizon.png"), dpi=130)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
