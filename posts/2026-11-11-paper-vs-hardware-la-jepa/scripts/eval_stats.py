"""WS-167: A1 / A2 numbers from the per-episode held-out MAE files (PROTOCOL sections 3 and 5).

  python eval_stats.py --full RUN_s0 [RUN_s1 ...] --nofuture RUN_s0 [...] --smolvla results/smolvla_heldout.json --out results/a1_a2.json

Per-episode MAE is averaged over seeds first; the ratio is mean over the 10 episodes of the numerator / mean of the denominator; bootstrap =
1,000 resamples of the 10 episodes, seed 0, percentile 95 % interval. Mechanical readings of the bar wording (the owner judges, this prints the numbers):
  A1: point ratio full/SmolVLA <= 1.10 and the interval excludes 1.10 (upper bound < 1.10);  else fail if the lower bound > 1.10, else not proven.
  A2: point ratio nofuture/full >= 1.05 and the interval excludes 1.00 (lower bound > 1.00).
"""
import argparse, json
import numpy as np


def load_eps(paths, key="mae"):
    runs = [json.load(open(f"{p}/heldout_mae.json"))["per_episode"] if not str(p).endswith(".json") else json.load(open(p))["per_episode"] for p in paths]
    eps = sorted(runs[0], key=int)
    return eps, np.array([[r[e][key] for e in eps] for r in runs]).mean(0)


def boot(num, den, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    k = len(num)
    r = np.empty(n)
    for i in range(n):
        j = rng.integers(0, k, k)
        r[i] = num[j].mean() / den[j].mean()
    return float(num.mean() / den.mean()), [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", nargs="+", required=True); ap.add_argument("--nofuture", nargs="+", required=True)
    ap.add_argument("--smolvla", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ef, full = load_eps(a.full); en, nof = load_eps(a.nofuture); es, smol = load_eps([a.smolvla]); _, null = load_eps(a.full, "identity_null")
    assert ef == en == es, ("episode sets differ", ef, en, es)   # same-set rule
    res = {"episodes": ef, "seeds_full": len(a.full), "seeds_nofuture": len(a.nofuture),
           "mean_mae": {"full": float(full.mean()), "nofuture": float(nof.mean()), "smolvla": float(smol.mean()), "identity_null": float(null.mean())}}
    r, ci = boot(full, smol)
    res["A1"] = {"ratio_full_over_smolvla": r, "ci95": ci, "bar": "<= 1.10 and CI excludes 1.10",
                 "reading": "pass" if r <= 1.10 and ci[1] < 1.10 else ("fail" if ci[0] > 1.10 else "not proven")}
    r2, ci2 = boot(nof, full)
    res["A2"] = {"ratio_nofuture_over_full": r2, "ci95": ci2, "bar": ">= 1.05 and CI excludes 1.00",
                 "reading": "pass" if r2 >= 1.05 and ci2[0] > 1.0 else ("fail" if r2 < 1.05 and ci2[1] < 1.05 else "not proven")}
    res["ratio_to_null"] = {"full": float(full.mean() / null.mean()), "smolvla": float(smol.mean() / null.mean())}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
