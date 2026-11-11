"""H1/H2 aggregation exactly as PROTOCOL.md §3/§5: success per seed and mean; OpTTA-AdaJEPA with a seed-paired
bootstrap 95% CI (2000 resamples, seed 0). Pairing: the same 50 start-goal pairs per seed for every method, so each
resample draws episode indices with replacement within each seed (one draw shared by both methods) and averages the
per-seed differences over the 3 seeds. Anything marked exploratory was not pre-registered."""
import json, numpy as np
R = "projects/optta/results/"
PAPER = dict(frozen=118, adajepa=141, optta=163, optta30=165)
def load(m, s): return json.load(open(f"{R}h2_{m}_s{s}.json"))["episodes"]
succ = {m: np.array([[e["success"] for e in load(m, s)] for s in range(3)], float) for m in ("frozen", "adajepa", "optta")}
pess = np.array([e["success"] for e in load("pess", 0)], float)
out = {"H2": {}, "H1": {}}
for m, a in succ.items():
    out["H2"][m] = dict(per_seed_pct=[round(100 * x, 1) for x in a.mean(1)], mean_pct=round(100 * a.mean(), 2), n_per_seed=50)
out["H2"]["pess_s0_pct"] = round(100 * pess.mean(), 1)
out["H2"]["seed0_paired_bootstrap_ref"] = dict(adajepa_s0_pct=round(100*succ["adajepa"][0].mean(),1), optta_s0_pct=round(100*succ["optta"][0].mean(),1))
def boot(x, y, B=2000):
    g = np.random.default_rng(0); n = x.shape[1]; d = []
    for _ in range(B):
        idx = g.integers(0, n, size=(3, n))
        d.append(np.mean([(x[s, idx[s]] - y[s, idx[s]]).mean() for s in range(3)]) * 100)
    d = np.array(d); return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]
for a, b in (("optta", "adajepa"), ("adajepa", "frozen"), ("optta", "frozen")):
    diff = round(100 * (succ[a].mean() - succ[b].mean()), 2)
    out["H2"][f"{a}_minus_{b}"] = dict(pp=diff, ci95=boot(succ[a], succ[b]), per_seed_pp=[round(100 * x, 1) for x in (succ[a] - succ[b]).mean(1)],
                                       preregistered=(a, b) == ("optta", "adajepa"))
d = out["H2"]["optta_minus_adajepa"]
pess_ok = out["H2"]["pess_s0_pct"] < 100 * succ["adajepa"][0].mean()
bar = d["pp"] >= 5 and (d["ci95"][0] > 0 or d["ci95"][1] < 0)
out["H2"]["verdict"] = dict(bar_met=bool(bar), pess_below_adajepa_s0=bool(pess_ok),
    word="pass" if (bar and pess_ok) else "fail")
# H1: ms per replan = CEM + adaptation step (CUDA-synced), first 100 replans of each run; 25 W FP32 Orin
def ms(m, s):
    eps = load(m, s); p = [x for e in eps for x in e["plan_ms"]]; a = [x for e in eps for x in e["adapt_ms"]]
    n = min(100, len(p)); tot = [p[i] + (a[i] if i < len(a) else 0) for i in range(n)]
    return float(np.median(tot)), float(np.median(a[:100])) if a else 0.0
for m in ("frozen", "adajepa", "optta"):
    v = [ms(m, s) for s in range(3)]
    out["H1"][m] = dict(ms_per_seed=[round(x[0]) for x in v], adapt_ms_per_seed=[round(x[1]) for x in v], ms_median_of_seeds=round(float(np.median([x[0] for x in v]))),
                        ratio_vs_paper=round(float(np.median([x[0] for x in v])) / PAPER[m], 1), paper_ms=PAPER[m])
v = ms("optta30", 0)
out["H1"]["optta30"] = dict(ms=round(v[0]), adapt_ms=round(v[1]), ratio_vs_paper=round(v[0] / PAPER["optta30"], 1), paper_ms=165, n_episodes=5)
out["H1"]["optta_update_alone_ms"] = out["H1"]["optta"]["adapt_ms_per_seed"]
out["H1"]["optta_update_over_1s"] = bool(max(out["H1"]["optta_update_alone_ms"]) > 1000)
out["H1"]["note"] = "25 W quiet profile, FP32 TF32 off, Orin NX, not the paper's RTX PRO 6000; frozen s0 CEM (3596 ms) is 2.5x the other 8 runs (about 1.4 s) and is flagged as an unexplained outlier (the pilot also saw 3.6 s); median of seeds used."
json.dump(out, open(R + "summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
