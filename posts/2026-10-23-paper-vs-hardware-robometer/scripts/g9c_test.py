#!/usr/bin/env python3
"""G9c test run (G9C_PROTOCOL.md): the frozen judge (g9c_config.json) on the 88 labelled bowls rollouts, one call.

Test set = the G9a-own episodes: ravediamond/so101-bowls-policy-rollouts (Orin NX, labels meta/rollout_labels.csv) +
the M1 sessions (labels shared/gate/data/rollout_catalog.csv, labelled rows only). Prints and writes every metric.

  python g9c_test.py --orin ~/robot-data/so101-bowls-policy-rollouts --m1 ~/.cache/huggingface/lerobot/ravediamond \
      --catalog shared/gate/data/rollout_catalog.csv --config g9c_config.json --out results/g9c
"""
import argparse, glob, json, os, sys, time

import cv2
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g9c_io import Episodes  # noqa: E402
from g9c_judge import episode_score, load_config  # noqa: E402

ROBOMETER = {"g9a_own_A_auroc": 0.577, "g9a_own_B_auroc": 0.620, "policy_identity_auroc": 0.878}


def auroc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); r = rankdata(s); n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(fn, n, rng, b=2000):
    v = []
    for _ in range(b):
        i = rng.integers(0, n, n)
        try:
            v.append(fn(i))
        except ZeroDivisionError:
            pass
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    for k in ("orin", "m1", "catalog", "config", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--end-at-reset", action="store_true",
                    help="EXPLORATORY: judge Orin episodes at their operator-marked reset_start (rollout_labels.csv)")
    a = ap.parse_args(); cfg = load_config(a.config); os.makedirs(a.out, exist_ok=True)
    lab = pd.read_csv(f"{a.orin}/meta/rollout_labels.csv")
    sets = [("orin", a.orin, {int(r.episode_index): r for r in lab.itertuples()})]
    cat = pd.read_csv(a.catalog); cat = cat[(cat.device == "m1") & cat.labeled.astype(bool)]
    for ds in sorted(cat.dataset.unique()):
        rows = cat[cat.dataset == ds]
        sets.append((ds, f"{a.m1}/{ds}", {int(r.episode): r for r in rows.itertuples()}))
    recs, imgs = [], {}
    t_all = time.time()
    for name, root, labels in sets:
        E = Episodes(root)
        for ep in E.ids():
            if int(ep) not in labels:
                continue
            L = labels[int(ep)]
            if name == "orin":
                outcome, policy, mode, device = L.outcome, L.policy, L.failure_mode, "orin_nx"
            else:
                outcome = "success" if str(L.success) == "True" else "failure"; policy, mode, device = L.policy, L.failure_mode, "m1"
            t0 = time.time()
            end = None
            if a.end_at_reset and name == "orin" and int(getattr(L, "reset_start", -1)) >= 0:
                end = int(L.reset_start)
            first, win, dur = E.frames(ep, cfg["max_window_s"], cfg["step_s"], end_frame=end)
            s, info = episode_score(first, win, cfg)
            recs.append({"dataset": name, "episode": int(ep), "device": device, "policy": policy, "outcome": outcome,
                         "failure_mode": mode if isinstance(mode, str) else ("unspecified" if outcome == "failure" else None),
                         "s": s, "undecidable": s is None, "end_frame": end, "seconds": time.time() - t0,
                         "blue_vis_end": info.get("frames", [{}])[0].get("blue_vis") if info.get("frames") else None,
                         "pink_vis_end": info.get("frames", [{}])[0].get("pink_vis") if info.get("frames") else None})
            imgs[(name, int(ep))] = win[0][1]
    d = pd.DataFrame(recs)
    y = (d.outcome == "success").values; s = d.s.fillna(-1).values
    pred = (d.s.fillna(-1) >= cfg["tau"]).values & ~d.undecidable.values
    correct = (pred == y) & ~d.undecidable.values
    rng = np.random.default_rng(0)
    res = {"config": os.path.basename(a.config), "tau": cfg["tau"], "n": len(d), "n_success": int(y.sum()), "n_failure": int((~y).sum()),
           "undecidable": int(d.undecidable.sum()),
           "accuracy": float(correct.mean()), "accuracy_ci95": boot(lambda i: correct[i].mean(), len(d), rng),
           "auroc": auroc(y, s), "auroc_ci95": boot(lambda i: auroc(y[i], s[i]) if 0 < y[i].sum() < len(i) else (_ for _ in ()).throw(ZeroDivisionError()), len(d), rng),
           "confusion": {"tp": int((pred & y).sum()), "fn": int((~pred & y).sum()), "fp": int((pred & ~y).sum()), "tn": int((~pred & ~y).sum())},
           "per_failure_type_recall": d[~y].assign(c=correct[~y]).groupby("failure_mode").c.agg(["mean", "size"]).to_dict(orient="index"),
           "success_recall": float(correct[y].mean()),
           "per_policy": d.assign(c=correct).groupby("policy").agg(n=("c", "size"), accuracy=("c", "mean"),
                                                                     n_failure=("outcome", lambda x: int((x == "failure").sum()))).to_dict(orient="index"),
           "per_device": d.assign(c=correct).groupby("device").agg(n=("c", "size"), accuracy=("c", "mean")).to_dict(orient="index"),
           "runtime_mac_s_per_episode": float(d.seconds.mean()), "comparison": ROBOMETER,
           "errors": d[~correct][["dataset", "episode", "policy", "outcome", "failure_mode", "s", "blue_vis_end", "pink_vis_end"]].to_dict(orient="records")}
    res["pass"] = bool(res["accuracy"] >= 0.95 and res["auroc"] >= 0.95)
    json.dump(res, open(f"{a.out}/g9c_results.json", "w"), indent=1, default=float)
    d.to_csv(f"{a.out}/g9c_scores.csv", index=False)  # small: one row per episode, no frames
    errs = d[~correct]
    if len(errs):
        tiles = []
        for r in errs.itertuples():
            im = cv2.resize(imgs[(r.dataset, r.episode)], (320, 240))
            cv2.putText(im, f"{r.policy} ep{r.episode} {r.outcome}/{r.failure_mode} s={r.s}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles.append(im)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(f"{a.out}/g9c_errors.jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(json.dumps({k: v for k, v in res.items() if k != "per_policy"}, indent=1, default=float))
    print(f"total {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
