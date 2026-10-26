#!/usr/bin/env python3
"""G9c-3 (G9C3_PROTOCOL.md): the frozen G9c judge at the first still moment after the last gripper release, or at the
subtask timeout (= recording end), whichever first. Any hand from the release to the judge frame -> invalid.
Coverage and accuracy on judged episodes are reported separately. One call for the test; --dev for the sanity run.

  python g9c3_test.py --orin ~/robot-data/so101-bowls-policy-rollouts --m1 ~/.cache/huggingface/lerobot/ravediamond \
      --catalog shared/gate/data/rollout_catalog.csv --config g9c_config.json --rule g9c2_rule.json --out results/g9c3 [--dev]
"""
import argparse, json, os, sys, time

import cv2
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g9c_io import Episodes  # noqa: E402
from g9c_judge import episode_score, load_config  # noqa: E402
from g9c_test import auroc, boot  # noqa: E402
from g9c2_endrule import hand_scan, load_states  # noqa: E402

PRIOR = {"g9c": {"coverage": 1.0, "accuracy": 0.8636, "fp": 0}, "g9c2": {"coverage": 59 / 88, "accuracy_on_judged": 0.9661, "fp": 0}}
OLD_LABELS = {("orin", 29): "success", ("orin", 30): "success"}  # before the confirmed relabels (for the side report)


def judge_frame3(E, ep, states, rule, cfg):
    st = states[int(ep)]; T = len(st); fps = E.fps; arm = st[:, :5]
    opened = np.nonzero(st[:, 5] > rule["g_open"])[0]
    t_rel = int(opened[-1]) if len(opened) else 0
    w = max(1, int(round(rule["still_s"] * fps)))
    jf, why = T - 1, "timeout"
    for t in range(t_rel + w, T):
        if np.abs(arm[t] - arm[t - w]).max() <= rule["still_tol_deg"]:
            jf, why = t, "still"; break
    hand, _ = hand_scan_range(E, ep, t_rel, jf, rule, cfg)
    return jf, {"t_rel": t_rel, "why": why, "hand_frame": hand, "n_frames": T, "s_after_release": (jf - t_rel) / fps,
                "s_before_end": (T - 1 - jf) / fps}


def hand_scan_range(E, ep, start, stop, rule, cfg):
    k, px = hand_scan(E, ep, start, rule, cfg)
    return (k if k is not None and k <= stop else None), px


def episodes(a):
    if a.dev:
        D = os.path.expanduser("~/.cache/huggingface/lerobot/ravediamond")
        for ds in ["so101_bowl_blue_in_pink", "so101_bowl_pink_in_blue"]:
            E = Episodes(f"{D}/{ds}")
            for ep in E.ids():
                y = "success" if "blue bowl in the pink" in E.meta.loc[ep, "tasks"][0] else "failure"
                yield ds, f"{D}/{ds}", int(ep), y, "teleop", "dev", (None if y == "success" else "wrong_object")
        return
    lab = pd.read_csv(f"{a.orin}/meta/rollout_labels.csv")
    for r in lab.itertuples():
        yield "orin", a.orin, int(r.episode_index), r.outcome, r.policy, "orin_nx", r.failure_mode if isinstance(r.failure_mode, str) else ("unspecified" if r.outcome == "failure" else None)
    cat = pd.read_csv(a.catalog); cat = cat[(cat.device == "m1") & cat.labeled.astype(bool)]
    for r in cat.itertuples():
        y = "success" if str(r.success) == "True" else "failure"
        yield r.dataset, f"{a.m1}/{r.dataset}", int(r.episode), y, r.policy, "m1", r.failure_mode if isinstance(r.failure_mode, str) else ("unspecified" if y == "failure" else None)


def summarise(d, rng):
    judged = ~d.invalid.values; y = (d.outcome == "success").values; s = d.s.fillna(-1).values
    pred = s >= d.attrs["tau"]; correct = (pred == y)
    yj, sj, cj = y[judged], s[judged], correct[judged]

    def auc_i(i):
        if not 0 < yj[i].sum() < len(i):
            raise ZeroDivisionError
        return auroc(yj[i], sj[i])
    return {"n": len(d), "n_success": int(y.sum()), "n_failure": int((~y).sum()), "judged": int(judged.sum()),
            "coverage": float(judged.mean()),
            "invalid_by_cause": d[~judged].invalid_cause.value_counts().to_dict(),
            "invalid_by_outcome": d[~judged].outcome.value_counts().to_dict(),
            "accuracy_on_judged": float(cj.mean()), "accuracy_on_judged_ci95": boot(lambda i: cj[i].mean(), len(cj), rng),
            "false_positives": int((pred & ~y & judged).sum()), "false_negatives": int((~pred & y & judged).sum()),
            "auroc_on_judged": auroc(yj, sj) if 0 < yj.sum() < len(yj) else None,
            "auroc_on_judged_ci95": boot(auc_i, len(yj), rng) if 0 < yj.sum() < len(yj) else None,
            "per_failure_type_recall_on_judged": d[judged & ~y].assign(c=correct[judged & ~y]).groupby("failure_mode").c.agg(["mean", "size"]).to_dict(orient="index"),
            "judge_frame_why": d.why.value_counts().to_dict(),
            "s_after_release_p50_p90": [float(d.s_after_release.median()), float(d.s_after_release.quantile(.9))],
            "s_before_end_p50_p90": [float(d.s_before_end.median()), float(d.s_before_end.quantile(.9))]}


def main():
    ap = argparse.ArgumentParser()
    for k in ("config", "rule", "out"):
        ap.add_argument(f"--{k}", required=True)
    for k in ("orin", "m1", "catalog"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args(); cfg = load_config(a.config); rule = load_config(a.rule); os.makedirs(a.out, exist_ok=True)
    recs, imgs, cache = [], {}, {}; t_all = time.time()
    for name, root, ep, outcome, policy, device, mode in episodes(a):
        if root not in cache:
            cache[root] = (Episodes(root), load_states(root))
        E, states = cache[root]
        jf, info = judge_frame3(E, ep, states, rule, cfg)
        first, win, _ = E.frames(ep, cfg["max_window_s"], cfg["step_s"], end_frame=jf + 1)
        s, sinfo = episode_score(first, win, cfg)
        cause = "hand" if info["hand_frame"] is not None else ("undecidable" if s is None else None)
        recs.append({"dataset": name, "episode": ep, "device": device, "policy": policy, "outcome": outcome, "failure_mode": mode,
                     "s": s if cause is None else None, "invalid": cause is not None, "invalid_cause": cause, "judge_frame": jf, **info})
        imgs[(name, ep)] = win[0][1]
    d = pd.DataFrame(recs); d.attrs["tau"] = cfg["tau"]
    res = {"mode": "dev sanity (cannot change parameters)" if a.dev else "test", "rule": os.path.basename(a.rule),
           "config": os.path.basename(a.config), "tau": cfg["tau"],
           "result": summarise(d, np.random.default_rng(0)), "prior": PRIOR, "runtime_s": time.time() - t_all}
    r = res["result"]
    res["pass"] = bool(r["accuracy_on_judged"] >= 0.95 and r["false_positives"] == 0 and r["coverage"] >= 0.90)
    if not a.dev:
        old = d.copy()
        for (n, e), v in OLD_LABELS.items():
            old.loc[(old.dataset == n) & (old.episode == e), "outcome"] = v
        old.attrs["tau"] = cfg["tau"]
        res["result_on_old_labels"] = summarise(old, np.random.default_rng(0))
    y = (d.outcome == "success"); pred = d.s.fillna(-1) >= cfg["tau"]
    res["errors_on_judged"] = d[~d.invalid & (pred != y)][["dataset", "episode", "policy", "outcome", "failure_mode", "s", "why", "judge_frame", "n_frames"]].to_dict(orient="records")
    res["invalid"] = d[d.invalid][["dataset", "episode", "policy", "outcome", "invalid_cause", "why", "t_rel", "hand_frame", "judge_frame", "n_frames"]].to_dict(orient="records")
    tag = "dev" if a.dev else "test"
    json.dump(res, open(f"{a.out}/g9c3_{tag}_results.json", "w"), indent=1, default=float)
    d.to_csv(f"{a.out}/g9c3_{tag}_scores.csv", index=False)
    bad = d[d.invalid | (~d.invalid & (pred != y))]
    if len(bad):
        tiles = []
        for b in bad.itertuples():
            im = cv2.resize(imgs[(b.dataset, b.episode)], (320, 240))
            cv2.putText(im, f"{b.policy} ep{b.episode} {b.outcome} s={b.s} {b.invalid_cause or ''} {b.why}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            tiles.append(im)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(f"{a.out}/g9c3_{tag}_errors_and_invalid.jpg", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(json.dumps({k: v for k, v in res.items() if k not in ("errors_on_judged", "invalid")}, indent=1, default=float))
    print("errors on judged:", [(e["policy"], e["episode"], e["outcome"], e["s"]) for e in res["errors_on_judged"]])
    print("invalid:", [(e["policy"], e["episode"], e["outcome"], e["invalid_cause"]) for e in res["invalid"]])


if __name__ == "__main__":
    main()
