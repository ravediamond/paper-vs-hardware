"""WS-107 analysis, PROTOCOL.md §4-§6 as frozen (fa3636b + the §9 D1/D2 text). Reads out/<model>/{embeddings.npz,collapse_checks.json} of the ten
trained models, fdino_{top,wrist}.npz and wrist_mask.csv; runs the §6 checks FIRST (refuses to compute an alignment if one fails); writes
results.json / results.md. Models: E_top_s3072 E_top_s3073 E_wrist_s3072 E_wrist_s3073 P_top_s3072 P_top_s3073 P_wrist_s3072 P_wrist_s3073
E_crop1 E_crop2 (a retrained void under §6.3 takes the same name with seed 3074: pass --replace NAME=DIR).
  python3 -I analyze.py --models DIR --fdino DIR --wrist-mask CSV --out DIR [--boot 2000] [--check-only]"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import align as A

SEEDS = (3072, 3073)
NAMES = [f"{p}_{v}_s{s}" for p in "EP" for v in ("top", "wrist") for s in SEEDS] + ["E_crop1", "E_crop2"]
MIN_STD, MIN_RANK = 0.1, 0.1  # config/lewm_so101.yaml checks, §6.3
BINS = [(0, .25, 12.5), (.25, .5, 37.5), (.5, .75, 62.5), (.75, 1e9, 87.5)]  # values above 100 % go to the last bin


def load(models_dir, names, replace):
    out = {}
    for n in names:
        d = Path(replace.get(n, Path(models_dir) / n))
        z = np.load(d / "embeddings.npz")
        out[n] = {"emb": z["emb"], "ep": z["episode"], "fr": z["frame"], "dir": d}
        cp = d / "collapse_checks.json"
        out[n]["collapse"] = json.loads(cp.read_text()) if cp.exists() else None
    return out


def checks(M, fd):
    """§6.1, §6.3 (and the strict load, recorded by extract_embeddings). Returns (ok, report)."""
    rep, ok = {}, True
    ref = M[NAMES[0]]
    for n, m in M.items():
        same = np.array_equal(m["ep"], ref["ep"]) and np.array_equal(m["fr"], ref["fr"])
        rep[n] = {"shape": list(m["emb"].shape), "frames_match_reference": bool(same), "finite": bool(np.isfinite(m["emb"]).all())}
        ok &= same and rep[n]["finite"] and m["emb"].shape[1] == 192
        c = m["collapse"]
        if c is None:
            rep[n]["collapse"] = "missing"; ok = False; continue
        rep[n]["collapse"] = {"emb_std_mean": c["emb_std_mean"], "eff_rank_frac": c["eff_rank_frac"], "pass_std": c["emb_std_mean"] >= MIN_STD,
                              "pass_rank": c["eff_rank_frac"] >= MIN_RANK, "all_pass_config_incl_idm_arm": c.get("all_pass")}
        rep[n]["void"] = not (rep[n]["collapse"]["pass_std"] and rep[n]["collapse"]["pass_rank"])
    for k, z in fd.items():
        rep[f"F_dino_{k}"] = {"frames_match_reference": bool(np.array_equal(z["ep"], ref["ep"]) and np.array_equal(z["fr"], ref["fr"])), "shape": list(z["emb"].shape)}
        ok &= rep[f"F_dino_{k}"]["frames_match_reference"]
    return ok, rep


def spearman(x, y):
    r = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(*r)[0, 1]) if np.std(r[0]) > 0 and np.std(r[1]) > 0 else float("nan")


def procrustes(fa, fb, ep, fit_eps):
    """Secondary (exploratory): orthogonal Q fitted on `fit_eps`, residual ||Q a - b|| / ||b|| on the other episodes (raw, mean-centred embeddings)."""
    fit = np.isin(ep, fit_eps)
    a, b = fa - fa[fit].mean(0), fb - fb[fit].mean(0)
    u, _, vt = np.linalg.svd(a[fit].T @ b[fit]); q = u @ vt
    t = ~fit
    return float(np.linalg.norm(a[t] @ q - b[t]) / np.linalg.norm(b[t]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True); ap.add_argument("--fdino"); ap.add_argument("--wrist-mask"); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--replace", nargs="*", default=[]); ap.add_argument("--split"); ap.add_argument("--skip-p", action="store_true", help="the P models were not trained: no alignment(4), B2 not evaluable")
    a = ap.parse_args()
    replace = dict(r.split("=") for r in a.replace)
    names = [n for n in NAMES if not (a.skip_p and n.startswith("P_"))]
    M = load(a.models, names, replace)
    fd = {}
    if a.fdino:
        for k in ("top", "wrist"):
            z = np.load(Path(a.fdino) / f"fdino_{k}.npz"); fd[k] = {"emb": z["emb"], "ep": z["episode"], "fr": z["frame"]}
    ok, rep = checks(M, fd)
    res = {"checks": rep, "checks_ok": bool(ok)}
    ref = M[NAMES[0]]; ep_all, fr_all = ref["ep"], ref["fr"]
    elig = fr_all >= 12  # 5 fps index >= 2
    res["N_frames_5fps"] = int(len(ep_all)); res["N_eligible"] = int(elig.sum()); res["n_episodes"] = int(len(np.unique(ep_all)))
    res["chance_10nn"] = 10 / (int(elig.sum()) - int(elig.sum()) / res["n_episodes"])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if not ok or a.check_only:
        (Path(a.out) / "results.json").write_text(json.dumps(res, indent=1)); print("checks:", ok); return 0 if ok else 2
    void = [n for n, r in rep.items() if r.get("void")]
    res["void_models"] = void
    ep = ep_all[elig]; ids = np.unique(ep); ep_ids = np.searchsorted(ids, ep); n_ep = len(ids)
    S = {n: A.Space(M[n]["emb"], elig, ep_ids) for n in names}
    if fd: S["Fd_top"], S["Fd_wrist"] = A.Space(fd["top"]["emb"], elig, ep_ids), A.Space(fd["wrist"]["emb"], elig, ep_ids)
    pairs = {  # §4 seed pairing; each entry: list of (A, B) seed-matched pairs
        "a1_crop": [("E_crop1", "E_crop2")], "a2_top_topprime": [("E_top_s3072", "E_top_s3073")],
        "a3_top_wrist": [(f"E_top_s{s}", f"E_wrist_s{s}") for s in SEEDS]}
    if not a.skip_p: pairs["a4_Ptop_Pwrist"] = [(f"P_top_s{s}", f"P_wrist_s{s}") for s in SEEDS]
    if fd: pairs["a5_fdino"] = [("Fd_top", "Fd_wrist")]
    cov = None
    if a.wrist_mask:
        df = pd.read_csv(a.wrist_mask).set_index(["episode", "frame"])
        cov = df["coverage"].reindex(list(zip(ep_all[elig], fr_all[elig]))).to_numpy()
        res["b3_excluded_episodes_area0_zero"] = sorted({int(e) for e in df.index.get_level_values(0).unique() if df.loc[e].area0.iloc[0] == 0})
    fr_el = fr_all[elig]

    def evaluate(w):
        """w = episode multiplicities of the resample (all ones = the point value on all eligible frames)."""
        need = {m for ps in pairs.values() for p in ps for m in p}
        T = {m: A.taken(S[m], w) for m in need}
        rw = w[ep_ids]  # copies of each row in the resample
        out, row = {}, {}
        for k, ps in pairs.items():
            sc = np.mean([A.row_scores(T[x], T[y]) for x, y in ps], axis=0)
            out[k], row[k] = float((sc * rw).sum() / rw.sum()), sc
        out["B1_a3_minus_a1"] = out["a3_top_wrist"] - out["a1_crop"]; out["B2_a3_minus_a4"] = out["a3_top_wrist"] - out["a4_Ptop_Pwrist"] if "a4_Ptop_Pwrist" in out else float("nan")
        out["O5_a2_minus_a1"] = out["a2_top_topprime"] - out["a1_crop"]
        if cov is not None:
            s3 = np.repeat(row["a3_top_wrist"], rw); c = np.repeat(cov, rw); v = ~np.isnan(c)
            b = np.full(len(c), -1)
            for i, (lo, hi, _) in enumerate(BINS): b[v & (c >= lo) & (c < hi)] = i
            for i in range(4): out[f"b3_bin{i}_mean"] = float(s3[b == i].mean()) if (b == i).any() else float("nan"); out[f"b3_bin{i}_n"] = int((b == i).sum())
            mid = np.array([m for _, _, m in BINS]); sel = b >= 0
            out["b3_spearman"] = spearman(s3[sel], mid[b[sel]]) if sel.any() else float("nan")
        return out

    point = evaluate(np.ones(n_ep, int))
    res["point"] = point
    B = A.episode_bootstrap(evaluate, n_ep, n_boot=a.boot, seed=0)
    res["ci95"] = {k: A.ci95(v[~np.isnan(v)]) if (~np.isnan(v)).any() else None for k, v in B.items()}
    res["n_boot"] = a.boot
    np.savez_compressed(Path(a.out) / "bootstrap_draws.npz", **B)
    o5 = A.verdict(res["ci95"]["O5_a2_minus_a1"])
    res["O5_anchors_separate"] = o5 == "pass"
    b1 = A.verdict(res["ci95"]["B1_a3_minus_a1"]); b2 = A.verdict(res["ci95"]["B2_a3_minus_a4"]) if res["ci95"].get("B2_a3_minus_a4") else "not evaluable (P models not trained)"
    v1 = [n for n in void if n.startswith(("E_top", "E_wrist", "E_crop"))]; v2 = [n for n in void if n.startswith(("E_top", "E_wrist", "P_"))]
    res["verdict"] = {"B1": ("void (§6.3 / O8: %s failed the collapse check)" % ", ".join(v1)) if v1 else (b1 if res["O5_anchors_separate"] else "not proven"),
                      "B2": ("void (§6.3 / O8: %s)" % ", ".join(v2)) if v2 else (b2 if res["O5_anchors_separate"] else "not proven"),
                      "B1_ci_only_exploratory": b1, "B2_ci_only_exploratory": b2, "O5_exploratory": o5, "voided_models": void,
                      "note": "numbers of a void model are exploratory: the model failed a pre-registered check"}
    if cov is not None:
        sp = res["ci95"].get("b3_spearman"); nb = [point.get(f"b3_bin{i}_n", 0) for i in range(4)]
        res["B3"] = {"descriptive": True, "bins_n": nb, "evaluable": bool(all(n > 0 for n in nb)), "spearman": point.get("b3_spearman"), "spearman_ci95": sp,
                     "bin_means": [point.get(f"b3_bin{i}_mean") for i in range(4)]}
    if a.split:  # secondary, exploratory
        eps = json.loads(Path(a.split).read_text())["val_episodes"]; fit = eps[:12]
        res["procrustes_exploratory"] = {"a3_E_top_E_wrist": float(np.mean([procrustes(M[f"E_top_s{s}"]["emb"], M[f"E_wrist_s{s}"]["emb"], ep_all, fit) for s in SEEDS]))}
        if not a.skip_p: res["procrustes_exploratory"]["a4_P_top_P_wrist"] = float(np.mean([procrustes(M[f"P_top_s{s}"]["emb"], M[f"P_wrist_s{s}"]["emb"], ep_all, fit) for s in SEEDS]))
    (Path(a.out) / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"point": point, "ci95": {k: res["ci95"][k] for k in res["ci95"] if not k.startswith("b3_bin")}, "verdict": res["verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
