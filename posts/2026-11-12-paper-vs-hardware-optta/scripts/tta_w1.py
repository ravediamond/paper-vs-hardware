"""WS-196 H3/H4: prediction-only TTA of W1's last predictor block (PROTOCOL.md section 3, 5). h4 run, h3 implemented.
  h4 : wall time of one Adam step (lr 5e-5, buffer 5) on W1's last predictor block, median of 100 steps.
  h3 : monitor false alarms / recall under synthetic darkening with and without adaptation, plus shuffled null.
       Input: per-episode npz of strict-load W1 latents (clean and darkened) written by extract_w1_h3.py in the W1 image
       (run_h3_extract.sh); the analysis itself needs no ViT (predictor / action encoder / pred_proj only, FP32).
  selftest : synthetic checks of the H3 logic (hand-computed scores, causality, reset, shuffled null), CPU.
Weights: W1 `weights_final.pt` state dict; predictor / action_encoder / pred_proj keys loaded strictly (no ViT needed for H4).
Adapted parameters (same scope as H2): last predictor block, transformer.norm, pred_proj (BatchNorm in eval mode).
Run (GPU, only when no other model is served; see notes):
  WS=WS-196 timeout 300 scripts/memcap.sh -m 8G ~/venvs/optta/bin/python projects/optta/tta_w1.py h4"""
import argparse, json, os, sys, time
import numpy as np, torch
from torch import nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor", "lewm")); sys.path.insert(0, os.path.join(HERE, "vendor", "swm_min"))
from module import ARPredictor, Embedder, MLP  # noqa: E402
W1 = os.path.expanduser("~/wm_data/weights_final.pt")


class W1Pred(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192, depth=4, heads=8, mlp_dim=1024,
                                     dim_head=64, dropout=0.1, emb_dropout=0.0)
        self.action_encoder = Embedder(input_dim=6, smoothed_dim=6, emb_dim=192)
        self.pred_proj = MLP(192, 768, 192, norm_fn=nn.BatchNorm1d)

    def predict(self, emb, act_emb):
        p = self.predictor(emb, act_emb)
        b = emb.size(0)
        return self.pred_proj(p.reshape(-1, p.size(-1))).reshape(b, -1, p.size(-1))


def load(dev):
    sd = torch.load(W1, map_location="cpu", weights_only=False)["state_dict"]
    m = W1Pred()
    sub = {k: v for k, v in sd.items() if k.split(".")[0] in ("predictor", "action_encoder", "pred_proj")}
    m.load_state_dict(sub, strict=True)
    return m.to(dev).eval().requires_grad_(False), len(sub)


def adapt_params(m):
    return (list(m.predictor.transformer.layers[-1].parameters()) + list(m.predictor.transformer.norm.parameters())
            + list(m.pred_proj.parameters()))


def step(m, params, ctx, act, tgt, lr=5e-5):
    for q in params:
        q.requires_grad_(True)
    p = m.predict(ctx, m.action_encoder(act))[:, -1]
    loss = ((p - tgt) ** 2).mean()
    g = torch.autograd.grad(loss, params)
    opt = torch.optim.Adam(params, lr=lr)
    for q, gq in zip(params, g):
        q.grad = gq
    opt.step()
    for q in params:
        q.requires_grad_(False); q.grad = None
    return float(loss)


def h4(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, nkeys = load(dev); params = adapt_params(m)
    g = torch.Generator().manual_seed(0)
    ctx = torch.randn(5, 3, 192, generator=g).to(dev); act = torch.randn(5, 3, 6, generator=g).to(dev)
    tgt = torch.randn(5, 192, generator=g).to(dev)
    for _ in range(10):
        step(m, params, ctx, act, tgt)
    t = []
    for _ in range(100):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter(); step(m, params, ctx, act, tgt)
        if dev == "cuda":
            torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1e3)
    r = dict(device=dev, dtype="fp32 (TF32 off)", n_loaded_keys=nkeys, n_adapted_params=int(sum(q.numel() for q in params)),
             median_ms=round(float(np.median(t)), 2), p90_ms=round(float(np.percentile(t, 90)), 2), n_steps=100, bar_ms=20,
             bar_met=bool(np.median(t) <= 20), inputs="random latents (timing only, cost is data independent), buffer 5, context 3",
             power=a.label)
    json.dump(r, open(a.out, "w"), indent=1); print(json.dumps(r, indent=1))


# ----------------------------------------------------------------------------------------------------------------------
# H3 (PROTOCOL.md sections 2, 3, 5). Monitor = the W1-A L2 surprise scorer of shared/gate/data/2026-10-19_w1_screening:
# anchor every STRIDE frames from frame 2, context = anchor and 2 previous latents with recorded actions, imagined to k=25;
# anchor score = mean over k in CHECK_K of ||z[a+k] - zhat_k||; an anchor alarms at score >= TAU (fitted on unshifted data,
# 17.3143 from w1a_summary.json g16_set.l2.tau, NOT refitted); an episode is flagged if any anchor alarms.
# Interpretations (non-bar choices, stated in the result): false-alarm rate = flagged share of the 44 rollout successes
# (PROTOCOL section 8 proposal); recall = share of the 25 failures with an alarm at an anchor whose window reaches the labelled
# onset (a + 25 >= onset), episode-level "any alarm" recall reported next to it; adaptation tick every CHUNK = 7 frames
# (238 ms at 30 fps), one Adam step (lr 5e-5, re-initialised each step) on the last 5 transitions observed up to that tick,
# weights reset at the start of every episode; shuffled null: each buffer target replaced by the latent of another random
# observed frame of the same episode.
STRIDE, HORIZON, CHECK_K, CHUNK, BUF, CTX = 25, 25, (15, 20, 25), 7, 5, 3
TAU = 17.3143
A_MEAN = (3.823351, -32.272697, 30.039783, 74.51135, -10.073078, 5.0724087)
A_STD = (17.493738, 42.95796, 40.683086, 9.202332, 7.6247225, 10.140567)
BARS = dict(fa_drop_min=0.50, recall_ratio_min=0.80, shuffle_drop_max=0.10)


def anchors_of(z, end):
    out = []
    for a in range(2, end - HORIZON, STRIDE):
        if np.isnan(z[a - 2:a + 1]).any() or np.isnan(z[a + 1:a + HORIZON + 1]).any():
            continue
        out.append(a)
    return out


@torch.no_grad()
def imagine(m, z, A, anchors):
    """Batched W1LiveMonitor.imagine_chunk: returns (B, HORIZON, 192) predictions for each anchor (FP32)."""
    mean, std = torch.tensor(A_MEAN), torch.tensor(A_STD)
    ctx = torch.stack([torch.from_numpy(z[a - 2:a + 1]) for a in anchors]).float()
    dev = next(m.parameters()).device if any(True for _ in m.parameters()) else "cpu"
    acts = torch.stack([torch.from_numpy(A[a - 2:a + HORIZON]) for a in anchors]).float()
    a_emb = m.action_encoder(((acts - mean) / std).to(dev))
    ctx = ctx.to(dev); preds = []
    for k in range(HORIZON):
        nxt = m.predict(ctx[:, -CTX:], a_emb[:, k:k + CTX])[:, -1:]
        ctx = torch.cat([ctx, nxt], 1); preds.append(nxt[:, 0])
    return torch.stack(preds, 1).cpu()


def anchor_scores(zhat, z, anchors):
    """(B,) mean over CHECK_K of the L2 distance between the imagined and the real latent at a+k."""
    return np.array([np.mean([np.linalg.norm(z[a + k] - zhat[i, k - 1].numpy()) for k in CHECK_K])
                     for i, a in enumerate(anchors)])


def run_episode(m, pristine, z, A, end, mode, seed=0, step_fn=None):
    """Scores of every anchor of one episode. mode: frozen | adapt | shuffle. Causal: the anchor at frame a is imagined with
    weights that have seen only latents up to frame a. Weights are reset first and are left adapted (caller resets)."""
    step_fn = step_fn or step
    m.load_state_dict(pristine)
    anchors = anchors_of(z, end)
    if not anchors:
        return anchors, np.zeros(0), 0
    if mode == "frozen":
        return anchors, anchor_scores(imagine(m, z, A, anchors), z, anchors), 0
    params = adapt_params(m); rng = np.random.default_rng(seed)
    mean, std = torch.tensor(A_MEAN), torch.tensor(A_STD)
    scores, n_steps, aset = [], 0, set(anchors)
    if 2 in aset:  # anchor 2 precedes the first transition: pristine weights
        scores.append(float(anchor_scores(imagine(m, z, A, [2]), z, [2])[0]))
    for t in range(CTX, end):
        if t % CHUNK == 0 and t >= CTX + BUF - 1:
            tr = [i for i in range(t - BUF + 1, t + 1) if not np.isnan(z[i - CTX:i + 1]).any()]
            if len(tr) == BUF:
                ctx = torch.stack([torch.from_numpy(z[i - CTX:i]) for i in tr]).float()
                act = (torch.stack([torch.from_numpy(A[i - CTX:i]) for i in tr]).float() - mean) / std
                if mode == "shuffle":
                    pick = [int(rng.choice([j for j in range(CTX, t + 1) if j != i and not np.isnan(z[j]).any()])) for i in tr]
                else:
                    pick = tr
                tgt = torch.stack([torch.from_numpy(z[j]) for j in pick]).float()
                step_fn(m, params, ctx, act, tgt); n_steps += 1
        if t in aset:
            scores.append(float(anchor_scores(imagine(m, z, A, [t]), z, [t])[0]))
    return anchors, np.array(scores), n_steps


def episode_stats(anchors, sc, onset, tau):
    al = [a for a, s in zip(anchors, sc) if s >= tau]
    return dict(score=float(sc.max()) if len(sc) else 0.0, flagged=bool(al), n_alarm=len(al),
                onset_hit=bool(onset >= 0 and any(a + HORIZON >= onset for a in al)))


def summarize(rows, tau):
    """rows: list of dicts with outcome, score, flagged, onset_hit -> FA rate on successes, recalls on failures."""
    s = [r for r in rows if r["outcome"] == "success"]; f = [r for r in rows if r["outcome"] == "failure"]
    return dict(fa=sum(r["flagged"] for r in s) / max(len(s), 1), n_success=len(s), recall_onset=sum(r["onset_hit"] for r in f) / max(len(f), 1),
                recall_any=sum(r["flagged"] for r in f) / max(len(f), 1), n_failure=len(f))


def verdict(c0, c1, cs, bars=BARS):
    """c0 frozen, c1 adapted, cs shuffled summaries under the same corruption. pass / fail / not proven."""
    if c0["fa"] == 0 or c0["recall_onset"] == 0:
        return "not proven", "frozen false alarms or frozen onset recall is 0 under the shift: the ratios are undefined"
    drop = (c0["fa"] - c1["fa"]) / c0["fa"]; ratio = c1["recall_onset"] / c0["recall_onset"]; sdrop = (c0["fa"] - cs["fa"]) / c0["fa"]
    ok = (drop >= bars["fa_drop_min"], ratio >= bars["recall_ratio_min"], sdrop <= bars["shuffle_drop_max"])
    why = f"FA drop {drop:.3f} (>= {bars['fa_drop_min']}), recall ratio {ratio:.3f} (>= {bars['recall_ratio_min']}), shuffled FA drop {sdrop:.3f} (<= {bars['shuffle_drop_max']})"
    return ("pass" if all(ok) else "fail"), why


def auroc(y, s):
    pos = [b for a, b in zip(y, s) if a]; neg = [b for a, b in zip(y, s) if not a]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if pos and neg else None


def boot(cells, B=2000, seed=0):
    """Episode-level bootstrap (stratified by outcome, 2000 resamples, seed 0) of FA drop, recall ratio, shuffled drop.
    cells[arm] = list of per-episode dicts in a common episode order."""
    rng = np.random.default_rng(seed)
    out = [r["outcome"] for r in cells["frozen"]]
    si = [i for i, o in enumerate(out) if o == "success"]; fi = [i for i, o in enumerate(out) if o == "failure"]
    res = {k: [] for k in ("fa_frozen", "fa_adapt", "fa_drop", "recall_frozen", "recall_adapt", "recall_ratio", "shuffle_drop")}
    for _ in range(B):
        a = rng.choice(si, len(si)); b = rng.choice(fi, len(fi))
        v = {}
        for arm in ("frozen", "adapt", "shuffle"):
            v[arm] = (np.mean([cells[arm][i]["flagged"] for i in a]), np.mean([cells[arm][i]["onset_hit"] for i in b]))
        res["fa_frozen"].append(v["frozen"][0]); res["fa_adapt"].append(v["adapt"][0]); res["recall_frozen"].append(v["frozen"][1])
        res["recall_adapt"].append(v["adapt"][1])
        res["fa_drop"].append((v["frozen"][0] - v["adapt"][0]) / v["frozen"][0] if v["frozen"][0] > 0 else np.nan)
        res["recall_ratio"].append(v["adapt"][1] / v["frozen"][1] if v["frozen"][1] > 0 else np.nan)
        res["shuffle_drop"].append((v["frozen"][0] - v["shuffle"][0]) / v["frozen"][0] if v["frozen"][0] > 0 else np.nan)
    def ci(x):
        ok = ~np.isnan(x)
        return [round(float(np.percentile(np.asarray(x)[ok], q)), 4) if ok.any() else None for q in (2.5, 97.5)] + [int((~ok).sum())]
    return {k: ci(np.array(x, float)) for k, x in res.items()}  # [lo, hi, resamples with undefined ratio]


def load_w1_pred(dev="cpu"):
    return load(dev)[0]


def h3(a):
    import glob
    torch.set_num_threads(a.threads)
    files = sorted(glob.glob(os.path.join(a.lat, "ep*.npz")))
    eps = {int(os.path.basename(f)[2:5]): np.load(f) for f in files}
    print(f"{len(eps)} episodes in {a.lat}", flush=True)
    if len(eps) != 69 and not a.allow_partial:
        sys.exit(f"expected 69 episodes, found {len(eps)} (extraction incomplete; --allow-partial for a labelled partial run)")
    dev = "cpu"  # the analysis is small; CPU keeps the GPU lock free
    m, nkeys = load(dev); pristine = {k: v.clone() for k, v in m.state_dict().items()}
    ref = json.load(open(os.path.join(HERE, "..", "..", "shared/gate/data/2026-10-19_w1_screening/w1a_summary.json")))
    refsc = {e["episode_index"]: e for e in ref["episodes"]}
    tau = ref["g16_set"]["l2"]["tau"]; assert abs(tau - TAU) < 1e-3
    cells = {c: {arm: [] for arm in ("frozen", "adapt", "shuffle")} for c in ("clean", "dark")}
    per_anchor = {}; t0 = time.time()
    cdir = os.path.expanduser("~/scratch/WS-196/h3_cache"); os.makedirs(cdir, exist_ok=True)
    for n, (ep, d) in enumerate(eps.items()):
        if a.shard and n % a.shard[1] != a.shard[0]:
            continue
        cf = os.path.join(cdir, f"ep{ep:03d}.json")
        if os.path.exists(cf):
            c = json.load(open(cf))
            for cond in cells:
                for arm, row in c["cells"][cond].items():
                    cells[cond][arm].append(row)
            per_anchor.update(c["per_anchor"]); continue
        z0, z1, A, end, onset, oc = d["z_clean"], d["z_dark"], d["actions"], int(d["end"]), int(d["onset"]), str(d["outcome"])
        mine = {c: {} for c in cells}; pa = {}
        for cond, z in (("clean", z0), ("dark", z1)):
            for arm in (("frozen",) if cond == "clean" else ("frozen", "adapt", "shuffle")):
                an, sc, ns = run_episode(m, pristine, z, A, end, arm, seed=ep)
                mine[cond][arm] = dict(episode=ep, outcome=oc, onset=onset, steps=ns, **episode_stats(an, sc, onset, tau))
                cells[cond][arm].append(mine[cond][arm])
                pa[f"{ep}/{cond}/{arm}"] = [an, [round(float(x), 3) for x in sc]]
        per_anchor.update(pa)
        json.dump(dict(cells=mine, per_anchor=pa), open(cf, "w"), default=float)
        print(f"[{n + 1}/{len(eps)}] ep {ep} {oc} " + " ".join(f"{c}.{r}={mine[c][r]['score']:.1f}" for c in mine for r in mine[c])
              + f" steps={mine['dark']['adapt']['steps']} {time.time() - t0:.0f}s", flush=True)
    if a.shard:
        print("shard done"); return
    for cond in cells:
        for arm in cells[cond]:
            cells[cond][arm].sort(key=lambda r: r["episode"])
    # check 6.2: frozen W1 on unshifted data reproduces the W1-A scorer
    g16 = [r for r in cells["clean"]["frozen"] if refsc[r["episode"]]["in_g16_set"]]
    ys = [int(r["outcome"] == "failure") for r in g16]
    chk = dict(auroc_g16_now=auroc(ys, [r["score"] for r in g16]), auroc_g16_ref=ref["g16_set"]["l2"]["auroc"],
               max_abs_score_diff=float(max(abs(r["score"] - refsc[r["episode"]]["score_l2"]) for r in cells["clean"]["frozen"])),
               n_g16=len(g16), tolerance_auroc=0.02)
    chk["reproduced"] = bool(chk["auroc_g16_now"] is not None and abs(chk["auroc_g16_now"] - chk["auroc_g16_ref"]) <= chk["tolerance_auroc"])
    summ = {c: {arm: summarize(cells[c][arm], tau) for arm in cells[c] if cells[c][arm]} for c in cells}
    v, why = verdict(summ["dark"]["frozen"], summ["dark"]["adapt"], summ["dark"]["shuffle"])
    if not chk["reproduced"]:
        v, why = "not proven", "check 6.2 failed (frozen W1 does not reproduce the W1-A scorer); " + why
    r = dict(protocol="PROTOCOL.md section 5 H3", tau=tau, tau_note="fitted on unshifted data (W1-A, 50 episodes), not refitted",
             shift="clip(x)^1.5 * 0.7 on the [0,1] frame, both cameras", chunk_frames=CHUNK, buffer=BUF, lr=5e-5, n_loaded_keys=nkeys,
             n_episodes=len(eps), check_6_2=chk, summary=summ, bars=BARS, verdict=v, verdict_detail=why,
             bootstrap_95ci_dark=boot({arm: cells["dark"][arm] for arm in ("frozen", "adapt", "shuffle")}) if len(eps) == 69 or a.allow_partial else None,
             episodes={c: cells[c] for c in cells}, per_anchor=per_anchor, partial=len(eps) != 69, device=dev)
    json.dump(r, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: r[k] for k in ("check_6_2", "summary", "verdict", "verdict_detail", "bootstrap_95ci_dark")}, indent=1, default=float))


def selftest():
    """Synthetic CPU checks of the H3 logic (PROTOCOL check 6.3 plus causality, reset, shuffled null)."""
    torch.manual_seed(0); np.random.seed(0)
    # 1. hand-computed scores with a persistence stub: zhat_k = z[a], z[t] = s*t*e0 -> score(a) = s*mean(15,20,25) = 20 s
    class Stub(nn.Module):
        def __init__(s):
            super().__init__(); s.p = nn.Parameter(torch.zeros(1)); s.action_encoder = lambda x: x
        def predict(s, emb, act):
            return emb
    z = np.zeros((300, 192), np.float32); z[:, 0] = np.arange(300) * 1.0
    A = np.zeros((300, 6), np.float32)
    an = anchors_of(z, 300)
    assert an == [2, 27, 52, 77, 102, 127, 152, 177, 202, 227, 252], an  # range(2, 275, 25) by hand
    z[:, 0] *= 1.0
    sc = anchor_scores(imagine(Stub(), z, A, an), z, an)
    assert np.allclose(sc, 20.0, atol=1e-4), sc          # s = 1: every anchor scores 20 by hand
    st = episode_stats(an, sc, onset=100, tau=17.3143)   # 20 >= tau at all 11 anchors
    assert st["flagged"] and st["n_alarm"] == 11 and st["onset_hit"]
    z2 = z.copy(); z2[:, 0] *= 0.5                         # s = 0.5: score 10 < tau, no alarm
    assert not episode_stats(an, anchor_scores(imagine(Stub(), z2, A, an), z2, an), 100, 17.3143)["flagged"]
    z3 = z.copy(); z3[:, 0] = np.where(np.arange(300) < 160, np.arange(300) * 0.5, 80 + (np.arange(300) - 160) * 1.0)  # alarm only late
    sc3 = anchor_scores(imagine(Stub(), z3, A, an), z3, an); al = [a for a, s in zip(an, sc3) if s >= 17.3143]
    st3 = episode_stats(an, sc3, onset=250, tau=17.3143)
    assert st3["onset_hit"] == any(a + 25 >= 250 for a in al)
    # 2. summarize / verdict
    rows = [dict(outcome="success", flagged=f, onset_hit=False) for f in [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]] + \
           [dict(outcome="failure", flagged=True, onset_hit=h) for h in [1, 1, 1, 1, 0]]
    s0 = summarize(rows, 0); assert abs(s0["fa"] - 0.4) < 1e-9 and abs(s0["recall_onset"] - 0.8) < 1e-9
    s1 = dict(fa=0.2, recall_onset=0.8); s2 = dict(fa=0.38, recall_onset=0.8)
    assert verdict(s0, s1, s2)[0] == "pass" and verdict(s0, dict(fa=0.3, recall_onset=0.8), s2)[0] == "fail"
    assert verdict(s0, s1, dict(fa=0.2, recall_onset=0.8))[0] == "fail"           # shuffled null reduces FA too
    assert verdict(dict(fa=0.0, recall_onset=0.8), s1, s2)[0] == "not proven"
    # 3. adaptation logic on the real W1 predictor with random latents
    if not os.path.exists(W1):
        print("W1 weights absent: skipping part 3"); return
    m, _ = load("cpu"); pristine = {k: v.clone() for k, v in m.state_dict().items()}
    T = 200; zr = (torch.randn(T, 192) * 2).numpy(); Ar = (torch.randn(T, 6) * 20).numpy()
    an, sf, n0 = run_episode(m, pristine, zr, Ar, T, "frozen")
    assert n0 == 0 and all(torch.equal(pristine[k], v) for k, v in m.state_dict().items())
    an, sa, n1 = run_episode(m, pristine, zr, Ar, T, "adapt")
    assert n1 == len([t for t in range(CTX, T) if t % CHUNK == 0 and t >= CTX + BUF - 1]), n1
    assert any(not torch.equal(pristine[k], v) for k, v in m.state_dict().items()) and abs(sa[0] - sf[0]) < 1e-5  # anchor 2: no data yet
    assert not np.allclose(sa[2:], sf[2:])
    changed = {k for k, v in m.state_dict().items() if not torch.equal(pristine[k], v)}
    allowed = ("predictor.transformer.layers.3.", "predictor.transformer.norm.", "pred_proj.")
    assert changed and all(k.startswith(allowed) for k in changed), changed   # only the adapted scope moved
    # causality: changing latents after frame a leaves the adapted score of the anchors <= a - 25 ... unchanged
    zc = zr.copy(); zc[120:] += 5.0
    _, sa2, _ = run_episode(m, pristine, zc, Ar, T, "adapt")
    k_ok = [i for i, a in enumerate(an) if a + HORIZON < 120]
    assert k_ok and np.allclose(sa[k_ok], sa2[k_ok], atol=1e-5), "adaptation used the future"
    # reset: a second run from the same pristine weights is identical
    _, sa3, _ = run_episode(m, pristine, zr, Ar, T, "adapt"); assert np.allclose(sa, sa3, atol=1e-6)
    # shuffled null never feeds the true target: capture targets
    seen = []
    def spy(mm, params, ctx, act, tgt, lr=5e-5):
        seen.append((ctx.clone(), tgt.clone())); return step(mm, params, ctx, act, tgt, lr)
    run_episode(m, pristine, zr, Ar, T, "shuffle", step_fn=spy)
    zt = torch.from_numpy(zr)
    for ctx, tgt in seen:
        for c, t in zip(ctx, tgt):
            i = [j for j in range(CTX, T) if torch.equal(zt[j - CTX:j], c)][0]
            assert not torch.equal(zt[i], t), "shuffled transition kept its true target"
    # boot runs
    cells = {arm: [dict(outcome=("success" if i < 6 else "failure"), flagged=(i % 2 == 0), onset_hit=(i % 3 != 0)) for i in range(10)] for arm in ("frozen", "adapt", "shuffle")}
    assert "fa_drop" in boot(cells, B=50)
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["h4", "h3", "selftest"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "h4.json"))
    ap.add_argument("--lat", default=os.path.expanduser("~/scratch/WS-196/h3_lat"))
    ap.add_argument("--label", default="25 W quiet profile")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", default="", help="k/n: process episodes with index %% n == k into the cache and stop")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    a.shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
    if a.cmd == "h3" and a.out.endswith("h4.json"):
        a.out = os.path.join(HERE, "results", "h3.json")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if a.cmd == "h4":
        h4(a)
    elif a.cmd == "selftest":
        selftest()
    else:
        h3(a)
