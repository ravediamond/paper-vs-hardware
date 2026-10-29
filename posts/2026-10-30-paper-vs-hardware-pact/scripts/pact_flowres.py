#!/usr/bin/env python3
"""WS-059 gate B (projects/pact-vla/PROTOCOL.md): PACT-VLA's pre-commit FlowRes selection on SmolVLA, offline.

Per G13b boundary b at frame t (s = 25, the 20 G13b episodes): N = 4 candidates from o_t (seeds 4b … 4b + 3, single
calls, 10 Euler steps); FlowRes E = Σ_τ Σ_{j∈[k, k+s)} ‖v̂_j(x_τ, τ | o) − u_j‖² with τ ∈ {0.2, 0.4, 0.6, 0.8}, one
ε_ver per boundary (seed 10 000 + b), v̂ from SmolVLA's own denoise_step on the scoring observation's cached prefix,
u = ε_ver − A (normalised action space, all 6 real dims). Scoring observation: o_t (Old-FlowRes) and o_{t+k−1} (PACT)
for each commit offset k. Errors: mean |Δ| in degrees over the 5 arm joints between chunk[k : k + s] and the recorded
a[t + k : t + k + s]. Also FlowRes of the recorded future (the sanity check) and a shuffled-observation smoke.

    IMAGE=lerobot-jetson:rc LEROBOT_SRC=image shared/gate/jetson_run.sh --python scripts/pact_flowres.py \\
        --policy-path ravediamond/smolvla_2026-09-24_14-34-40 --revision c1d76f6 --device cuda \\
        [--episodes 2 --checks] [--k 8,12] --out outputs/pact
Writes <out>/rows.jsonl (one row per boundary and k, resumable per episode).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_g13b_rtc_offline as G  # noqa: E402  LeRobotSmolVLA, choose_episodes

S, N_CAND, TAUS, ARM = 25, 4, (0.2, 0.4, 0.6, 0.8), slice(0, 5)
G13B = ROOT / "shared/gate/data/2026-10-19_g13b_offline/g13b_summary.json"


class Pact(G.LeRobotSmolVLA):
    def batch(self, obs):
        b = {k: torch.from_numpy(obs[k]).permute(2, 0, 1).float().unsqueeze(0) / 255.0 for k in self.image_keys}
        b["observation.state"] = torch.as_tensor(obs["state"], dtype=torch.float32).unsqueeze(0)
        b["task"] = [obs["task"]]
        return self.policy._prepare_batch(self.pre(b))

    @torch.no_grad()
    def prefix(self, obs):
        """SmolVLA's own prefix path (as sample_actions): KV cache of images + language + state for one observation."""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        pol, mdl = self.policy, self.policy.model
        b = self.batch(obs)
        images, img_masks = pol.prepare_images(b)
        state = pol.prepare_state(b)
        lt, lm = b["observation.language.tokens"], b["observation.language.attention_mask"]
        embs, pad, att = mdl.embed_prefix(images, img_masks, lt, lm, state=state)
        _, kv = mdl.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad, att),
                                             position_ids=torch.cumsum(pad, dim=1) - 1, past_key_values=None,
                                             inputs_embeds=[embs, None], use_cache=mdl.config.use_cache)
        return pad, kv

    @torch.no_grad()
    def velocity(self, pre, x, tau):
        pad, kv = pre
        v = self.policy.model.denoise_step(prefix_pad_masks=pad, past_key_values=kv, x_t=x.to(self.device),
                                           timestep=torch.tensor([tau], device=self.device, dtype=torch.float32))
        return v.float().cpu()

    def flowres(self, pre, cands, eps, k, real):
        """E per candidate (cands: (C, H, A) normalised, padded to max_action_dim)."""
        E = []
        for A in cands:
            r = 0.0
            for tau in TAUS:
                x = ((1 - tau) * A + tau * eps).unsqueeze(0)
                v = self.velocity(pre, x, tau)[0]
                u = eps - A
                r += float(((v[k:k + S, :real] - u[k:k + S, :real]) ** 2).sum())
            E.append(r)
        return np.array(E)


def affine_of(m, real):
    """The postprocessor's unnormalisation is affine per dim: robot = scale * model + offset."""
    H = m.horizon
    z, o = torch.zeros(H, real), torch.ones(H, real)
    off = m.to_robot(z)[0, :real]; sc = m.to_robot(o)[0, :real] - off
    return sc, off


def err(chunk_deg, rec, k):
    seg = rec[k:k + S, ARM]
    return float(np.mean(np.abs(chunk_deg[k:k + len(seg), ARM] - seg))) if len(seg) == S else None


def frames(ds, ep, need, keys):
    return {k: {i: im for i, im in ds.iter_frames(ep, k) if i in need} for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path", required=True); ap.add_argument("--revision")
    ap.add_argument("--repo-id", default="ravediamond/so101_bowls_all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", default="8,12", help="commit offsets (scoring frame t + k - 1)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--checks", action="store_true", help="the protocol's three checks on the first episodes, then stop")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from so101_harness.episodes import EpisodeDataset, download

    G.check_hook()
    m = Pact(snapshot_download(a.policy_path, revision=a.revision) if a.revision else a.policy_path, a.device)
    real = m.real_dim
    sc, off = affine_of(m, real)
    ds = EpisodeDataset(download(a.repo_id, with_videos=True))
    eps_list = json.loads(G13B.read_text())["meta"]["episodes"][:a.episodes]
    ks = [int(x) for x in a.k.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    rows_path = a.out / ("checks.jsonl" if a.checks else "rows.jsonl")
    done = {json.loads(x)["episode"] for x in rows_path.read_text().splitlines()} if rows_path.exists() else set()
    keys = m.image_keys
    other = None  # the shuffled-observation smoke scores with an observation from another episode
    for e, ep in enumerate(eps_list):
        if ep in done:
            continue
        states, acts, task = ds.states(ep), ds.actions(ep), ds.task(ep)
        T = len(states)
        bnds = [t for t in range(S, T - m.horizon, S) if t + max(ks) + S <= T]
        need = set(bnds) | {t + k - 1 for t in bnds for k in ks}
        fr = frames(ds, ep, need, keys)
        obs = lambda i: {"state": states[i], "task": task, **{kk: fr[kk][i] for kk in keys}}  # noqa: E731
        t0 = time.perf_counter(); out = []
        for b_i, t in enumerate(bnds):
            b = e * 1000 + b_i
            cands = [m.predict(obs(t), m.noise(4 * b + c), "naive", 0, None) for c in range(N_CAND)]
            if b_i == 0:
                again = m.predict(obs(t), m.noise(4 * b), "naive", 0, None)
                if not torch.equal(again, cands[0]):
                    sys.exit("check failed: candidate 0 is not bit-identical to a fresh single sample")
            C = torch.zeros(N_CAND, m.horizon, m.action_dim)            # padded to max_action_dim with zeros
            C[:, :, :real] = torch.stack(cands)[:, :, :real]             # (the residual reads the real dims only)
            deg = [m.to_robot(c) for c in cands]
            g = torch.Generator().manual_seed(10_000 + b)
            eps = torch.randn(m.horizon, m.action_dim, generator=g)
            rec_n = torch.zeros(m.horizon, m.action_dim)               # recorded future, normalised (5th candidate)
            rec_n[:, :real] = torch.as_tensor((acts[t:t + m.horizon, :real] - off) / sc, dtype=torch.float32)
            pre_old = m.prefix(obs(t))
            for k in ks:
                E_old = m.flowres(pre_old, C, eps, k, real)
                pre_new = m.prefix(obs(t + k - 1))
                E_new_all = m.flowres(pre_new, torch.cat([C, rec_n[None]]), eps, k, real)
                E_new, E_rec = E_new_all[:N_CAND], E_new_all[N_CAND]
                errs = [err(dg, acts[t:], k) for dg in deg]
                mean_chunk = np.mean(np.stack(deg), axis=0)
                row = {"episode": ep, "t": t, "k": k, "errors": errs,
                       "single": errs[0], "random": float(np.mean(errs)), "oracle": float(np.min(errs)),
                       "old_flowres": errs[int(np.argmin(E_old))], "pact": errs[int(np.argmin(E_new))],
                       "mean_of_n": err(mean_chunk, acts[t:], k), "E_old": E_old.tolist(), "E_new": E_new.tolist(),
                       "rec_below_median": bool(E_rec < np.median(E_new))}
                if a.checks and other is not None:
                    E_sh = m.flowres(m.prefix(other), C, eps, k, real)
                    row["shuffled_pick"] = errs[int(np.argmin(E_sh))]
                out.append(row)
            other = obs(t)
        with rows_path.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in out)
        print(f"episode {e + 1}/{len(eps_list)} ({ep}): {len(bnds)} boundaries, {time.perf_counter() - t0:.0f} s", flush=True)
    if a.checks:
        rows = [json.loads(x) for x in rows_path.read_text().splitlines()]
        san = float(np.mean([r["rec_below_median"] for r in rows]))
        print(json.dumps({"rows": len(rows), "recorded_future_below_median_frac": san, "sanity_ok": san >= 0.6}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
