"""WS-167 O2 control: held-out first-action MAE of the reference SmolVLA checkpoint (smolvla-so101-bowls-020000) on the same 10 episodes.

  scripts/docker_py.sh -i 0.7 -m 11g -v CKPT_PARENT:/ck -v ... -- projects/lajepa/smolvla_heldout.py --ckpt /ck/pretrained_model --root DATASET_SNAPSHOT --out RESULT.json

Same frames and same metric as projects/lajepa/train.py evaluate(): every frame of episodes 172-181 of so101_bowls_all_trimmed (rev 4e5c6fda...),
chunk predicted from the observation at that frame (full-resolution cameras, the checkpoint's own pre/post-processors, so its own normalisation),
first action vs the recorded action, raw dataset units, mean over 6 dims then over frames, per episode. The identity null is written next to it.
CAVEAT recorded in notes/WS-167.md: this checkpoint was trained on so101_bowls_all (all 182 episodes, train_config.json), i.e. it has seen
these 10 episodes; the number is an in-sample reference, not a held-out one.
"""
import argparse, json, time
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="ravediamond/so101_bowls_all_trimmed")
    ap.add_argument("--revision", default="4e5c6fda8dc5ce2252214bc086181d3df8b594b0")
    ap.add_argument("--episodes", default="172,173,174,175,176,177,178,179,180,181")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N frames only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    dev = torch.device(a.device)
    policy = SmolVLAPolicy.from_pretrained(a.ckpt).to(dev).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=a.ckpt,
                                        preprocessor_overrides={"device_processor": {"device": a.device}},
                                        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    eps = [int(x) for x in a.episodes.split(",")]
    ds = LeRobotDataset(a.repo, root=a.root, episodes=eps, revision=a.revision, video_backend="pyav")
    n = len(ds) if a.limit is None else min(a.limit, len(ds))
    torch.manual_seed(a.seed)
    rows, t0 = [], time.time()
    for i in range(n):
        it = ds[i]
        obs = {k: v for k, v in it.items() if k.startswith("observation.")}
        obs["task"] = it["task"]
        with torch.no_grad():
            chunk = policy.predict_action_chunk(pre(obs))
            a0 = post(chunk)[0, 0].float().cpu().numpy()
        tgt = it["action"].float().numpy()
        rows.append((int(it["episode_index"]), np.abs(a0[:6] - tgt), np.abs(it["observation.state"].float().numpy() - tgt)))
        if i % 200 == 0:
            print(i, n, round(time.time() - t0), "s", flush=True)
    by = {}
    for e, l1, nl in rows:
        by.setdefault(e, []).append((l1, nl))
    per = {}
    for e, v in sorted(by.items()):
        l1 = np.stack([x[0] for x in v]); nl = np.stack([x[1] for x in v])
        per[str(e)] = {"mae": float(l1.mean()), "identity_null": float(nl.mean()), "frames": len(v), "mae_per_dim": l1.mean(0).tolist()}
    res = {"per_episode": per, "mean_mae": float(np.mean([v["mae"] for v in per.values()])),
           "mean_identity_null": float(np.mean([v["identity_null"] for v in per.values()])), "checkpoint": a.ckpt, "n_frames": n,
           "in_sample": "trained on so101_bowls_all (182 episodes) incl. these 10", "seed": a.seed,
           "units": "raw dataset units (joint degrees; gripper in dataset units)"}
    json.dump(res, open(a.out, "w"), indent=1)
    print("SmolVLA mean MAE", res["mean_mae"], "null", res["mean_identity_null"])


if __name__ == "__main__":
    main()
