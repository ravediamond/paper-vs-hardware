import sys, numpy as np, pandas as pd
sys.path[:0] = ["/work/scripts", "/work/src", "/work"]
import run_w1_offline_screening as w
from so101_harness.episodes import EpisodeDataset, download
from so101_harness.w1_live import W1LiveMonitor
ds = EpisodeDataset(download("ravediamond/so101-bowls-policy-rollouts"))
o = pd.read_csv("/work/data/rollouts/so101-bowls-policy-rollouts_labels.csv")
mon = W1LiveMonitor.from_checkpoint("<W1 weights_final.pt>")
rows = []
for r in o[o.outcome.isin(["success", "failure"])].itertuples():
    ep = int(r.episode_index); z = np.load(f"/work/outputs/w1_screening/latents/ep{ep:03d}.npy")
    T = len(ds.states(ep)); end = T if r.reset_start < 0 else min(r.reset_start, T)
    for c in w.anchor_curves(mon, z, ds.actions(ep), end):
        rows.append({"ep": ep, "outcome": r.outcome, "k3": c["l2"][3], "k10": c["l2"][10],
                     "k15_25": max(c["l2"][k] for k in (15, 20, 25))})
d = pd.DataFrame(rows)
for oc, g in d.groupby("outcome"):
    print(oc, "anchors", len(g), "| k3: median %.2f p90 %.2f, share > 2.2: %.2f | k15-25 max: median %.2f p90 %.2f, share > 10: %.2f" % (
        g.k3.median(), g.k3.quantile(.9), (g.k3 > 2.2).mean(), g.k15_25.median(), g.k15_25.quantile(.9), (g.k15_25 > 10).mean()))
    e = g.groupby("ep").agg(stall=("k3", lambda x: (x > 2.2).any()), slip=("k15_25", lambda x: (x > 10).any()))
    print("   episodes flagged by stall rule:", int(e.stall.sum()), "/", len(e), " by slip rule:", int(e.slip.sum()), "/", len(e))
