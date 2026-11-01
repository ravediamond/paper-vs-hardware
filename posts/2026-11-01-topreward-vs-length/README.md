# TOPReward vs elapsed time on ArmnetBench and our rollouts (2026-11-01)

Training-free reward model TOPReward (arXiv 2602.19313) with Qwen3-VL-8B @ `0c351dd`, the paper's success score, as a
failure detector on ArmnetBench (`armnet/armnetbench_v01_lerobot_so101` @ `1db69c8`, top view, n = 2,047) and on our
69 bowls rollouts (`ravediamond/so101-bowls-policy-rollouts` @ `bad1c8e`). Protocol frozen before any score
(`scripts/TOPREWARD_PROTOCOL.md`).

| | within-policy AUROC | 95 % CI |
|---|---|---|
| TOPReward (8B) | 0.661 | 0.631–0.692 |
| episode length | 0.942 | 0.927–0.956 |

Paired difference −0.31 to −0.25: **below** elapsed time; at the 50 % cut 0.667 vs 0.851, also below. On our own
operator-stopped rollouts: TOPReward 0.825 (CI 0.70–0.93), length 0.56. Compute: one RTX 3090 on vast.ai, 2.71 h, $0.69.
