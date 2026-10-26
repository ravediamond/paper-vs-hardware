# Step D: latency on the Jetson Orin NX 16 GB (MAXN_SUPER, bf16 backbone, 224x224, 4 Euler steps)

50 timed runs after 5 warm-up, one real observation; ms, median (p95).

| N | preprocess | backbone | head (batched N) | postprocess | total | total / N=1 |
|---|---|---|---|---|---|---|
| 1 | 1.4 (1.4) | 242.9 (244.0) | 104.1 (104.6) | 0.6 (0.7) | 349.3 (350.8) | 1.00x |
| 4 | 1.4 (1.4) | 243.1 (243.5) | 211.7 (212.2) | 0.7 (0.7) | 457.2 (458.1) | 1.31x |
| 8 | 1.4 (1.4) | 243.4 (243.7) | 397.2 (397.5) | 0.7 (0.7) | 643.1 (643.4) | 1.84x |

Reference: deployed `predict_action_chunk` (backbone + head, N = 1) 350.4 (351.1) ms. Peak torch allocation 6.17 GB.
