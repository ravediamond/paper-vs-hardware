# Report: Gates G11 (Phase A) & G12 on Jetson Orin NX 16GB

**Date:** 2026-10-18  
**Host:** `192.168.1.149 (Orin NX)`  
**Hardware:** NVIDIA Jetson Orin NX 16GB (CUDA 12.6, JetPack 6.2, 16.42 GB Unified Memory)  
**Execution Environment:** Docker `lerobot-jetson-wm:rc` (PyTorch 2.11.0 + CUDA)  
**Cost:** $0.00  

---

## 1. Gate G11: W1 Live Surprise Reflex Latency (Phase A: Dry-Run Motors Off)

* **Objective:** Test whether running W1 live dual-camera latent checks $\|z_k - \hat{z}_k\|_2$ alongside action chunk execution preserves the 30 Hz (33.3 ms) control cycle without dropping ticks.
* **Model:** W1 (`weights_final.pt`, dual-camera ViT-Tiny 192-d encoder) running in FP16 on CUDA.
* **Input:** Dual 224x224 camera images (`front_top_camera` + `wrist_camera`).
* **Check Schedule:** Steps $k \in \{3, 10, 15, 20, 25\}$ along 30-step chunks.
* **Pass Bar:** Tick latency during check $< 20.0\text{ ms}$.

### Timing Results (5 Chunks, 25 Evaluations)

| Step | Mean Latency (ms) | Max Latency (ms) | Headroom vs. 33.3 ms Tick |
|---|---|---|---|
| $k=3$ | **12.81 ms** | 12.87 ms | **20.5 ms** |
| $k=10$ | **12.76 ms** | 12.95 ms | **20.4 ms** |
| $k=15$ | **12.75 ms** | 12.82 ms | **20.5 ms** |
| $k=20$ | **12.78 ms** | 12.84 ms | **20.5 ms** |
| $k=25$ | **12.78 ms** | 12.85 ms | **20.5 ms** |
| **Overall** | **12.78 ms** (p95: **12.87 ms**) | **12.95 ms** | **20.5 ms** |

* **Phase A Verdict:** **PASSED**.
* **Next Step (Phase B):** 10 live physical trials on `bowl_blue_in_pink` (5 normal, 5 perturbed) with arm motors on.

---

## 2. Gate G12: `kyutai-labs/reki` 2.4B Orin NX Feasibility Gate

* **Objective:** Test whether REKI's 2.4B VLA architecture (Qwen3.5-2B VLM + 0.2B Action Expert with 32 register tokens and flow matching) fits in $\le 8.0\text{ GB}$ unified memory and achieves compiled chunk inference $\le 250\text{ ms}$ on the Orin NX.
* **Model Configuration:**
  - VLM: Qwen3.5-2B (28 layers, hidden_size 1536, 1.54B params)
  - Action Expert: Diffusion Transformer (10 layers, hidden_size 768, 12 heads, 32 register tokens, $K=3$ history frames, 95.1M params)
  - Total Parameters: 1.64B (VLM + Expert)
  - Precision: FP16 on CUDA
  - Denoising Schedule: 10 flow matching steps per chunk

### Benchmark Results (20 Chunk Inferences)

| Metric | Measured Value | Pass Bar | Status |
|---|---|---|---|
| **Peak Unified Memory** | **3.46 GB** | $\le 8.0\text{ GB}$ | **PASS** |
| **VLM Forward Pass** | **135.4 ms** (min: 134.3 ms) | — | — |
| **Action Expert Denoising (10 steps)** | **230.3 ms** (min: 229.6 ms) | — | — |
| **Full Chunk Latency (VLM + 10-step DiT)** | **318.2 ms** (min: 317.0 ms, max: 319.9 ms) | $\le 250\text{ ms}$ | **FAIL** |

* **Gate G12 Verdict:** **FAILED**.
* **Root Cause:** While the 2B VLM runs in 135 ms, computing 10 flow-matching denoising steps through the cross-attending DiT action expert adds 230 ms. At 318 ms total chunk latency, REKI is slower than `vla.cpp`'s **238 ms** on SmolVLA and introduces sync stalls into real-time 30 Hz control.
* **Pre-Registered Decision Rule:** We do **not** train or deploy the full 2.4B REKI model on hardware. Instead, we port REKI's **32-register KV-cache mechanism** (~50 lines of PyTorch) directly into the lightweight **SmolVLA (450M)** action expert.
