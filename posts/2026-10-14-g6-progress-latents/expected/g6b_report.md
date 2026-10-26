# Gate G6b Report: H-JEPA Level-2 Slow-Feature Progress Test

**Protocol:** [`../../G6B_PROTOCOL.md`](../../G6B_PROTOCOL.md)  
**Date:** 2026-10-18  
**Device:** Apple Silicon (mps)  
**Cost:** $0.00  

---

## 1. Executive Summary

| Test / Metric | Level-1 Baseline (W1) | H-JEPA Level-2 ($s=15$) | Pass Bar | Status |
|---|---|---|---|---|
| **T1 Monotonicity (Spearman $\rho$)** | **-0.7215** [-0.7932, -0.6467] | **-0.7268** [-0.7901, -0.5874] | $\le -0.80$ (or $\le -0.9215$) | **FAIL** |
| **False Stall Rate (Held-out)** | **32.0%** | **52.0%** | $\le 10.0\%$ | **FAIL** |
| **G10 Sanity (Judge AUROC)** | **1.0000** [1.0000, 1.0000] | **0.9977** [0.9915, 1.0000] | $\ge 0.95$ | **PASS** |
| **G10 Sanity (GT AUROC)** | **0.9703** [0.9357, 0.9938] | **0.9374** [0.8846, 0.9783] | $\ge 0.90$ | **PASS** |

**Overall Gate Verdict:** **FAILED**

---

## 2. Detailed Findings

1. **Monotonic Progress Tracking ($T_1$):**
   - Level-1 (W1) achieved Spearman $\rho = -0.7215$ (matching G6a's -0.499).
   - Level-2 H-JEPA achieved Spearman $\rho = -0.7268$ (CI [-0.7901, -0.5874]).
   - Improvement: **+0.0053** Spearman correlation units.

2. **False Stall Elimination:**
   - Under the frozen G6a stall rule ($W=8.0\text{ s}$, $\epsilon=0.02 \times d_{\text{ref}}$), Level-1 produced a 32.0% false stall rate.
   - Level-2 reduced the false stall rate to **52.0%**.

3. **Outcome Discriminability (G10 Sanity):**
   - Outcome classification on $z_{t_j}^{(2)}$ achieved **0.9977 AUROC** against the deterministic color judge and **0.9374 AUROC** against human ground truth.

---

## 3. Decision Implication

* **H-JEPA Hierarchy is DECISIVELY REJECTED:** As pre-registered in the protocol, multi-level temporal abstraction does not solve the monotonic progress failure. The world model drops the progress tracking role entirely and remains strictly a **Level-1 Pre-Execution Veto** and **Mid-Chunk Surprise Monitor** (||z_t - ẑ_t|| > 10.0).
