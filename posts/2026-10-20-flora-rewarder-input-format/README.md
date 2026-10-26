# Judging, part 2: the input format did the work (Flora-Rewarder vs general models)

Posts:
- https://shissorobotics.com/blog/2026/10/20/the-input-format-did-the-work/

## Claim

Flora-Rewarder-27B flags 8/8 failures at 2/42 successes (AUROC 1.00); with the same input, Qwen3.5-27B passes too (7/8 at 2/42), Qwen3.5-9B and Gemini 3.8 Flash fail.

## Compute

**Compute.** External GPU: yes (vast.ai): G-flora ~$0.7, ablation ~$1.5. External API: Gemini ~$2.75. Total ~$4.95. Retries: not itemised in the gate notes.

## What is not proven

50 episodes, 8 failures; one prompt per model; Flora's licence is noncommercial.

See `REPRODUCE.md`.
