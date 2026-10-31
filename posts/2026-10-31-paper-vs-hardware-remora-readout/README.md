# Paper vs hardware 22: REMORA's readout (fewer flips, no setback detection on this evidence)

Posts:
- https://shissorobotics.com/blog/2026/10/31/paper-vs-hardware-remora/

## Claim

REMORA (Wang, Niu, Zhu et al., *Tracking Task Progress Through Errors and Recovery*, arXiv 2610.39125) tracks task
progress as a distribution and reads it out causally by taking the supported peak nearest the previous estimate (§3.3),
trained against a target that falls inside error intervals and recovers after them (§3.2, Eq. 2). On their D2 (25 error
episodes in 120), peak selection against the per-frame median lowers MAE by 45 % and raises Spearman ρ from 0.870 to
0.959. Shown on an AgiBot G02 (head + 2 wrist cameras, 30 Hz), trained on 8 × H200. Only the two model-free parts are
tested here, on the evidence this rig already has.

**Verdict: P1 pass, P2 fail, P3 fail; not adopted.** The nearest peak points the way the paper says (fewer flips
than per-frame readouts, no worse detection) but cuts flips by 30 %, not half, and cannot tell a real setback from
evidence noise.

Protocol `PROTOCOL.md` (frozen 2026-10-31, before any number). Script `scripts/remora_readout.py`, committed before any
number (`59b1ede`); the dev-half fit (`expected/p_t_fit.json`) and the targets (`expected/target.jsonl`) committed before
the test half was read (`f8c5bee`).

## Setup

G15's evidence on the 69 rollouts (proprio + top-camera blobs, a query every 3 frames); p_t over three atoms {0, 0.5, 1}
(reach / hold / placed) from two logistic regressions fitted on G15's dev half (7,069 queries); four readouts of the
same p_t: R0 G15's monotone stage machine, R1 per-frame argmax, R2 per-frame median (the paper's A5), R3 causal nearest
peak (peaks ≥ 0.2, ties → lower atom). G14's reversal windows play the error intervals. Test half: G15's split, 34
episodes (the protocol said 35; G15's list has 34).

Checks (all passed, `expected/checks.json`): R0 on the recomputed evidence reproduces G15's dev rates within 2 points;
the target is in [0, 1], non-decreasing outside the windows and decreasing inside those where the stage was advancing;
R3 does not flip on two alternating equal peaks. 9 of the 14 G14 windows fall where the target is flat.

## Results (34 test episodes, `expected/summary.json`)

| readout | premature | missed | flips / episode | regression recall (7 windows) | false regressions | MAE vs ỹ |
|---|---|---|---|---|---|---|
| R0 stage machine | 0.485 | 0.467 | 1.00 | 0 / 7 | 0.00 | 0.248 |
| R1 argmax | 0.250 | 0.133 | 5.97 | 3 / 7 | 0.56 | 0.183 |
| R2 median | 0.397 | 0.133 | 6.35 | 3 / 7 | 0.74 | 0.218 |
| **R3 nearest peak** | **0.221** | **0.133** | **4.21** | **2 / 7** | **0.52** | 0.184 |

| bar | value | verdict |
|---|---|---|
| **P1** R3 premature ≤ R1, missed ≤ R1 + 2 points | premature −5.9 points (CI −17.6 to 0), missed +0 | **pass** |
| **P2** R3 flips ≤ 0.5 × R1, CI upper ≤ 0.7 | 0.70 (CI 0.48–0.94) | **fail** |
| **P3** regression recall ≥ 0.5, false regressions ≤ 0.2 | 0.29, 0.52 | **fail** |

Hand check (`expected/hand_check.jpg`): on the seven stalled or retried-grasp failures R3 stays at reach for the whole
episode, which is right (none lifted a bowl); they have no progress to lose, so they cannot exercise the regression claim.

## Compute

External GPU: no. Orin NX CPU (feature extraction in the lerobot-jetson container). Cost: $0.

## What is not proven

Three atoms only (the paper uses 201 bins); the evidence is hand-written predicates, not a learned representation; the
G14 windows are a mechanical criterion, not human error labels; the recurrent memory and the backbone are not tested.
