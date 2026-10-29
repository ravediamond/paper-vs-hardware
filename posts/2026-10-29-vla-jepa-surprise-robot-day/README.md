# VLA-JEPA's own surprise as a failure warning: the robot day ran 20 of 20 successes (not proven)

Posts:
- https://shissorobotics.com/blog/2026/10/29/vla-jepa-surprise-robot-day/

## Claim

The question (frozen 2026-10-25, before the robot day): does VLA-JEPA's own world-model surprise warn before a failure
on the arm, better than W1's surprise, a stall rule, elapsed time and the REAP risk head? The bar needs at least 6
failures to score anything. On 2026-10-27 VLA-JEPA (checkpoint `c65c98b`, Orin NX at MAXN_SUPER) completed **20 of 20**
sealed layouts, 9 of them with each bowl shifted 3–6 cm from a teleop position, so there was nothing to warn about:
**not proven, too few failures (0 < 6)**. The owner stopped at 20 of the 60 planned rollouts (0 failures in 13 made 6
failures in the remaining 47 about 1–2 % likely); the verdict does not depend on the rollouts not run.

## Compute

**Compute.** External GPU: no (Orin NX, arm). Cost: $0.

## What is not proven

Everything the bar was about: no failure, so no monitor was scored. The 20 episodes are not published yet. A harder
layout set (an amendment with its own bar) is the next step.

See `REPRODUCE.md`.
