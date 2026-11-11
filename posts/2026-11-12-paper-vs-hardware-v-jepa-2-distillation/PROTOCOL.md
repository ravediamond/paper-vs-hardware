# WS-110 D1 + random teacher — PROTOCOL (DRAFT, not frozen: open fields in §6; no training before the owner settles them)

Written 2026-11-02 after the owner's `go vast WS-110 D1` (cap $6). Every bar and design line below is copied with its
source; nothing here is a new bar. Gap 5 (teacher reference) is closed: `GAP5_PROTOCOL.md` @ 767ded3, results in
`results/gap5.json`, `results/gap5_pooled_rerun.json`.

## 1. Question
Does a W1-sized student trained to regress the frozen V-JEPA 2 ViT-L latent (D1) beat W1-scratch on G1b or §30, and does
it beat a student trained the same way on a randomly initialised ViT-L (knowledge, not regression to a big target)?

## 2. Design (copied)
- **D1** (prompt `journal/prompts/2026-11-01_paper_jepa-distill-edge.md` line 37; memo `journal/papers/jepa-distill-edge/FEASIBILITY.md`
  row 3): W1 encoder + a 192 → 1024 projector (training only, dropped at inference); cosine + MSE to the teacher's pooled latent, per frame, no JEPA loss.
- **Random-teacher student** (prompt line 40; memo row 4): D1's loss on a randomly initialised ViT-L's latents (seed fixed, memo row 2).
- **Teacher**: V-JEPA 2 ViT-L on disk (owner F-ruling for gap 5; `GAP5_PROTOCOL.md` §7: encoder 388 tensors).
- **Probes**: G1b term and §30 term exactly as `GAP5_PROTOCOL.md` §5 (owner F1, F2), 25 validation episodes, paired by episode.

## 3. Bars (copied, prompt line 49–51)
- **B1**: some student beats W1-scratch on G1b top-1 or §30 Δ with CI > 0 **and** beats the random-teacher control on the same probe.
- **B2**: latency ≤ 2 × W1. **B3** descriptive: fraction of the teacher's probe score kept. **Pass = B1 ∧ B2.**
- Fail reading (prompt line 51): "a big pretrained predictor's knowledge does not survive the shrink at 653 episodes".

## 4. Cost (memo row 1–4, 8; not re-estimated from an offer)
| component | memo | in the $5.30 next-step line (memo line 133)? |
|---|---|---|
| teacher latents on the training set (gap 1) | ≈ $1 | yes |
| random-teacher latents (gap 2) | ≈ $1 | yes |
| D1 student (one G0 at fixed steps) | $3.30 | ? |
| random-teacher student (gap 4) | $3.30 | ? |
| W1-scratch at the same data (WS-106 gap 1, shared) | $3.30 | no (WS-106) |
| bandwidth (gap-5 lesson: download billed separately, up to 2× GPU) | — | no |

1 + 1 + 3.30 + 3.30 = $8.60 before bandwidth > the $6 cap; $5.30 = 1 + 1 + 3.30 covers only one of the two students.
At 100 epochs on S4 each student is ~$10 (memo row 3).

## 5. Checks before numbers (proposed, as gap 5 §6)
Teacher strict load 388 encoder tensors; random teacher same architecture, seed logged; student = W1 architecture
(strict load on the Orin); W1-scratch reproduces its stored G1b / §30 rows on the same 25 episodes.

## 6. Owner answers (O1–O6, 2026-11-02, `journal/workstack/notes/WS-110.md`)
- **O1 training data**: W1's bowls set (S3 = `so101_bowls_all_trimmed` @ 4e5c6fda; W1-scratch exists, no extra control).
- **O2 budget / checkpoint rule**: fixed steps as G0 (WS-106 option b); final vs best-val per the gap-5 rule (`GAP5_PROTOCOL.md` §2, owner F6: best-val checkpoint), the final-step checkpoint also reported.
- **O3 target**: pooled 1024-d at 30 Hz.
- **O4 loss weights**: cosine : MSE = 1 : 1.
- **O5 cost scope**: D1 only, cap $6 including bandwidth (cut scope to fit, do not raise); the random-teacher student is the next step.
- **O6 B2**: measured on the Orin at $0 after D1.
- Owner 2026-11-05 ~00:20: the credit reserve rule ($6.50) is waived; spend the credit to finish as many items as possible. The $6 cap of this step stands.

## 7. Gaps before this can be frozen (owner; not covered by O1–O6; each has a proposed default, none is a bar)
- **G1 target alignment**: which teacher latent is frame t's target. Proposed: the pooled 1024-d latent of the 8-frame clip ending at t, per view, the student regressing both views' targets (one 192 → 1024 projector per view), losses averaged. (O3 says "pooled 1024-d at 30 Hz"; the clip-to-frame rule and the two views are not stated.)
- **G2 student recipe**: optimizer, learning rate, batch size and encoder config. Proposed: G0's multi-view LeWM config unchanged (`train_lewm_mv.py`, WS-106 P2) with the predictor and SIGReg losses removed, the projector added for training only and dropped before the strict load on the Orin.
- **G3 probes for an encoder-only student**: §30 is encoder-only and runs as in `GAP5_PROTOCOL.md` §3 on the student's 192-d latent; G1b needs a predictor trained on the student's latents, which no field specifies. Proposed: D1's decision uses the §30 term (B1 reads "G1b top-1 **or** §30 Δ"); G1b is not computed in this step and is reported as not measured.
- **G4 B1's random-teacher clause**: the random-teacher student is not in D1-only scope (O5), so B1 cannot be decided in this step. Proposed: this step reports D1 vs W1-scratch (§30 term, paired by episode, 95 % CI) and the teacher's reference row; B1 stays "not proven" until the random-teacher student runs.
- Reply `D1 defaults` to accept G1–G4 as proposed, or edit them; then the protocol is frozen ($0), the job script is built and run in the WS-127 path, and the D1 student launches within the $6 cap.

## 8. Not built yet (so nothing can launch tonight whatever the credit)
`train_d1.py` (student regression), the clip-aligned teacher-latent extraction at 30 Hz, the §30 probe on student latents, and a remote mode in `wm_train/vast/run_remote.sh`. None of it can be written without G1–G2.
