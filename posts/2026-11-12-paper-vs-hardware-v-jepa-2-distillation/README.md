# Distilling V-JEPA 2 into a student the size of my SO-101 world model: does the teacher's knowledge survive?

Post: https://shissorobotics.com/blog/2026/11/12/paper-vs-hardware-v-jepa-2-distillation/

Paper's claim: not tested. On this rig: unclear.

The model is V-JEPA 2 (Assran et al., arXiv 2506.09985). Its paper makes no claim about shrinking it, so the question here is mine: if a 192-dimensional student (the encoder of my SO-101 world model, W1) is trained to regress the frozen V-JEPA 2 ViT-L latent of the same frames, does it carry more beyond-state information than W1 trained from scratch, and more than a student trained the same way on a randomly initialised ViT-L of the same shape? The bar (B1) needs both; the latency bar (B2) is at most twice W1's.

What is here:
- `PROTOCOL.md`: the question, the bars B1 and B2 and the descriptive B3, copied from the work-stack prompt. The status line still reads "DRAFT, not frozen"; the owner accepted the open fields G1 to G4 on 2026-11-07 and the protocol was frozen from that commit.
- `scripts/d1_train.py` (teacher-latent cache, student training, per-anchor rows), `d1_compare.py` (the beyond-state term with a paired episode bootstrap), `d1_b1.py` (D1 against the random-teacher student), `d1_b2.py` (latency on the Orin), `test_d1.py`.
- `expected/b1.json` (the six terms and the paired differences, which the plot reads), `b2.json` (latency), `d1.json` and `rt.json` (the two students' rows), `teacher_reference.json` (the V-JEPA 2 teacher's own term, 1.6510, key `terms.VJEPA2`).

Not here: the 81,100-step training logs, the per-anchor prediction files (`d1_preds.npz`, `rt_preds.npz`), the teacher latents and the weights; they stay on the Orin and the rented instance, and the two students' rows in `d1.json` and `rt.json` are what the comparison reads. The D1 training took 7,101 s on an RTX A4000 (`d1_metrics.jsonl`, kept on the Orin).

Limits: 25 validation episodes, one seed per student, two different rented GPUs for the D1 student and the random-teacher student (W1's term differs by 0.0004 between them, see `b1.json`, `w1_rows_across_instances`); the teacher is V-JEPA 2 and not V-JEPA 2.1; the D1 run was labelled exploratory in the protocol; the G1b probe was not run for the students.
