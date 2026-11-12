# Do the top and wrist cameras of an SO-101 learn the same latent? A Platonic test on a robot

Post: https://shissorobotics.com/blog/2026/11/12/paper-vs-hardware-platonic-cameras/

Paper's claim: partly tested. On this rig: unclear.

The paper is "Proving and Improving the Platonic Representation Hypothesis with JEPAs" (Balestriero and LeCun, arXiv 2610.39302). It proves that joint-embedding predictive models trained separately on good views of one scene learn the same representation up to one rotation, that misalignment is bounded when the views are bad, and it leaves action-conditioned predictors outside the theorem. My SO-101 has a top camera and a wrist camera on one scene. I trained single-camera encoders on 157 episodes of bowl pours, with the recipe of my own world model (ViT-Tiny, 192-dimensional latent), and measured how well the two cameras' latents agree (mutual 10-nearest-neighbour overlap on 25 held-out episodes, 95 % episode-bootstrap intervals), against a bad-view control (two random crops of the top camera) and an upper anchor (two seeds of the top camera). Because the authors' code was not used (my own encoder-only trainer), the claim line is "partly tested" at most.

What is here:
- `PROTOCOL.md`: the question, the ladder, the bars B1 and B2 (pass or fail) and B3 (descriptive), frozen at commit `fa3636b` of the road-to-robotics repo before any number existed; `AMENDMENT_2026-11-01.md` holds the owner's answers to the open items.
- `RESULTS.md`: the checks, the ladder and the spend.
- `scripts/align.py` (mutual 10-NN, ZCA, episode bootstrap), `analyze.py` (ladder and verdict), `fdino.py` (the pretrained DINO reference), `wrist_mask.py` (the wrist bowl mask for B3), `test_align.py` (the synthetic checks).
- `expected/alignment.json` (the alignment of every rung with its interval, the paired differences, the B3 bins, the void models; the plot reads it).

Not here: the training logs, the model weights and the embeddings of the eleven models; they stay on the Orin (`projects/jepa-platonic-cameras/` in the road-to-robotics repo) and `alignment.json` is what the analysis writes from them. Training ran on rented GPUs: about 4 USD for the first phase and 4.83 USD for the predictor-on models.

Limits: the top-camera encoder collapsed twice under the pre-registered check (effective rank fraction 0.084 and 0.082, retraining 0.082, line 0.1), so B1 and B2 are void and every alignment value that follows is exploratory; 25 validation episodes; two seeds per model; the predictor-on models were trained after the void, so rung 4 is exploratory too.
