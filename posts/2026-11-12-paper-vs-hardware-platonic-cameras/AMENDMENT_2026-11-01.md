# Owner decisions on PROTOCOL.md §9 (2026-11-01, before any training)

- **O1** (b): two bad-view JEPAs E_crop1 / E_crop2 (seeds 3072 / 3073), alignment(1) measured on full frames. (a) would
  train the encoder to map the crops together and inflate the anchor.
- **O2** W1's augmentation for the crop range and output size (one recipe everywhere); the paper's App. B crops only
  if W1 has none — then DINO-style, scale 0.4–1.0, stated.
- **O3** G9c HSV masks of the bowl on the wrist frame, fraction of the bowl's first-frame area visible (the G9c
  "visible fraction" cue), computed on the val episodes.
- **O4** Whatever W1's encoder exposes as its frame embedding (mean-pooled if both exist); fixed in the code item and stated.
- **O5** If alignment(2) is not above alignment(1) with CI, B1 and B2 are **not proven** (the anchors do not separate);
  the paper reports the ladder and stops.
- **O6** Confirmed: 95 % episode-bootstrap CI of the paired difference, two-sided interval; B2's direction one-sided
  (the theorem's), pass = the interval excludes 0 on the predicted side, fail = excludes 0 on the other side, else not proven.
- **O7** B3: four bins of wrist bowl coverage (0–25 / 25–50 / 50–75 / 75–100 %); "monotone" = Spearman of alignment vs
  bin midpoint with CI; descriptive, no bar.
- **O8** A collapse check failure voids the pair (not "not proven"): retrain once with the next seed; a second failure
  = the model is reported as collapsed and the bar is void.
- **O9** Add one seed each to E_wrist, P_top, P_wrist (+$9) so B2 is a seed-matched comparison; E_top×wrist stays one
  seed (descriptive).
