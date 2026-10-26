#!/usr/bin/env python3
"""G9c deterministic judge (G9C_PROTOCOL.md): bowls task from the top camera, no learning.

Per frame: HSV masks for the blue and pink bowl (morphological opening, largest connected component). Per episode:
each bowl's visible area in the last 1 s relative to its area in the first frame of the same episode.

- When one bowl sits in the other, the inner bowl hides the outer one (same size bowls, top view).
- Success (blue in pink): blue visible, pink hidden. Wrong object (pink in blue): pink visible, blue hidden.
- Not done: both visible.

Placement score per frame: s = blue_visible * (1 - pink_visible) * inside, where inside = 1 if the blue centroid
lies within the pink bowl's first-frame enclosing circle, grown by `inside_margin` bowl radii, else 0. (The pink bowl
cannot move once covered; a blue bowl hiding the pink one must be on top of it.)
- A frame is "occluded" when the two bowls' combined visible area is below `occluded_below` of one bowl's
  first-frame area. That means something, in practice the arm, covers the stack.
- The episode score is the median s over the non-occluded frames of the last `window_s` seconds. If all frames are
  occluded, the window extends back in steps up to `max_window_s`; if still occluded, the episode is "undecidable".
Decision: s >= tau. Every parameter is read from g9c_config.json (set on dev data only).
"""
import json

import cv2
import numpy as np


def mask(hsv, ranges, open_px):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:  # several hue ranges (pink wraps around 180)
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    if open_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return 0, None, None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == i)
    (cx, cy), r = cv2.minEnclosingCircle(np.stack([xs, ys], 1).astype(np.float32))
    return int(stats[i, cv2.CC_STAT_AREA]), (float(cent[i][0]), float(cent[i][1])), (float(cx), float(cy), float(r))


def objects(bgr, cfg):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return {name: mask(hsv, c["hsv_ranges"], cfg["open_px"]) for name, c in cfg["colors"].items()}


def frame_score(obj, ref, cfg):
    """obj/ref: {color: (area, centroid, circle)} for this frame and the episode's first frame."""
    a_b, c_b, _ = obj["blue"]; a_p, _, _ = obj["pink"]
    A_b, _, _ = ref["blue"]; A_p, _, circ_p = ref["pink"]
    if not A_b or not A_p:
        return None, {"error": "bowl not found in the first frame"}
    vb, vp = min(a_b / A_b, 1.0), min(a_p / A_p, 1.0)
    occluded = (a_b + a_p) < cfg["occluded_below"] * max(A_b, A_p)
    inside = 0.0
    if c_b is not None and circ_p is not None:
        d = np.hypot(c_b[0] - circ_p[0], c_b[1] - circ_p[1])
        inside = float(d <= circ_p[2] * (1 + cfg["inside_margin"]))
    return vb * (1 - vp) * inside, {"blue_vis": vb, "pink_vis": vp, "inside": inside, "occluded": bool(occluded)}


def episode_score(first_bgr, frames_by_time, cfg):
    """frames_by_time: list of (t_from_end_s, bgr), any order. Returns score (None = undecidable) and details."""
    ref = objects(first_bgr, cfg)
    rows = []
    for t, bgr in frames_by_time:
        s, info = frame_score(objects(bgr, cfg), ref, cfg)
        if s is None:
            return None, {"error": info["error"]}
        rows.append((t, s, info))
    w = cfg["window_s"]
    while w <= cfg["max_window_s"] + 1e-9:
        ok = [s for t, s, i in rows if t <= w + 1e-9 and not i["occluded"]]
        if ok:
            return float(np.median(ok)), {"window_s": w, "n_frames": len(ok), "frames": [i for t, s, i in rows if t <= w + 1e-9]}
        w += cfg["step_s"]
    return None, {"undecidable": True, "frames": [i for _, _, i in rows]}


def load_config(path):
    return json.load(open(path))
