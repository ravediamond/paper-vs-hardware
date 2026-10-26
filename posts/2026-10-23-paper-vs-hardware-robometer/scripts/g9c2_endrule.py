"""G9c-2 automatic end-of-subtask rule (G9C2_PROTOCOL.md): which frame the judge looks at.

judge frame = the last frame after the final gripper release, before any hand enters the top view, where the arm is
at rest (all 5 arm joints within rest_tol_deg of the episode's first pose, and still over still_s). None = invalid.
All parameters come from g9c2_rule.json (set on dev only).
"""
import glob

import av
import cv2
import numpy as np
import pandas as pd

from g9c_io import CAM


def load_states(root):
    """{episode_index: (T, 6) array of observation.state}, from the dataset's data parquet files."""
    x = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                   for f in sorted(glob.glob(f"{root}/data/*/*.parquet"))]).sort_values(["episode_index", "frame_index"])
    return {int(e): np.stack(g["observation.state"].values) for e, g in x.groupby("episode_index")}


def skin_mask(bgr, rule, judge_cfg):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, np.array(rule["skin_ycrcb"][0], np.uint8), np.array(rule["skin_ycrcb"][1], np.uint8))
    m &= cv2.inRange(hsv, np.array(rule["skin_hsv"][0], np.uint8), np.array(rule["skin_hsv"][1], np.uint8))
    pink = np.zeros_like(m)
    for lo, hi in judge_cfg["colors"]["pink"]["hsv_ranges"]:
        pink |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    m &= ~pink
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rule["skin_open_px"], rule["skin_open_px"]))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0


def hand_scan(E, ep, start_frame, rule, judge_cfg):
    """First frame index >= start_frame with a skin component >= skin_min_px (top view, scanned at hand_fps)."""
    m = E.meta.loc[ep]
    path = f"{E.root}/" + E.info["video_path"].format(video_key=CAM, chunk_index=int(m[f"videos/{CAM}/chunk_index"]),
                                                     file_index=int(m[f"videos/{CAM}/file_index"]))
    t0 = float(m[f"videos/{CAM}/from_timestamp"]); n = int(m["length"]); fps = E.fps
    step = max(1, int(round(fps / rule["hand_fps"])))
    wanted = set(range(start_frame, n, step)); best = 0
    with av.open(path) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"
        c.seek(int(max(t0 + start_frame / fps - 0.5, 0) / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            k = round(float(fr.pts * s.time_base) * fps) - round(t0 * fps)
            if k >= n:
                break
            if k in wanted:
                a = skin_mask(fr.to_ndarray(format="bgr24"), rule, judge_cfg); best = max(best, a)
                if a >= rule["skin_min_px"]:
                    return k, best
    return None, best


def judge_frame(E, ep, states, rule, judge_cfg):
    """Returns (frame index or None, details)."""
    st = states[int(ep)]; T = len(st); fps = E.fps
    arm = st[:, :5]; grip = st[:, 5]
    opened = np.nonzero(grip > rule["g_open"])[0]
    t_rel = int(opened[-1]) if len(opened) else 0
    dev = np.abs(arm - arm[0]).max(1)
    w = max(1, int(round(rule["still_s"] * fps)))
    move = np.full(T, np.inf)
    for t in range(w, T):
        move[t] = np.abs(arm[t] - arm[t - w]).max()
    rest = (dev <= rule["rest_tol_deg"]) & (move <= rule["still_tol_deg"])
    t_hand, max_skin = hand_scan(E, ep, t_rel, rule, judge_cfg)
    stop = T if t_hand is None else t_hand
    cand = np.nonzero(rest[t_rel:stop])[0]
    info = {"t_rel": t_rel, "t_hand": t_hand, "max_skin_px": max_skin, "n_frames": T,
            "rest_frames_after_release": int(rest[t_rel:].sum())}
    if not len(cand):
        return None, {**info, "invalid": True}
    jf = t_rel + int(cand[-1])
    return jf, {**info, "judge_frame": jf, "s_before_end": (T - 1 - jf) / fps}
