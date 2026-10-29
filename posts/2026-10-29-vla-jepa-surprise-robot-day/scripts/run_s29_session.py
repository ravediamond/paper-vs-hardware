#!/usr/bin/env python3
"""NEXT_ANGLES §29 step 2 robot day (WS-018): the sealed layouts in order, one VLA-JEPA trial each, labels at rest.

Per rollout, in `LAYOUTS.csv` order (its sha256 is checked against the sealed value first):
1. **Placement.** The top camera (C920, front_top_camera) is read live; both bowls are found with the precheck's HSV
   detector and compared with the layout's two pixel targets. The terminal prints how far each bowl is from its target
   in cm and which way to move it ("towards the robot / away", "left / right" as seen in the top image). With a display
   (DISPLAY set) a window also draws the target circles on the live image. Enter accepts once both bowls are within
   PLACE_TOL_CM; the residual offsets are logged.
2. **Trial.** shared/gate/monitor_rollout.py --policy vla_jepa (checkpoint @ c65c98b), one trial, 25 s, the layout's
   instruction, recorded (both cameras, state, action) under --record-dir; its own y/n prompt is off.
3. **Label at rest**, before anything is scored: outcome = success / missed grasp / dropped / wrong bowl / stalled /
   timeout (the bar's list), written to `labels.csv` immediately. For a failure, the onset frame is marked on the
   front_top video with projects/vla-jepa-world-model/recording/label_onset.py: in a window right away if a display is
   available, otherwise after the session (on the Mac) and always before `s29_score.py` runs.
4. **Stop** at 10 manipulation failures or 60 rollouts. **Thermal pause** while the hottest servo is >= 58 °C, until < 50 °C.

The arm moves only after `go arm WS-018`. Inside the VLA-JEPA image, from the repo root:

    IMAGE=lerobot-jetson:rc LEROBOT_SRC=image shared/gate/jetson_run.sh --python scripts/run_s29_session.py \\
        --record-dir outputs/s29/<date> [--start 1] [--dry-run] [--fake-frame <start-frame.jpg>]

--dry-run moves nothing: it checks the seal, the checkpoint and the camera (or --fake-frame), prints the plan, the
placement guidance for rollout --start and the monitor_rollout command, and exits.

Steered mode (owner 2026-10-27: Claude Code steers, the owner moves the bowls and calls the outcome), one step per call,
no stdin:
    --step place --rollout N              camera only: residual per bowl and READY or not (never moves)
    --step trial --rollout N              re-checks placement (refuses unless both within PLACE_TOL_CM), thermal pause,
                                          then the one trial (the arm moves: only after `go arm WS-018`)
    --step label --rollout N --outcome O  writes the owner's outcome (success | missed_grasp | dropped | wrong_bowl |
                                          stalled | timeout) for the last trial of rollout N to labels.csv
    --step status                         rollouts labelled, failures, next rollout
"""

import argparse
import csv
import getpass
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "projects/vla-jepa-world-model/prechecks"))
import layouts as L  # noqa: E402  (bowl_centroids: the detector the layouts were drawn with)

DAY = ROOT / "shared/gate/data/2026-10-26_s29_robot_day"
SEALED_SHA256 = "49e4d364c5843f079cbc278ccf2c09ef87ca66aa7bb8b3c7b68d13608c9348aa"
CKPT = Path.home() / ".cache/huggingface/hub/models--ravediamond--vla_jepa_so101_bowls_trimmed/snapshots/c65c98bb713d0e8bf3692e2fbc624869e1d10f05"
PROFILE = ROOT / "shared/gate/configs/robots/so-101-jetson.json"
TASKS = {"blue-in-pink": "Put the blue bowl in the pink bowl.", "pink-in-blue": "Put the pink bowl in the blue bowl."}
OUTCOMES = ["success", "missed_grasp", "dropped", "wrong_bowl", "stalled", "timeout"]
PLACE_TOL_CM, MAX_FAILURES, DURATION_S, HOT_C, COOL_C = 1.0, 10, 25.0, 58.0, 50.0
LABEL_FIELDS = ["episode_index", "reset_start", "rollout", "instruction", "stratum", "base_episode", "outcome", "failure_mode", "rollout_path",
                "blue_residual_cm", "pink_residual_cm", "labeller", "timestamp"]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def top_camera_index():
    return next(c["camera_index"] for c in json.loads(PROFILE.read_text())["cameras"] if c["name"] == "front_top_camera")


def grab(cap):
    import cv2

    for _ in range(3):                      # drop buffered frames
        cap.grab()
    ok, bgr = cap.read()
    if not ok:
        raise SystemExit("top camera read failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def guidance(rgb, row, cm_px):
    """Per bowl: residual (cm) and a hint in the top image's frame (the robot is at the top edge)."""
    H, W = rgb.shape[:2]
    found = dict(zip(("blue", "pink"), L.bowl_centroids(rgb)))
    out = {}
    for c in ("blue", "pink"):
        tgt = np.array([row[f"{c}_x_px"], row[f"{c}_y_px"]], float)
        if np.isnan(found[c]).any():
            out[c] = (None, f"{c}: not seen")
            continue
        d = (tgt - found[c] * [W, H]) * cm_px          # + x = move right in the image, + y = move away from the robot
        hint = []
        if abs(d[0]) >= 0.3:
            hint.append(f"{abs(d[0]):.1f} cm {'right' if d[0] > 0 else 'left'}")
        if abs(d[1]) >= 0.3:
            hint.append(f"{abs(d[1]):.1f} cm {'away from the robot' if d[1] > 0 else 'towards the robot'}")
        r = float(np.hypot(*d))
        out[c] = (r, f"{c}: {r:.1f} cm off" + (f" (move {', '.join(hint)})" if hint else " (in place)"))
    return out


def place(row, cm_px, cap=None, fake=None, once=False):
    """Loop until both bowls are within PLACE_TOL_CM and the owner presses Enter; returns the residuals (cm)."""
    import select

    import cv2

    window = bool(os.environ.get("DISPLAY")) and fake is None
    D = 12.0 / cm_px
    print(f"\nRollout {row['rollout']}: {row['stratum']}, {row['instruction']}. Slide the bowls into place; Enter when "
          f"both are within {PLACE_TOL_CM} cm.", flush=True)
    while True:
        rgb = fake if fake is not None else grab(cap)
        g = guidance(rgb, row, cm_px)
        ok = all(g[c][0] is not None and g[c][0] <= PLACE_TOL_CM for c in g)
        print("\r  " + " | ".join(g[c][1] for c in ("blue", "pink")) + ("   READY" if ok else "") + " " * 8, end="", flush=True)
        if window:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for c, col in (("blue", (255, 160, 0)), ("pink", (160, 80, 255))):
                cv2.circle(bgr, (int(row[f"{c}_x_px"]), int(row[f"{c}_y_px"])), int(D / 2), col, 3)
            cv2.imshow("s29 placement", bgr); cv2.waitKey(1)
        if once:
            print()
            return {c: g[c][0] for c in g}
        if select.select([sys.stdin], [], [], 0.3)[0]:
            sys.stdin.readline()
            if ok:
                print()
                return {c: g[c][0] for c in g}
            print("\n  not yet: both bowls must be within tolerance", flush=True)


def ask_outcome():
    for i, o in enumerate(OUTCOMES, 1):
        print(f"    {i}. {o}")
    while True:
        a = input("  Outcome at rest [1-6]: ").strip()
        if a.isdigit() and 1 <= int(a) <= len(OUTCOMES):
            return OUTCOMES[int(a) - 1]


def hottest_c():
    """Hottest of the six STS3215 servos (Present_Temperature, read without torque change; the 58 / 50 °C rule is per
    servo, as in lego-kitting's ThermalGuard). The Orin's own sensors idle at 54-59 °C in the quiet fan profile and
    are not what the rule is about."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    port = json.loads(PROFILE.read_text())["follower_port"]
    bus = FeetechMotorsBus(port, {n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(names)})
    bus.connect(handshake=False)
    try:
        t = bus.sync_read("Present_Temperature", normalize=False)
    finally:
        bus.disconnect(disable_torque=False)
    return float(max(t.values()))


def thermal_pause():
    t = hottest_c()
    print(f"  hottest servo {t:.0f} °C", flush=True)
    if t is not None and t >= HOT_C:
        print(f"  thermal pause: servo at {t:.1f} °C >= {HOT_C}; waiting for < {COOL_C} °C", flush=True)
        while (t := hottest_c()) is not None and t >= COOL_C:
            time.sleep(15)
        print(f"  resumed at {t:.1f} °C", flush=True)


def trial_cmd(row, record_dir):
    return [sys.executable, str(ROOT / "shared/gate/monitor_rollout.py"), "--device-name", "orin_nx",
            "--robot-profile", str(PROFILE), "--play_sounds=false", "--policy", "vla_jepa", "--policy-path", str(CKPT),
            "--task", TASKS[row["instruction"]], "--trials", "1", "--duration", str(DURATION_S), "--no-prompt",
            "--out", str(record_dir)]


def last_rollout_path(record_dir):
    rows = list(csv.DictReader(open(Path(record_dir) / "trials.csv")))
    return rows[-1]["rollout_path"] if rows else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record-dir", type=Path, required=True)
    ap.add_argument("--start", type=int, default=1, help="resume at this rollout number")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fake-frame", type=Path, help="dry run: a start-frame image instead of the live camera")
    ap.add_argument("--step", choices=["place", "trial", "label", "status"])
    ap.add_argument("--rollout", type=int)
    ap.add_argument("--outcome", choices=OUTCOMES)
    a = ap.parse_args()

    lay = DAY / "LAYOUTS.csv"
    digest = sha256(lay)
    if digest != SEALED_SHA256:
        raise SystemExit(f"LAYOUTS.csv sha256 {digest} != sealed {SEALED_SHA256}: the list was changed, stop")
    rows = list(csv.DictReader(open(lay)))
    for r in rows:
        for k in r:
            if k.endswith("_px"):
                r[k] = float(r[k])
    cm_px = json.loads((DAY / "layouts_meta.json").read_text())["cm_per_px"]
    print(f"LAYOUTS.csv sealed ({digest[:12]}…), {len(rows)} rollouts; checkpoint "
          f"{'found' if CKPT.exists() else 'MISSING'} ({CKPT.name[:8]}); stop at {MAX_FAILURES} failures or {len(rows)}")
    if not CKPT.exists():
        raise SystemExit("checkpoint snapshot c65c98b not in the HF cache")

    labels = a.record_dir / "labels.csv"
    done = list(csv.DictReader(open(labels))) if labels.exists() else []
    fails = sum(r["outcome"] != "success" for r in done)

    import cv2

    fake = None
    if a.fake_frame:
        fake = cv2.cvtColor(cv2.imread(str(a.fake_frame)), cv2.COLOR_BGR2RGB)
        if fake.shape[:2] != (480, 640):
            fake = cv2.resize(fake, (640, 480))

    if a.step:
        return steered(a, rows, cm_px, labels, done, fails, fake)

    if a.dry_run:
        row = rows[a.start - 1]
        print(f"already labelled: {len(done)} rollouts, {fails} failures")
        cap = None if fake is not None else cv2.VideoCapture(top_camera_index())
        if cap is not None and not cap.isOpened():
            print("top camera not available: placement not checked (use --fake-frame)")
        else:
            place(row, cm_px, cap, fake, once=True)
        print("trial command:", shlex.join(trial_cmd(row, a.record_dir)))
        return

    a.record_dir.mkdir(parents=True, exist_ok=True)
    new = not labels.exists()
    for row in rows[a.start - 1:]:
        if fails >= MAX_FAILURES:
            print(f"\nstop: {fails} manipulation failures"); break
        thermal_pause()
        cap = cv2.VideoCapture(top_camera_index())
        res = place(row, cm_px, cap, fake)
        cap.release()
        if os.environ.get("DISPLAY"):
            cv2.destroyAllWindows()
        input("  Bowls placed. Press Enter to start the trial (the arm moves)...")
        subprocess.run(trial_cmd(row, a.record_dir), check=True)
        outcome = ask_outcome()
        rp = last_rollout_path(a.record_dir)
        root, ep = rp.split("#episode=")
        lab = {"episode_index": int(ep), "reset_start": -1, "rollout": row["rollout"], "instruction": row["instruction"], "stratum": row["stratum"],
               "base_episode": row["base_episode"], "outcome": "success" if outcome == "success" else "failure",
               "failure_mode": "" if outcome == "success" else outcome, "rollout_path": rp,
               "blue_residual_cm": round(res["blue"], 2), "pink_residual_cm": round(res["pink"], 2),
               "labeller": getpass.getuser(), "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with open(labels, "a", newline="") as f:
            w = csv.DictWriter(f, LABEL_FIELDS)
            if new:
                w.writeheader(); new = False
            w.writerow(lab)
        if outcome != "success":
            fails += 1
            print(f"  failure {fails}/{MAX_FAILURES}: mark its onset", flush=True)
            one = a.record_dir / f"onset_todo_ep{int(ep):03d}.csv"          # one row: label_onset shows this episode only
            one.write_text(f"episode_index,outcome\n{int(ep)},failure\n")
            cmd = [sys.executable, str(ROOT / "projects/vla-jepa-world-model/recording/label_onset.py"),
                   "--dataset", root, "--outcomes", str(one), "--out", str(a.record_dir / "onsets.csv")]
            if os.environ.get("DISPLAY"):
                subprocess.run(cmd)
            else:
                print(f"  (no display: after the session, before any scoring: {shlex.join(cmd)})")
    print(f"\nsession: {sum(1 for _ in open(labels)) - 1} rollouts labelled, {fails} failures. Next: onsets (if pending), "
          "then offline: s29_score.py baselines, vla_jepa_surprise.py replay --episodes, s29_score.py score "
          f"--labels {labels} --onsets {a.record_dir / 'onsets.csv'}.")


def steered(a, rows, cm_px, labels, done, fails, fake):
    import cv2

    labelled = {int(r["rollout"]) for r in done}
    nxt = next((int(r["rollout"]) for r in rows if int(r["rollout"]) not in labelled), None)
    if a.step == "status" or (a.step != "status" and a.rollout is None):
        print(json.dumps({"labelled": len(done), "failures": fails, "next_rollout": nxt,
                          "stop": fails >= MAX_FAILURES or nxt is None}))
        return
    row = rows[a.rollout - 1]
    if a.step in ("place", "trial"):
        if fails >= MAX_FAILURES:
            raise SystemExit(f"stop rule reached: {fails} failures")
        if a.rollout in labelled:
            raise SystemExit(f"rollout {a.rollout} is already labelled")
        cap = None if fake is not None else cv2.VideoCapture(top_camera_index())
        res = place(row, cm_px, cap, fake, once=True)
        if cap is not None:
            cap.release()
        ok = all(v is not None and v <= PLACE_TOL_CM for v in res.values())
        state = a.record_dir / "placement.json"
        a.record_dir.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"rollout": a.rollout, "residual_cm": res, "ok": ok,
                                     "t": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
        print(json.dumps({"rollout": a.rollout, "stratum": row["stratum"], "instruction": row["instruction"],
                          "ready": ok, "residual_cm": res}))
        if a.step == "place":
            return
        if not ok:
            raise SystemExit("not placed: both bowls must be within tolerance before the trial")
        thermal_pause()
        t0 = time.time()
        # the harness asks for Enter before the first trial even with --no-prompt: the placement check above is it
        subprocess.run(trial_cmd(row, a.record_dir), check=True, input="\n", text=True)
        print(json.dumps({"rollout": a.rollout, "trial_wall_s": round(time.time() - t0, 1),
                          "rollout_path": last_rollout_path(a.record_dir)}))
        return
    if a.step == "label":
        if a.outcome is None:
            raise SystemExit("--outcome needed")
        if a.rollout in labelled:
            raise SystemExit(f"rollout {a.rollout} is already labelled (rows are never rewritten)")
        rp = last_rollout_path(a.record_dir)
        root, ep = rp.split("#episode=")
        place_state = json.loads((a.record_dir / "placement.json").read_text())
        res = place_state["residual_cm"] if place_state.get("rollout") == a.rollout else {"blue": None, "pink": None}
        lab = {"episode_index": int(ep), "reset_start": -1, "rollout": row["rollout"], "instruction": row["instruction"],
               "stratum": row["stratum"], "base_episode": row["base_episode"],
               "outcome": "success" if a.outcome == "success" else "failure",
               "failure_mode": "" if a.outcome == "success" else a.outcome, "rollout_path": rp,
               "blue_residual_cm": None if res["blue"] is None else round(res["blue"], 2),
               "pink_residual_cm": None if res["pink"] is None else round(res["pink"], 2),
               "labeller": "owner (via Claude Code)", "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        new = not labels.exists()
        with open(labels, "a", newline="") as f:
            w = csv.DictWriter(f, LABEL_FIELDS)
            if new:
                w.writeheader()
            w.writerow(lab)
        fails2 = fails + (a.outcome != "success")
        print(json.dumps({"rollout": a.rollout, "outcome": a.outcome, "episode_index": int(ep), "failures": fails2,
                          "stop": fails2 >= MAX_FAILURES or a.rollout >= len(rows)}))


if __name__ == "__main__":
    main()
