"""Package the 182 demos for LeWM training and write the held-out split. Runs on the Mac (no token, the dataset is public).

    python projects/vla-jepa-world-model/wm_train/prepare_data.py                 # resolve + download the pinned revision, write split.json
    python projects/vla-jepa-world-model/wm_train/prepare_data.py --pack          # also write a tarball for a direct upload to the instance
    python projects/vla-jepa-world-model/wm_train/prepare_data.py --transcode --pack   # H.264 224x224 all-intra copy (what training uses), packed

--transcode: the Hub videos are AV1 640x480 with a keyframe every 2 frames; decoding them on the CPU fed the GPU only
~25% of the time (vast.ai smoke 2026-10-08: 0.45 s GPU vs 1.4 s data wait per step). The copy re-encodes every video to
H.264 224x224 (the size training resizes to anyway, same squash of the 4:3 frame), every frame a keyframe, with the same
frame count and timestamps (checked per file), and updates meta/info.json. Same LeRobot v3 layout, same episodes,
parquet and stats: the connector reads it unchanged. Local dir <revision>-h264-224.

The data stays in the LeRobot v3 format the stable-worldmodel connector (`lerobot://`) reads; nothing is converted.
Outputs:
  projects/vla-jepa-world-model/wm_train/split.json                     (git) dataset repo + revision, train / val episodes, how they were chosen
  ~/robot-data/wm/so101_bowls_all_trimmed/<revision>/      the snapshot (not git)
  ~/robot-data/wm/so101_bowls_all_trimmed-<rev12>.tar     with --pack, plus a .sha256 next to it (not git)

Held-out split: whole grid layouts (projects/vla-jepa-world-model/prechecks/out/layouts/index.csv, bowl cells in the first front frame), shuffled
with a fixed seed, are moved to validation until >= 10% of the episodes are held out; both tasks must be present.
The VLA saw every episode; this split is only for the world model's own validation loss and collapse checks.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "ravediamond/so101_bowls_all_trimmed"
HERE = Path(__file__).resolve().parent
LAYOUTS = HERE.parent / "prechecks" / "out" / "layouts" / "index.csv"
SEED = 1234
VAL_FRACTION = 0.10
SIZE = 224


def frame_times(path: Path) -> list[str]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
                          "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return [t.strip(",") for t in out.split() if t.strip(",")]


def transcode_one(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an",
                    "-vf", f"scale={SIZE}:{SIZE}:flags=area", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-g", "1", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-video_track_timescale", "15360",
                    str(dst)], check=True)
    a, b = frame_times(src), frame_times(dst)
    if len(a) != len(b) or any(abs(float(x) - float(y)) > 1e-3 for x, y in zip(a, b)):
        raise RuntimeError(f"{dst}: {len(b)} frames vs {len(a)}, or timestamps differ")
    return f"{src.name}: {len(b)} frames ok"


def transcode(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("videos", ".cache"))
    videos = sorted(src.glob("videos/**/*.mp4"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for msg in pool.map(lambda v: transcode_one(v, dst / v.relative_to(src)), videos):
            print("  ", msg, flush=True)
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key, feat in info["features"].items():
        if feat.get("dtype") == "video":
            feat["shape"] = [SIZE, SIZE, 3]
            feat["info"].update({"video.height": SIZE, "video.width": SIZE, "video.codec": "h264", "video.g": 1,
                                 "video.crf": 18, "video.preset": "medium", "video.pix_fmt": "yuv420p",
                                 "transcoded_from": "av1 640x480 g2 (Hub revision, see split.json)"})
    info_path.write_text(json.dumps(info, indent=4))


def make_split(rows: list[dict]) -> dict:
    by_layout: dict[str, list[dict]] = {}
    for r in rows:
        by_layout.setdefault(r["layout_label"], []).append(r)
    labels = sorted(by_layout)
    random.Random(SEED).shuffle(labels)
    need = int(round(VAL_FRACTION * len(rows)))
    val_labels, val = [], []
    for lab in labels:
        if len(val) >= need:
            break
        val_labels.append(lab)
        val += by_layout[lab]
    assert len({r["task"] for r in val}) == 2, "validation misses a task: change SEED"
    val_eps = sorted(int(r["episode"]) for r in val)
    train_eps = sorted(int(r["episode"]) for r in rows if int(r["episode"]) not in set(val_eps))
    return {"val_layouts": sorted(val_labels), "train_episodes": train_eps, "val_episodes": val_eps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", help="dataset commit (default: current main, resolved and pinned)")
    ap.add_argument("--root", type=Path, default=Path.home() / "robot-data" / "wm")
    ap.add_argument("--pack", action="store_true", help="also write a tarball for a direct upload")
    ap.add_argument("--no-download", action="store_true", help="only (re)write split.json")
    ap.add_argument("--transcode", action="store_true", help="H.264 224x224 all-intra copy for training (packed if --pack)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    revision = args.revision or HfApi().dataset_info(REPO).sha
    rows = list(csv.DictReader(open(LAYOUTS)))
    split = make_split(rows)
    out = {"dataset": REPO, "revision": revision, "seed": SEED, "rule": "whole grid layouts until >= 10% of episodes",
           "n_train": len(split["train_episodes"]), "n_val": len(split["val_episodes"]), **split}
    (HERE / "split.json").write_text(json.dumps(out, indent=1))
    print(f"split.json: {out['n_train']} train / {out['n_val']} val episodes, revision {revision[:12]}, "
          f"val layouts {out['val_layouts']}")

    if args.no_download:
        return
    local = args.root / "so101_bowls_all_trimmed" / revision
    snapshot_download(REPO, repo_type="dataset", revision=revision, local_dir=local)
    info = json.loads((local / "meta" / "info.json").read_text())
    assert info["total_episodes"] == len(rows) == 182, info["total_episodes"]
    print(f"snapshot: {local} ({info['total_episodes']} episodes, {info['total_frames']} frames)")

    name = revision
    if args.transcode:
        name = f"{revision}-h264-{SIZE}"
        print(f"transcoding {len(list(local.glob('videos/**/*.mp4')))} videos -> {local.parent / name}")
        transcode(local, local.parent / name)
    if args.pack:
        tar = args.root / f"so101_bowls_all_trimmed-{name[:12]}{name[len(revision):]}.tar"
        subprocess.run(["tar", "-cf", str(tar), "--exclude", ".cache", "-C", str(local.parent), name], check=True)
        digest = hashlib.sha256()
        with open(tar, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        Path(f"{tar}.sha256").write_text(f"{digest.hexdigest()}  {tar.name}\n")
        print(f"tarball: {tar} ({tar.stat().st_size / 1e9:.2f} GB), sha256 {digest.hexdigest()[:16]}...")


if __name__ == "__main__":
    main()
