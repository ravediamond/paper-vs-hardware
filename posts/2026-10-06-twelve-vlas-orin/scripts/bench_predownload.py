"""Pre-download the sweep models in sweep order (sha256-checked by bench_sweep.fetch), keeping MARGIN free on disk.
Models that would not fit are left for bench_sweep.py to fetch once earlier ones are measured and deleted."""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home()))
import bench_sweep as b

MARGIN = 5 * 2**30
SIZES_MIB = {"Octo-Small": 522, "TurboVLA": 822, "SmolVLA": 1074, "Evo-1": 1480, "BitVLA": 1377, "VLA-Adapter": 2650,
             "VLA-JEPA": 4354, "GR00T N1.7": 6001, "π0": 6185, "GR00T N1.5": 6613, "GR00T N1.6": 8737, "π0.5": 12004}


def free_gib():
    return shutil.disk_usage("/").free / 2**30


for name, repo, fname, sha, *_ in b.MODELS:
    dst = b.MODELS_DIR / Path(fname).name
    need = SIZES_MIB[name] * 2**20
    if not dst.exists() and shutil.disk_usage("/").free - need < MARGIN:
        b.log(f"predownload: {name} left for the sweep ({need / 2**30:.1f} GiB would leave < 5 GiB free)")
        continue
    t = time.time()
    path, digest = b.fetch(repo, fname, sha)
    b.log(f"predownload: {name} ok, sha256 {digest[:12]}, {time.time() - t:.0f} s, {free_gib():.1f} GiB free")
b.log("predownload finished")
