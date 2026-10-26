# Running LeRobot on Jetson Orin (JetPack 6.2 / CUDA 12.6 / Python 3.12)

Verified end-to-end on a **Jetson Orin NX 16GB**, JetPack 6.2 (L4T R36.4.3), CUDA 12.6, aarch64.

**2026-08-11 — this recipe is now also packaged as a Dockerfile**: [`ravediamond/lerobot-jetson`](https://github.com/ravediamond/lerobot-jetson) turns Sections 2–5 below into a single buildable image (source-built torch + torchcodec, same flags), instead of the jetson-containers path in Sections 9–13, which is confirmed stalled (see `ROADMAP_CONTRIBUTION_OPPORTUNITIES.md`'s 2026-08-11 update — no human-merged PR there since June, original maintainer stepped away mid-2025). This guide's own gotchas (Sections 2 and 4) are exactly what the Dockerfile encodes, so if the image build fails, check here first for the matching error.

**2026-08-26 update — both Dockerfiles (`Dockerfile`, `Dockerfile.fast`) now build green end-to-end**, including a real `import lerobot` and a policy import (SmolVLA), not just torch/torchcodec. Bugs found and fixed along the way, none covered by this guide's Sections 2–5 since they're specific to *packaging* the stack, not the bare-metal build:

- **torchvision ABI mismatch** — lerobot depends on torchvision; pip pulls a generic PyPI wheel that imports fine but whose compiled ops silently fail to register against a from-source torch build (`RuntimeError: operator torchvision::nms does not exist`), breaking almost every lerobot import. Same class of bug as this guide's own torchcodec pip-swap gotcha (Section 5), just undiscovered longer because nothing had exercised `import lerobot` itself. Fix: build torchvision 0.26.0 from source too (pairs with torch 2.11 per the pytorch/vision compatibility matrix), reassert after the lerobot install same as torchcodec.
- **`torch.distributed` stub breaks `lerobot-train`** — this guide's torch build sets `USE_DISTRIBUTED=0` (reasonable on a single-GPU board), which leaves `torch.distributed` as a stub with no `is_initialized()`. lerobot's `is_main_process()` (`src/lerobot/distributed/utils.py`) calls it unconditionally — `AttributeError`, blocks `lerobot-train` entirely. Sed-patched a guard into the lerobot install step as an immediate unblock; real fix is `USE_DISTRIBUTED=1 USE_GLOO=1` (Gloo = CPU rendezvous backend, enough to satisfy the check without needing NCCL/real multi-GPU) — staged in the Dockerfile but **not yet rebuilt** as of this note, since it forces a full multi-hour torch recompile.
- **`USE_MEM_EFF_ATTENTION=1` also staged, untested** — Section 2's OOM writeup was specifically about flash-attention's backward kernels; mem-eff was disabled alongside it without independently confirming it also OOMs. Worth the retry now that the rest of the build is stable — every policy forward pass otherwise falls back to plain SDPA. Bundled with the `USE_DISTRIBUTED` change above for one supervised rebuild rather than two.
- Smaller fixes: `pip` binary missing from PATH (`ensurepip` only creates `pip3`/`pip3.12`, not a bare `pip`), no baked-in torch hub cache (ACT's ResNet18 backbone re-downloaded every fresh container).
- ~~**Known-broken**: GHCR (`ghcr.io/ravediamond/lerobot-jetson`) — an earlier broken image got tagged and a push started before the torchvision fix landed; the push was killed mid-transfer for bandwidth reasons. Treat GHCR as stale until a clean, complete `docker push` happens from a real network, not the phone hotspot this was built on.~~ **Resolved 2026-08-31** — see entry below: the verified `USE_DISTRIBUTED=1`/`USE_MEM_EFF_ATTENTION=1` rebuild was pushed clean to both `:public` and `:latest`.
- **Not yet attempted**: a multi-stage rewrite (`Dockerfile.multistage`, draft, untested) to fix `Dockerfile`'s 23.9GB image size (vs `Dockerfile.fast`'s 12.7GB) — the from-source build never cleans up compiler toolchain / git source trees / build artifacts from its layers.

**2026-08-31 — full rebuild with `USE_DISTRIBUTED=1 USE_GLOO=1` and `USE_MEM_EFF_ATTENTION=1` confirmed working.** The two flags staged in the 2026-08-26 entry above (commit `7bf2bf8`) were built and verified on real Orin hardware, not just staged. Disk was pruned first (89%→54% used, freed ~54GB of stale `jetson-containers` debug images) since a from-source rebuild needs the headroom.

Verified live in the built `lerobot-jetson:public` image (`docker run --rm --runtime nvidia lerobot-jetson:public python3 -c ...`):
- `torch.distributed.is_available()` → `True`, `is_gloo_available()` → `True`, `is_initialized()` → `False` with **no crash** (the old stub-AttributeError bug from the 2026-08-26 entry is gone — `torch.distributed` is a real module now, not a stub).
- `lerobot.distributed.utils.is_main_process()` → `True`, no crash — confirms `lerobot-train` is unblocked without needing the sed-patched guard (the sed patch, Dockerfile step 24/29, is still applied too — harmless belt-and-suspenders, now dead code in practice since the real fix makes it a no-op).
- `torch.backends.cuda.mem_efficient_sdp_enabled()` → `True` — mem-efficient attention is live.
- `torch==2.11.0`, CUDA available, device `Orin`; `torchcodec==0.11.0a0+c95b831`, `torchvision==0.26.0+336d36e` (git-hash-suffixed → confirmed real source builds, not generic PyPI wheels); `lerobot==0.6.2` imports; `SmolVLAPolicy` imports clean.
- Build succeeded end-to-end: `BUILD_EXIT_CODE=0`.

### Full build timing (`docker build -t lerobot-jetson:public .`, 2026-08-31, cold — no cache)

| Step | What | Duration |
|---|---|---|
| 1–4 | Docker internals + `FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0` | ~0.1s |
| 5 | `apt-get install python3.12, libopenblas0, ninja, ...` | 38.7s |
| 6 | Add NVIDIA CUDA apt repo + `libcusparselt0` | 12.5s |
| 7 | Install `uv` | 2.0s |
| 8 | `uv venv --python python3.12 /opt/venv` | 0.3s |
| 9 | `ensurepip` + pip install build deps (cmake, ninja, numpy, ...) | 18.1s |
| 10 | `git clone` PyTorch v2.11.0 source | 711.3s (11.9min) |
| 11 | `WORKDIR /opt/pytorch-src` | 0.0s |
| **12** | **Build PyTorch** (`USE_CUDA=1 USE_DISTRIBUTED=1 USE_GLOO=1 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=1 USE_NVSHMEM=0`, `MAX_JOBS=3`) + install wheel | **18371.5s (306.2min / 5.10hr)** |
| 13 | Clone + build dav1d 1.5.1 from source | 25.2s |
| 14 | Install nv-codec-headers n13.0.19.0 | 1.9s |
| 15 | Clone FFmpeg n8.1 | 8.7s |
| 16 | `WORKDIR /opt/ffmpeg-src` | 0.0s |
| 17 | Configure + build FFmpeg (NVDEC/CUVID, NVENC disabled, libdav1d) | 449.2s (7.5min) |
| 18 | `pip install pybind11` | 3.3s |
| 19 | Clone torchcodec v0.11.0 | 4.4s |
| 20 | `WORKDIR /opt/torchcodec-src` | 0.0s |
| 21 | Build torchcodec (`ENABLE_CUDA=1 CUDAARCHS=87`) + install wheel | 123.0s (2.1min) |
| 22 | Clone torchvision v0.26.0 | 4.8s |
| 23 | `WORKDIR /opt/vision-src` | 0.0s |
| 24 | Build torchvision (`FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.7`) + install wheel | 175.5s (2.9min) |
| 25 | `WORKDIR /opt/lerobot` | 0.0s |
| 26 | Clone lerobot `main` + `pip install -e ".[feetech,aloha,pusht,dynamixel,pi,smolvla,test]"` | 116.8s (1.9min) |
| 27 | Sed-patch `is_main_process()` distributed guard | 0.2s |
| 28 | Force-reinstall torchcodec + torchvision wheels (undo any pip dependency-resolution swap) | 2.9s |
| 29 | `apt-get install` speech-dispatcher, ffmpeg CLI | 17.6s |
| 30 | Symlink `pip` → `pip3` | 0.2s |
| 31 | Pre-warm torch hub cache (download + load ACT's ResNet18 backbone) | 15.9s |
| 32 | `WORKDIR /opt/lerobot` (final) | 0.0s |
| 33 | Export image layers | 45.5s |
| **Total** | | **19964.5s ≈ 5h 32m 45s** |

PyTorch compilation (step 12) is **92% of total build time**. Everything else — ffmpeg, torchcodec, torchvision, lerobot itself — is under 15 minutes combined. No OOM, no disk pressure, no crashes across the full run; memory stayed under ~5GB used throughout (16GB board), confirming `MAX_JOBS=3` is comfortably conservative for this board, not just OOM-avoidance superstition.

## Context

Related issues:
- [#819](https://github.com/huggingface/lerobot/issues/819) — Jetson support (closed/stale, maintainer invited investigation)
- [#2363](https://github.com/huggingface/lerobot/issues/2363) — Jetson torchcodec/GPU training
- [#3636](https://github.com/huggingface/lerobot/issues/3636) — SmolVLA wrong inference on Jetson ARM64
- [#3832](https://github.com/huggingface/lerobot/issues/3832) — 0.7.0 roadmap, item 4.10: torchcodec migration flagged as suspected cause of RAM spikes on constrained hardware

**Root blocker (as of 2026-07):** lerobot requires `python>=3.12` and `torchcodec>=0.11.0,<0.12.0` on aarch64+CUDA (`pyproject.toml`). No official channel — not NVIDIA's redist, not the jetson-ai-lab community index — ships a **cp312 PyTorch wheel for Jetson**. The community index (`pypi.jetson-ai-lab.io/jp6/cu126`) has torch 2.11.0 but only for **cp310**, and its torchcodec ceiling is 0.10.0 (below lerobot's floor). This blocks `pip install lerobot` on Jetson entirely regardless of torchcodec.

**This guide closes that gap**: build torch 2.11.0 and torchcodec 0.11.0 from source for cp312, targeting SM_87 (Orin).

---

## 1. Prerequisites

```bash
sudo nvpmodel -m 0   # MAXN_SUPER power mode, unlocks all cores
sudo jetson_clocks

sudo apt-get update
sudo apt-get install -y build-essential ninja-build git tmux
```

Install Python 3.12 + create a build venv (system Python on JetPack 6.2 is 3.10):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv ~/torch-build-venv --python 3.12
source ~/torch-build-venv/bin/activate
python3 -m ensurepip --upgrade   # venv may not ship pip by default
python3 -m pip install --upgrade pip setuptools wheel ninja cmake pyyaml numpy typing_extensions
```

> **Gotcha:** after activating the venv, double check `which pip` — on this system `pip` kept resolving to `/usr/local/bin/pip` (system pip) even inside an active venv, silently installing to `--user` instead of the venv. Always use **`python3 -m pip ...`**, never bare `pip`, to guarantee you're using the venv's own pip.

---

## 2. Build PyTorch 2.11.0 (cp312, CUDA 12.6, SM_87)

```bash
git clone --recursive --branch v2.11.0 https://github.com/pytorch/pytorch.git ~/pytorch-src
cd ~/pytorch-src
source ~/torch-build-venv/bin/activate

export USE_CUDA=1
export USE_CUDNN=1
export TORCH_CUDA_ARCH_LIST="8.7"
export PYTORCH_BUILD_VERSION=2.11.0
export PYTORCH_BUILD_NUMBER=1
export MAX_JOBS=3
export USE_DISTRIBUTED=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0

python3 setup.py bdist_wheel
```

Wheel lands at `dist/torch-2.11.0-cp312-cp312-linux_aarch64.whl`.

### Gotchas hit during this build

- **`ModuleNotFoundError: No module named 'typing_extensions'` during `cmake/Codegen.cmake`'s header generation** — never surfaced on this bare-metal build (the reused venv already had it as a transitive dep of something installed earlier), but hit immediately in a clean container build using the same prerequisite list above. `typing_extensions` isn't itself in that list — add it alongside `ninja cmake pyyaml numpy` in a fresh environment, don't assume it's implied.
- **`TORCH_CUDA_ARCH_LIST="8.7"# comment` with no space before `#`** — bash does not treat `#` as a comment start when it's glued to the value with no preceding whitespace, so the arch string literally becomes `8.7#`. CMake then fails: `Found Unknown CUDA Architecture Name in CUDA_SELECT_NVCC_ARCH_FLAGS: 8.7#`. Fix: never glue an inline comment directly after a quoted value; put a space before `#`, or put the comment on its own line.
- **OOM (`exit code 137`) compiling flash-attention backward kernels** (`flash_bwd_hdim128_bf16_sm80.cu.o` etc.) — these are heavily templated CUDA kernels; each `cicc`/`nvcc` process can eat 1.5GB+ RAM, and with `MAX_JOBS=6` on a 16GB device several run in parallel and get SIGKILLed by the OOM killer. Confirm via `sudo dmesg | grep -iE "out of memory|oom-kill|killed process"`. Fix: set `USE_FLASH_ATTENTION=0` and `USE_MEM_EFF_ATTENTION=0` (not required for lerobot's policies; PyTorch falls back to standard SDPA) and drop `MAX_JOBS` to 3.
- **CMake cache staleness** — after changing `USE_FLASH_ATTENTION` and rerunning `setup.py bdist_wheel`, the build silently kept compiling flash-attention kernels anyway. `setup.py` does not force a full CMake reconfigure once `build/CMakeCache.txt` exists, so the stale `USE_FLASH_ATTENTION:BOOL=ON` persisted. Fix: from inside `build/`, force it directly: `cmake -DUSE_FLASH_ATTENTION=OFF -DUSE_MEM_EFF_ATTENTION=OFF .` (requires the venv's cmake ≥3.27 on `PATH` — the system cmake, 3.22.1 from apt, is too old and will fail with `CMake 3.27 or higher is required`).
- **`PYTORCH_BUILD_NUMBER must be set when PYTORCH_BUILD_VERSION is set`** — happens if you resume a build in a fresh shell/tmux session without re-exporting all env vars. They do not persist across new tmux windows/sessions. Re-export everything before rerunning.
- Build took several hours; run inside `tmux` (`tmux new -s torchbuild`) so it survives SSH disconnects. The tmux server is independent of the SSH client — closing the terminal window is always safe as long as you're genuinely inside the tmux session (check with `echo $TMUX`), regardless of whether the detach keystroke visibly registered.

### Install + verify

```bash
python3 -m pip install ~/pytorch-src/dist/torch-2.11.0-cp312-cp312-linux_aarch64.whl
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.11.0  True  Orin
```

---

## 3. Build FFmpeg with NVDEC support

```bash
git clone https://github.com/FFmpeg/nv-codec-headers.git ~/nv-codec-headers-src
cd ~/nv-codec-headers-src && sudo make install

git clone https://github.com/FFmpeg/FFmpeg.git ~/ffmpeg-src
cd ~/ffmpeg-src
./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp \
  --extra-cflags=-I/usr/local/cuda/include \
  --extra-ldflags=-L/usr/local/cuda/lib64 \
  --disable-static --enable-shared
make -j3
sudo make install
sudo ldconfig
```

> **License note:** `--enable-nonfree` is required for NVENC/NVDEC support, but a nonfree ffmpeg build **cannot be legally redistributed** as a binary. This is fine for local builds/testing. If you intend to publish prebuilt torchcodec wheels for the community, either rebuild ffmpeg without `--enable-nonfree` (NVDEC-only decode still needs checking whether it's affected) or document "build ffmpeg yourself" rather than shipping a binary.

---

## 4. Build torchcodec 0.11.0 (CUDA-enabled)

```bash
git clone --branch v0.11.0 --depth 1 https://github.com/pytorch/torchcodec.git ~/torchcodec-src
cd ~/torchcodec-src
source ~/torch-build-venv/bin/activate

export ENABLE_CUDA=1
export CUDAARCHS=87
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 python3 setup.py bdist_wheel

python3 -m pip install dist/torchcodec-0.11.0a0-cp312-cp312-linux_aarch64.whl
```

### Gotchas hit during this build

- **`ValueError: ... set BUILD_AGAINST_ALL_FFMPEG_FROM_S3 ... or I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION`** — torchcodec's `setup.py` refuses to build a wheel without one of these flags set, since normally released wheels bundle ffmpeg binaries and the license implications need explicit ack. Since we build against our own local ffmpeg (no bundling), setting `I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1` is correct here.
- **`RuntimeError: Could not load libtorchcodec ... Unsupported device: cuda (device type: cuda, variant: ffmpeg)`** on first successful-looking build — the wheel imported fine but had no CUDA support compiled in at all. Root cause: torchcodec's `setup.py` reads `ENABLE_CUDA` from the environment (default `""` = off) and never sets `-DCMAKE_CUDA_ARCHITECTURES` explicitly either. **Both must be set explicitly**: `ENABLE_CUDA=1` (consumed directly by `setup.py`'s `cmake_args`) and `CUDAARCHS=87` (CMake's own native env var for default `CMAKE_CUDA_ARCHITECTURES`, since torchcodec's `cmake_args` list doesn't pass it and CMake would otherwise fall back to an old default like SM_52).
- **Requirements file doesn't exist** — the `v0.11.0` tag has no `requirements.txt`; dependencies are declared in `pyproject.toml`. Skip that pip step, torch (already installed) satisfies the main runtime dep for building.

### Verify GPU decode actually works (not just import)

```bash
python3 -c "
from torchcodec.decoders import VideoDecoder
d = VideoDecoder('test/resources/nasa_13013.mp4', device='cuda')
f = d[0]
print(f.data.shape, f.data.device, f.data.dtype)
"
# torch.Size([3, 270, 480]) cuda:0 torch.uint8
```

---

## 5. Install LeRobot

```bash
git clone --depth 1 https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
source ~/torch-build-venv/bin/activate
python3 -m pip install -e ".[dataset]" --no-build-isolation
```

### Critical gotcha: pip silently swaps out the custom torchcodec build

Our locally-built wheel is versioned `0.11.0a0` (an alpha/dev tag). Per PEP 440 version ordering, `0.11.0a0 < 0.11.0`, so when `pip install -e ".[dataset]"` resolves lerobot's `torchcodec>=0.11.0,<0.12.0` constraint, it does **not** consider our local build to satisfy the floor, and pulls **`torchcodec==0.11.1` from PyPI** instead — silently uninstalling our custom build.

That PyPI wheel is a generic manylinux aarch64/cp312 build and **crashes on Jetson** with:
```
OSError: libnvrtc.so.13: cannot open shared object file: No such file or directory
```
It was built against CUDA 13's runtime library naming, but JetPack 6.2 ships CUDA 12.6. It's simply the wrong ABI for this platform — pip has no way to know that from version numbers alone.

**Fix — reinstall the custom build after any `pip install` touching lerobot's deps:**
```bash
python3 -m pip install --force-reinstall --no-deps dist/torchcodec-0.11.0a0-cp312-cp312-linux_aarch64.whl
```
(from `~/torchcodec-src`, using the wheel built in step 4). Verify again with the decode snippet from step 4 after every reinstall.

### Also watch for: legacy `setup.py develop` / "UNKNOWN" package

If you ever see `pip` say `Defaulting to user installation because normal site-packages is not writeable` while your venv is active, and end up with a package literally named `UNKNOWN==0.0.0` installed instead of `lerobot`, it means `pip` resolved to the *system* pip binary again (see the gotcha in step 1) rather than the venv's. Uninstall the bogus package and rerun using `python3 -m pip install -e ".[dataset]"` explicitly.

---

## 6. Verified working stack (2026-07-17)

| Component | Version | Notes |
|---|---|---|
| JetPack | 6.2 (L4T R36.4.3) | |
| CUDA | 12.6 | |
| Python | 3.12.13 | via `uv`, since system Python is 3.10 |
| PyTorch | 2.11.0 | source build, cp312, SM_87, flash-attention disabled |
| FFmpeg | git main (7.x/8.x line) | `--enable-nonfree --enable-cuda-nvcc --enable-libnpp` |
| torchcodec | 0.11.0a0 | source build, `ENABLE_CUDA=1 CUDAARCHS=87` |
| LeRobot | 0.6.1 | `pip install -e ".[dataset]"` |

GPU decode confirmed with a real `.cuda()` tensor output (not just a clean import) — see verification snippet in step 4.

---

## 7. Open question this does *not* resolve: #3636 (SmolVLA wrong inference on Jetson)

#3636 hypothesizes that the pyav/torchcodec video-decoder discrepancy causes wrong SmolVLA actions during live inference on Jetson. Investigation while building this stack:

- **Pixel-level diff measured** (same frame, same stream index, `nasa_13013.mp4`): max abs diff 17/255 (~6.7%), mean abs diff 0.37/255 (~0.15%), only ~1.8% of pixels differ by more than 2/255. A real, small, systematic difference exists (likely differing YUV→RGB conversion / chroma upsampling between libswscale and pyav's internal path) — but this magnitude is typical decoder-implementation noise, not large enough by itself to explain "random/wildly wrong" robot actions.
- **More fundamentally**: `decode_video_frames_pyav` / `decode_video_frames_torchcodec` (`src/lerobot/datasets/video_utils.py`) are only used to load **recorded episode videos from a dataset** (training / eval-from-dataset). Live robot inference reads frames directly from the robot's cameras (OpenCV/pyrealsense), never through these functions. So the decoder-backend theory in #3636 doesn't structurally connect to the code path the reporter was actually exercising (live inference).
- **Likely real cause, unverified**: the reporter's Jetson used PyTorch 2.5.0 vs. 2.7.0 on the x86_64 reference machine — plausible candidates are numerical differences in the policy's forward pass between torch versions, or an ARM64 NEON vs. x86_64 AVX kernel-path discrepancy in that older torch build. Worth re-testing with the torch 2.11.0 build from this guide (much closer torch version to the x86_64 reference) to see if the discrepancy narrows or disappears.

This should be posted as a correction/redirect on #3636 rather than confirming the original decoder theory.

**Update (2026-08-03) — a community member beat us to the structural finding, but doesn't have Orin hardware.** `shaoningyu1231` posted a detailed comment on #3636 (2026-07-31) reaching the same core conclusion independently — decode backend isn't in the live-inference path — with far more rigorous cross-architecture measurement (x86/Graviton2/QEMU, cpuflags matrix, action-level sensitivity probe on `smolvla_base`). They explicitly say: *"I don't have Orin hardware"* and list exactly the experiments they'd want run on real Orin: fp32 end-to-end (isolate fp16/bf16 numerics), a camera-frame tensor diff between machines, and a fixed-input vision-encoder comparison. **This is the opening, not a scoop** — this build (Section 6) gives torch 2.11.0 on real Orin, much closer to the x86 reference (2.7.0) than the reporter's 2.5.0. Concrete next action: run their three suggested experiments on this hardware and post results as a comment crediting their analysis, rather than re-deriving the same structural point from scratch.

### Personal test plan — blocked on Orin arrival (2026-08-03)

**Status: waiting on hardware.** Orin not in hand yet — this is queued as the first thing to run once it arrives, no SO-101 or camera rig needed.

No robot, no live camera capture required — reuse the exact dataset clip `shaoningyu1231` already used (`lerobot/svla_so101_pickplace`), so results are directly comparable to their numbers instead of a fresh, non-matching setup.

1. **Fp32-only test.** Run `smolvla_base` end-to-end in fp32, no autocast/half, on the dataset clip. Compare against their fp32 vs bf16-autocast delta (0.0004 std on their side) — if this Jetson shows something bigger, that's evidence of torch-2.5.0/JetPack-numerics being the real cause rather than decode.
2. **Camera-frame tensor diff.** Decode the same frame from the same clip via pyav on this Jetson (real hardware, not QEMU/Graviton2 emulation) and diff against their measured x86 tensor. Confirms whether real Orin silicon matches their aarch64 emulation numbers (expected: yes, bit-identical to the C reference) or reveals something emulation missed.
3. **Vision-encoder comparison.** With one fixed input tensor saved to disk, compare `smolvla_base`'s vision-encoder output on this Jetson (torch 2.11.0, this guide's build) against x86 (torch 2.7.0) — much closer version pairing than the original reporter's torch 2.5.0 vs 2.7.0 gap. If the divergence they measured shrinks here, that's a strong signal it's a torch-version numerics issue, not decode or hardware.

**Deliverable**: post results as a #3636 comment crediting `shaoningyu1231`'s analysis, not a standalone "I fixed it" post — this is confirmation/extension of their work, not a competing finding.

---

## 8. Suggested community follow-up

1. ~~Publish this build recipe (this file, or a docs PR) so others don't have to rediscover the cp312/CUDA-arch/OOM gotchas from scratch.~~ **Done differently (2026-08-11)**: packaged as [`lerobot-jetson`](https://github.com/ravediamond/lerobot-jetson), a runnable Dockerfile, rather than a lerobot-core docs PR — a working image is more directly useful than prose, and doesn't need lerobot maintainer review to ship. A docs PR pointing at the image is still open as a lighter-weight follow-up.
2. Post the corrected root-cause analysis on #3636.
3. Reply to #819 (maintainer `imstevenpmwork` invited investigation there) with this recipe, referencing #2363, #3636, and roadmap #3832 §4.10.
4. Reproduce the actual SmolVLA robot task on this stack (SO-101 + Jetson) to confirm inference correctness end-to-end, not just that the stack imports and decodes.
5. ~~Contribute the fixes found in Section 10 to `dusty-nv/jetson-containers` so lerobot's official container "just works" on Jetson going forward~~ **Done (PR #1733, 2026-07-31)** — but confirmed stalled (0 reviews after 11+ days, no human-merged PR in that repo since June — see `ROADMAP_CONTRIBUTION_OPPORTUNITIES.md`'s 2026-08-11 update for the full maintainer-bandwidth finding). Left open as a free shot; `lerobot-jetson` (item 1) is the path that doesn't wait on it.

---

## 9. Faster path: `jetson-containers` already has (most of) this recipe

Before doing the from-scratch build in Sections 2-5, it's worth knowing that **`dusty-nv/jetson-containers`** (the repo that actually generates the wheels published at `pypi.jetson-ai-lab.io`, distinct from the `jetson-ai-lab` docs/tutorials site) already declares build recipes for this exact combination:

- `packages/ml/pytorch/version.py` already maps JetPack 6.2 / CUDA 12.6 → **torch 2.11**, correctly.
- `packages/ml/pytorch/torchcodec/config.py` already declares the pairing:
  ```python
  torchcodec('0.11.0', pytorch='2.11', depends=['ffmpeg:8.1'], requires='>=36'),
  ```
- `packages/build/python/config.py` already supports **Python 3.12** for JetPack 6 (`python('3.12', '>=34')`).

In principle, this should build the same stack as Sections 2-5 using their Docker pipeline (which has ccache and layer caching, likely faster than a bare-metal rebuild):

```bash
PYTHON_VERSION=3.12 jetson-containers build torchcodec:0.11.0-pytorch:2.11
```

**What's not yet confirmed:** whether this exact (cp312 + JetPack 6.2 + torch 2.11 + torchcodec 0.11.0) combination has actually been *built and published* to the public wheel index, since CI may only build/publish a default Python version per L4T release (likely still cp310). The recipe existing in the repo is not the same as a wheel existing on the index — this needs verifying on real hardware before relying on it.

**Recommended order of operations for someone hitting this fresh:**
1. Try the `jetson-containers build` command above first — cheapest option if it already works.
2. If it fails, you now have a precise, reproducible bug report for `dusty-nv/jetson-containers` (open an issue there), and Sections 2-5 of this guide are your proven fallback/workaround.
3. Either way, Sections 2-5 remain the reference for understanding *why* each piece is needed and what specifically breaks (arch flags, OOM, CMake cache staleness, `ENABLE_CUDA`), since the Docker pipeline abstracts those away without explaining them.

---

## 10. The `lerobot` package inside `jetson-containers` is stale

Contribution here isn't limited to the torch/torchcodec layer — `jetson-containers` also ships a `packages/physicalAI/lerobot/` package (clones `LEROBOT_BRANCH=main` by default) meant to give users a ready-to-run LeRobot container on Jetson. As of this investigation (2026-07-17) it does not reflect current `lerobot` `main`:

- **`torchcodec` is missing from `depends` entirely.** The Dockerfile only lists `[transformers, opencv, pyav, h5py, diffusers, jupyterlab:myst]`. Since current lerobot's aarch64 `torchcodec` constraint requires `>=0.11.0`, any build of this container either fails to resolve that extra or silently falls back to the `pyav` decoder — which means anyone using this "official" Jetson container for lerobot never actually exercises the torchcodec path, and would report exactly the kind of confusion behind #3636.
- **One dead extra in the install line**: `uv pip install --no-reinstall --no-deps -e ".[feetech, aloha, pusht, dynamixel, pi0, test]"` — `pi0` was renamed to `pi` in current `pyproject.toml` (`pi = ["lerobot[transformers-dep]", "lerobot[scipy-dep]"]`, line ~215). The other extras (`feetech`, `aloha`, `pusht`, `dynamixel`, `test`) still exist unchanged, so this is a small, single-token fix — not a full restructure.
- **Baked-in example commands use a dead CLI.** The image's `.bash_history` seeds `python lerobot/scripts/control_robot.py record --robot-path lerobot/configs/robot/koch.yaml ...` — that script and config format no longer exist. Current lerobot exposes proper installed entry points instead: `lerobot-record`, `lerobot-teleoperate`, `lerobot-calibrate`, `lerobot-find-cameras`, `lerobot-find-port`, `lerobot-setup-motors`, `lerobot-train`, `lerobot-eval`, etc. (see `pyproject.toml [project.scripts]`).
- **Not currently wired to exercise SmolVLA**, despite #3636 being specifically about SmolVLA on Jetson — adding the `smolvla` extra would make this container directly useful for reproducing/validating that issue going forward.

---

## 11. Concrete jetson-containers PR scope

Small, well-bounded diff against `packages/physicalAI/lerobot/Dockerfile`:

1. Add `torchcodec` (and its `ffmpeg:8.1` dependency, already declared elsewhere in the repo) to the package's `depends` list.
2. Fix the install line: `pi0` → `pi`; consider adding `smolvla` so the container can reproduce #3636 directly.
3. Replace the stale `.bash_history` seed commands with current CLI invocations (`lerobot-record`, `lerobot-teleoperate`, etc.).
4. Rebuild and verify with `PYTHON_VERSION=3.12 jetson-containers build lerobot` on real Orin hardware (per Section 9) before submitting.
5. Reference `huggingface/lerobot#819`, `#2363`, `#3636` in the PR description for context on why this matters.

Per `dusty-nv/jetson-containers`'s own `CONTRIBUTING.md`: package = `Dockerfile` with a `#---` YAML metadata header (`name`, `group`, `depends`, `requires`, `test`), must start with `ARG BASE_IMAGE` / `FROM ${BASE_IMAGE}`; verify locally via `jetson-containers build <pkg>`; one logical change per PR; reviewed by `@dusty-nv` and `@johnnynunez`.

---

## 12. Testing before submitting anything — commands

Don't submit either PR on the strength of reading the Dockerfiles alone — test both builds on real Orin hardware first.

### 12.1 Install the `jetson-containers` CLI

```bash
git clone https://github.com/dusty-nv/jetson-containers ~/jetson-containers
cd ~/jetson-containers
bash install.sh
```

Requires Docker (already present on JetPack 6.2 images) and BuildKit. First run will pull/build a fair number of base layers — expect this to take a while even before touching torch/torchcodec/lerobot specifically.

### 12.2 Test 1 — does the existing torchcodec:0.11.0-pytorch:2.11 recipe already work for cp312?

This checks whether jetson-containers' own recipe (Section 9) already produces a working cp312 build, before assuming a from-scratch rebuild (Sections 2-5) is the only path:

```bash
cd ~/jetson-containers
PYTHON_VERSION=3.12 jetson-containers build torchcodec:0.11.0-pytorch:2.11
```

Run in `tmux` (`tmux new -s jc_torchcodec`) — this is a multi-hour Docker build (torch from source, same as Sections 2-5, just containerized with ccache).

**Outcomes:**
- **Builds clean** → their existing recipe already covers this combination end-to-end. Contribution becomes: ask (issue, not PR) why cp312 wheels aren't published to `pypi.jetson-ai-lab.io` for JetPack 6.2, citing lerobot's `>=3.12` requirement as the motivating real-world case. No new code needed from you.
- **Fails** → you have a precise, reproducible bug report for `dusty-nv/jetson-containers` (open an issue with the exact error), and Sections 2-5 of this guide remain the proven fallback for anyone hitting the same wall.

### 12.3 Test 2 — verify the fixed `lerobot` package Dockerfile actually builds

After making the edits from Section 11 to a local clone/fork of `jetson-containers`:

```bash
cd ~/jetson-containers
PYTHON_VERSION=3.12 jetson-containers build lerobot
```

Confirm:
- The container builds without dependency resolution errors (the `pi0`→`pi` fix + added `torchcodec` dep)
- Inside the built container, `python3 -c "import torchcodec; from torchcodec.decoders import VideoDecoder"` succeeds (proves it's not silently falling back to pyav)
- The updated `lerobot-record`/`lerobot-teleoperate` example commands in `.bash_history` actually match current CLI syntax

Only submit the PR once this passes locally — an untested Dockerfile diff is the fastest way to get a PR ignored by `@dusty-nv`/`@johnnynunez`.

---

## 13. Does the SO-101 test need to wait for the Docker work?

**No — do the SO-101 / SmolVLA repro now, independently, on the bare-metal stack from Sections 2-5.** The two tracks don't depend on each other:

- The bare-metal stack (torch 2.11.0 + torchcodec 0.11.0 + lerobot 0.6.1) is **already built and verified working** on this Jetson right now. Nothing about it needs Docker.
- The `jetson-containers` work (Sections 9-12) is a *separate, parallel* contribution track aimed at fixing the **container path** for other users — it doesn't block, and isn't blocked by, validating SmolVLA correctness on real hardware.
- Running SO-101 through Docker would actually be *more* setup, not less: USB device passthrough for the feetech serial adapter and camera devices (`--device=/dev/ttyUSB0`, `--device=/dev/video0`, or `--privileged`), X11/display forwarding if using any viz tooling, etc. None of that is needed on bare metal, where the hardware is just already there.

**Recommended order:**
1. Do the SO-101 / SmolVLA repro now, on the current bare-metal install (Sections 2-5 stack). This is the one that actually answers the open technical question from #3636 (Section 7) and is the strongest content for a #819 reply.
2. Run the `jetson-containers` tests (Section 12) whenever convenient, in parallel or afterward — they're independent, lower-urgency, and mostly waiting on multi-hour Docker builds rather than your attention.
