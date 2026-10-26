# `jetson-containers build` fails on any package with `buildkit_device`: `ERROR: unknown entitlement device`

**2026-08-11 note**: this whole investigation is scoped to `jetson-containers`' own buildx/entitlement wrapper — it doesn't apply to [`lerobot-jetson`](https://github.com/ravediamond/lerobot-jetson), a plain `docker build` against a single Dockerfile with no buildx-device path at all. One real fix from deep in this doc did carry over though: **Bug 10** (`USE_NVSHMEM=0` — PyTorch's CMake auto-enables a multi-node feature that fails to link on a single-GPU board) is now in `lerobot-jetson`'s Dockerfile directly, since it's a PyTorch/CUDA-on-Jetson issue, not a jetson-containers-specific one.

## Environment

- Jetson Orin NX 16GB, JetPack 6.2 (L4T R36.4.3), CUDA 12.6, aarch64
- Docker 27.5.0, buildx v0.19.3, buildkitd v0.31.2 (via `moby/buildkit:buildx-stable-1`)
- `jetson-containers` at commit `70c149ae` (main, 2026-07-20)

## Problem

Running any package build that declares `buildkit_device` in its metadata header (e.g. `packages/build/build-essential/Dockerfile`, `packages/physicalAI/lerobot/Dockerfile`) fails immediately, before any actual build work happens:

```bash
PYTHON_VERSION=3.12 jetson-containers build lerobot
```

```
DOCKER_BUILDKIT=1 ... docker buildx build --network=host --shm-size=8g \
  --progress=tty \
  --load \
  --allow device \
  --tag lerobot:r36.4.tegra-aarch64-cp312-cu126-22.04-build-essential \
  ...

ERROR: unknown entitlement device
💣 `jetson-containers build` failed after 2.4 seconds (0.0 minutes)
```

This happens on the **first** package in the dependency chain (`build-essential`), so it blocks essentially every build — not something specific to the `lerobot` package.

## Root cause

`docs/build.md` states:

> When a BuildKit device is requested, the builder uses `docker buildx build --allow device`, writes a temporary Dockerfile with `# syntax=docker/dockerfile:1-labs`, and adds `RUN --device=nvidia.com/gpu=all` to build steps. This requires BuildKit 0.20.0 or newer, a builder/daemon started with the `device` entitlement enabled, and NVIDIA devices registered through CDI.

Two of those three prerequisites are **not actually set up by a fresh `jetson-containers` / `bash install.sh` install**, and nothing in the docs or install script configures them:

1. The default buildx builder uses the `docker` driver, which does not support the `device` entitlement at all — you must create a `docker-container` driver builder.
2. Even after creating a `docker-container` builder, its buildkitd daemon must be explicitly started with `--allow-insecure-entitlement device` (and, since the `docker-container` driver's isolated buildkitd doesn't inherit the host's implicit `network.host` allowance the way the `docker` driver does, also `--allow-insecure-entitlement network.host`).
3. The buildx **client** additionally needs `BUILDX_EXPERIMENTAL=1` set in the environment, or it rejects `--allow device` before ever contacting the daemon.

CDI registration (`/etc/cdi/nvidia.yaml`) was already present on this system via the standard JetPack NVIDIA Container Toolkit setup, so that part "just worked" — but the buildx/buildkitd side did not.

## Repro steps

```bash
git clone https://github.com/dusty-nv/jetson-containers ~/jetson-containers
cd ~/jetson-containers
bash install.sh
PYTHON_VERSION=3.12 jetson-containers build build-essential
# -> ERROR: unknown entitlement device
```

## What fixed it (workaround, not a proposed code change)

Option A — bypass the device entitlement path entirely for packages that don't strictly need GPU access during the build:

```bash
PYTHON_VERSION=3.12 jetson-containers build --buildkit-device=off lerobot
```

Note: `--no-buildkit` alone does **not** work — `container.py`'s `use_buildx` check ORs in `bool(active_buildkit_device)`, so a package's own `buildkit_device` metadata forces the buildx+entitlement path regardless of `--no-buildkit`. Only `--buildkit-device=off` (which normalizes to an empty string and overrides the per-package value) actually disables it.

Option B — properly enable the entitlement, for packages/builds that do need GPU access mid-build:

```bash
docker buildx create --name jetson-builder --driver docker-container \
  --buildkitd-flags "--allow-insecure-entitlement device --allow-insecure-entitlement network.host" \
  --use

BUILDX_EXPERIMENTAL=1 PYTHON_VERSION=3.12 jetson-containers build lerobot
```

## Suggested fix for `jetson-containers`

- `install.sh` (or a documented one-time setup step) should create and configure a `docker-container` builder with the required `--allow-insecure-entitlement` flags automatically, OR
- `docs/build.md` should spell out the exact `docker buildx create ...` command and `BUILDX_EXPERIMENTAL=1` requirement instead of just naming the BuildKit version floor, OR
- `container.py` could detect an unconfigured/incompatible builder up front and print a clear remediation message instead of surfacing the raw `unknown entitlement device` buildkit error.

## Context

Found while testing a fix for `packages/physicalAI/lerobot/Dockerfile` (missing `torchcodec` dependency, stale `pi0`→`pi` extra, dead example CLI commands in `.bash_history`) — see companion PR. This entitlement issue is orthogonal to that fix and blocks testing it (and any other package) from a clean install.

---

## Update (2026-07-29): three more bugs found while actually running the Section-12 build test

Set up Option B (`docker-container` builder + `--allow-insecure-entitlement device network.host` + `BUILDX_EXPERIMENTAL=1`) on the same Orin. That surfaced three further, distinct bugs — documented here since each blocks the next step in turn.

### Bug 2 — `docker buildx` client too old to support `device` entitlement at all

Even with the builder correctly configured (`docker buildx inspect` showed the right `--allow-insecure-entitlement` flags, `BUILDX_EXPERIMENTAL=1` exported, builder selected via `docker buildx use`), every build still failed immediately with the exact same `ERROR: failed to build: unknown entitlement device` from the original report.

Root cause: the **buildx client binary**, not the daemon. This system had `docker-buildx-plugin 0.19.3` installed via apt. Client-side support for the `device`/CDI entitlement (`--allow device`) was only added in **buildx v0.21.0** (`docker/buildx#2994`, released as part of `v0.21.0`). `buildkitd` v0.31.2 (daemon side) supports the entitlement fine — it's the client that can't request it correctly on anything before v0.21.0, and the daemon's error message is identical to the "builder isn't configured" case from Bug 1, which makes the two easy to conflate.

**Fix**: `sudo apt-get install --only-upgrade docker-buildx-plugin` (candidate on `download.docker.com/linux/ubuntu jammy/stable` was 0.35.0, far ahead of the installed 0.19.3). Confirmed `docker buildx version` → `v0.35.0`, and a minimal `--allow device` build got past the entitlement check.

**Suggested fix for `jetson-containers`**: `install.sh` or `docs/build.md` should state a minimum `docker-buildx-plugin`/`buildx` client version (≥0.21.0, ideally recommend current) alongside the existing BuildKit daemon version floor — the docs currently only mention the daemon-side 0.20.0 requirement, not the client.

### Bug 3 — auto-generated container tag is invalid when a package name has two colons

Compound package selectors like `torchcodec:0.11.0-pytorch:2.11` (used to pin torchcodec against a specific PyTorch build) contain two colons. `jetson_containers/container.py`'s `build_container()` sets `name = packages[-1]` for the default container name and only ever sanitizes `+` (`name = name.replace('+', '-')`, ~line 305) — it never collapses extra colons in `name` itself. Per-stage suffixes are sanitized (`package.replace(':','_')`, ~line 328), but the base `name` isn't, so the final `--tag` argument passed to `docker buildx build` ends up as e.g. `torchcodec:0.11.0-pytorch:2.11-r36.4.tegra-aarch64-cp312-cu126-22.04-build-essential` — a Docker reference with two colons, which is invalid (`name:tag` allows exactly one).

```
ERROR: failed to build: invalid tag "torchcodec:0.11.0-pytorch:2.11-r36.4...": invalid reference format
```

This was masked under the old buildx client (Bug 2): it always died on `unknown entitlement device` before ever reaching Docker's own reference-format validation. Upgrading buildx to 0.35.0 fixed Bug 2 and immediately exposed this pre-existing, independent tag bug.

**Workaround**: pass an explicit `--name` that has no colon, e.g. `jetson-containers build --name torchcodec-test torchcodec:0.11.0-pytorch:2.11`.

**Suggested fix for `jetson-containers`**: sanitize `name` for extra colons (not just `+`) in `build_container()` before it's used to construct the `--tag` argument, the same way the per-stage suffix already is.

### Bug 4 — `docker-container` driver can't see locally-built images between build stages

With Bugs 2 and 3 worked around, the first stage (`build-essential`) built successfully end-to-end using `--device=nvidia.com/gpu=all` (confirmed real device-entitlement usage, not just a no-op flag). The **second** stage then failed:

```
ERROR [internal] load metadata for docker.io/library/torchcodec-test:...-build-essential:
ERROR: failed to build: failed to solve: pull access denied, repository does not exist or may require authorization: server message: insufficient_scope: authorization failed
```

`jetson_containers` chains packages by running one `docker buildx build --load --tag <stage-N-tag> ...` per package, where stage N+1's Dockerfile does `FROM <stage-N-tag>` (a bare local tag, no registry prefix). This works with the default `docker` driver, because it shares the host Docker daemon's own image store directly. It does **not** work with the `docker-container` driver (required for the `device` entitlement in Bug 2/the original issue): that driver runs `buildkitd` in a separate container with no access to the host daemon's images at all — confirmed via `docker inspect buildx_buildkit_jetson-builder0 --format '{{json .Mounts}}'`, which shows only the builder's own internal state volume, no `docker.sock` or containerd socket mounted in. BuildKit therefore treats `FROM torchcodec-test:...` as an unqualified remote reference and tries (and fails) to pull `docker.io/library/torchcodec-test:...`.

Tried and ruled out: enabling the containerd image-store integration (`"features": {"containerd-snapshotter": true}` in `/etc/docker/daemon.json`, then `systemctl restart docker`) on the theory that it would let `docker-container` builds resolve local images without a registry round-trip. Confirmed via `docker info` that `driver-type: io.containerd.snapshotter.v1` was active, then re-tested a minimal two-stage local chain (`FROM chaintest:a` after a separate `--load` build of `chaintest:a`) — still failed with the identical `pull access denied`. Root cause is architectural, not a daemon config gap: the `docker-container` driver's `buildkitd` is a fully isolated daemon regardless of the host's snapshotter setting; the container has no socket into the host at all. Reverted the daemon.json change afterward since it didn't help and added unnecessary system-state drift.

This is a genuine structural conflict: **device entitlement requires the `docker-container` driver; jetson-containers' multi-stage local-tag chaining requires the `docker` driver's shared image store.** They cannot both be satisfied at once as the tool is currently built.

**Options, not yet implemented:**
- Route each stage through a local registry (`docker run -d -p 5000:5000 registry:2`, tag/push/pull `localhost:5000/...` between stages instead of bare local tags) — the standard workaround for `docker-container`/`kubernetes` drivers referencing prior local build output. Would need `jetson_containers/container.py` changed to push after each stage and reference the registry-qualified name as `BASE_IMAGE` for the next.
- Only use the `docker-container`+device-entitlement path for the specific stage(s) that actually need GPU access during build (e.g. the `pytorch`/`torchcodec` compile step), and build every earlier stage with the plain `docker` driver — requires `container.py` to switch drivers mid-chain rather than forcing buildx for the whole dependency tree once any package declares `buildkit_device` (see original Root Cause section above for the `use_buildx` OR-in behavior).
- Pragmatic unblock used to actually get a build signal today: `--buildkit-device=off`, which forces the `docker` driver for the whole chain and skips device entitlement entirely (see Section "What fixed it, Option A" above). Untested whether any step in the `torchcodec:0.11.0-pytorch:2.11` chain actually *requires* a live GPU at build time (vs. just CUDA toolkit/headers, which don't need device passthrough) — if none do, this workaround is sufficient for this package; if one does (e.g. an arch auto-detection or in-build smoke test), it may silently produce a wrong or untested build.

### Bug 5 — `python` package install fails: custom pip index doesn't carry `pip` itself

With Bug 4 sidestepped via `--buildkit-device=off` (forces plain `docker` driver, restores local-tag chaining), the build actually progressed: stages 1–4 (`build-essential`, `pip_cache:cu126`, `cuda`, `video-codec-sdk`) built and tested clean in ~7.5 minutes, including a stage that genuinely exercises `--device=nvidia.com/gpu=all` when device entitlement is on (confirmed in the Bug 4 partial run before switching to `--buildkit-device=off`). Stage 5 (`python`) then failed:

```
+ uv pip install --upgrade --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 pip pkginfo
× No solution found when resolving dependencies:
╰─▶ Because pip was not found in the package registry and you require pip, we can conclude that your requirements are unsatisfiable.
```

Root cause: `packages/cuda/cuda/Dockerfile.pip` sets `ENV PIP_INDEX_URL=${PIP_INDEX_REPO}` (~line 34), where `PIP_INDEX_REPO` is the jetson-ai-lab custom mirror (`https://pypi.jetson-ai-lab.io/jp6/cu126`). Docker `ENV` set in one stage of an image lineage persists into every downstream layer built `FROM` it — it's baked into image config, not scoped to that one build-arg invocation. `packages/build/python/install.sh` defaults `PIP_INDEX_URL` to `https://pypi.org/simple` (line 11) **only if unset**, but by the time this stage runs it's already inherited the custom mirror URL from the `cuda` stage above it. Line 66 then runs `uv pip install --index-url "${PIP_INDEX_URL}" pip pkginfo` — `--index-url` is *exclusive* (unlike `--extra-index-url`), and the jetson-ai-lab mirror only republishes ML/CUDA wheels (torch, torchcodec, etc.), not base tooling like `pip`/`pkginfo` itself. So the resolver correctly reports it can't find `pip` on the only index it's allowed to look at.

**Suggested fix for `jetson-containers`**: `python/install.sh`'s bootstrap `pip`/`pkginfo`/`twine` installs (lines 66, 82) should use `--extra-index-url` alongside the real PyPI, or explicitly pull from `https://pypi.org/simple` regardless of any inherited `PIP_INDEX_URL`, since those two packages are never going to be on a CUDA-wheel mirror.

**Not yet tried**: overriding `PIP_INDEX_URL` explicitly on the `jetson-containers build` invocation to see if it forces the right value through, or patching `install.sh` locally to add `--extra-index-url https://pypi.org/simple` and re-running from this stage.

### Bug 6 — the correct pip fallback (`UV_EXTRA_INDEX_URL`) was already half-built, just never wired up, and its target is also dead

Patched around Bug 5 first by adding `--extra-index-url https://pypi.org/simple` directly to the two affected `install.sh` lines. That unblocked `pip`/`pkginfo`, but the very next line (`uv pip install --no-binary :all: psutil`, no explicit `--index-url` at all) hit the identical failure — it silently inherits `UV_DEFAULT_INDEX` (the exclusive jetson-ai-lab mirror) with nothing to fall back to. Patching every affected line individually in `install.sh` would be endless whack-a-mole against every non-CUDA package in the chain.

Found the real, systemic fix already half-implemented: `packages/cuda/cuda/config.py:198` constructs a `UV_EXTRA_INDEX_URL` build-arg (`https://pypi.{index_host}/root/pypi/+simple` — a full PyPI mirror proxy, not just CUDA wheels) and passes it into every `docker buildx build` invocation (visible in the `--build-arg UV_EXTRA_INDEX_URL=...` line of every build command in the logs). But `packages/cuda/cuda/Dockerfile.pip` never declares it as `ARG` or writes it to `ENV` — it's computed, passed on the command line, and then silently dropped. `uv` reads `UV_EXTRA_INDEX_URL` from the environment automatically on every invocation with no per-command flag needed, so wiring this one existing build-arg through fixes every downstream package's pip install in one place, not just `python`'s.

**Fix applied**: reverted the per-line `install.sh` patch, then in `Dockerfile.pip` added `UV_EXTRA_INDEX_URL` to the `ARG INDEX_HOST \ ... FALLBACK_TAR_INDEX_URL` list and to the `ENV` block (`UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL} \`, alongside the existing `PIP_INDEX_URL`/`UV_DEFAULT_INDEX` lines).

**Result**: confirmed the wiring itself works — `uv` now actually attempts `https://pypi.jetson-ai-lab.io/root/pypi/+simple/pkginfo/` as fallback. But that URL returns `410 Gone`. The jetson-ai-lab "full PyPI mirror proxy" this was designed to point at has been decommissioned — infra-side, not fixable by any Dockerfile change.

**Fix applied (final)**: changed `config.py:198`'s `UV_EXTRA_INDEX_URL` value from `https://pypi.{index_host}/root/pypi/+simple` to `https://pypi.org/simple` directly. This got past `python`, `cmake`, `llvm:22` cleanly.

**Suggested fix for `jetson-containers`**: two independent fixes needed upstream — (a) wire `UV_EXTRA_INDEX_URL` through `Dockerfile.pip`'s `ARG`/`ENV` like every other pip-related variable there (it's currently dead code, computed and passed but never consumed), and (b) point it at `https://pypi.org/simple` (or otherwise confirm/fix whatever `pypi.{index_host}/root/pypi/+simple` was supposed to be, since it currently 410s).

### Bug 7 — `ffmpeg` install.sh silently reports success when the build actually failed

With Bugs 2–6 resolved, the chain progressed 7 stages clean (`build-essential` → `ffmpeg:8.1`) in ~13 minutes, including the `python`/`cmake`/`llvm:22` stages Bug 6 had blocked. But stage 8's **test** step then failed:

```
+ ffmpeg -version
/test/test.sh: line 4: ffmpeg: command not found
```

— even though the **build** step for that same stage had reported success and exported the image layer normally. Root cause is in `packages/multimedia/ffmpeg/install.sh`, the `git`-install branch:

```bash
tarpack install "ffmpeg-$FFMPEG_VERSION" || \
$TMP/build.sh || \
echo "FAILED to build FFMPEG $FFMPEG_VERSION"
ldconfig
```

This is a three-way `||` chain ending in a bare `echo`. If both `tarpack install` (fetch a prebuilt tarball) and `build.sh` (compile from source) fail, the `echo` still runs and exits 0 — there is no `exit 1` or equivalent after it. So the *shell script* reports success, the *Docker `RUN`* reports success, the *layer* gets built and tagged, and the missing binary is only ever discovered later, at the separate test stage — or not at all, if `--skip-tests` is used, or if a downstream package doesn't specifically test for the binary's presence. Confirmed via the build log (`~/jetson-containers/logs/.../build/*ffmpeg*.txt`): `tarpack` failed (no prebuilt `ffmpeg-8.1.tar.gz` exists yet at `apt.jetson-ai-lab.io/jp6/cu126/` for this combination — expected, cp312/JP6.2 is a less-common pairing), `build.sh` then failed on a `./configure` error (see Bug 8), and the log shows the literal text `FAILED to build FFMPEG 8.1` followed immediately by the *next* build step starting, with no non-zero exit anywhere in between.

**Suggested fix for `jetson-containers`**: replace the bare `echo` with something that actually fails the build, e.g. `{ echo "FAILED to build FFMPEG $FFMPEG_VERSION"; exit 1; }`. This pattern is worth grepping for across other `packages/*/install.sh` files — a `some_install_attempt || fallback || echo "failed"` chain with no trailing `exit` silently converts any double-failure into a "successful" image with a missing component, and is exactly the kind of bug that only surfaces much later (a user hits `ffmpeg: command not found` at runtime, with no link back to a build-time failure that was actually logged and then silently discarded).

### Bug 8 — ffmpeg 8.1's `./configure` requires dav1d ≥1.0.0; Ubuntu 22.04's apt package is 0.9.2

The actual `build.sh` failure Bug 7 was masking:

```
ERROR: dav1d >= 1.0.0 not found using pkg-config
```

`build.sh` passes `--enable-libdav1d` to ffmpeg's `./configure` (line 118) and `install.sh`'s `git`-branch apt-installs `libdav1d-dev` to satisfy it (line ~40). But on this system, `apt-cache policy libdav1d-dev` shows only `0.9.2-1` available from `jammy/universe` — Ubuntu 22.04's repos never shipped dav1d 1.0+. ffmpeg 8.1 apparently raised its minimum dav1d requirement past what jammy's apt can provide.

**Not yet fixed.** Options, not yet attempted: (a) build dav1d from source at a newer version before the ffmpeg configure step (adds real build time), (b) drop `--enable-libdav1d` if AV1 decode via this specific codec path isn't essential (torchcodec/lerobot's actual codec needs are H.264/H.265 for typical robot camera recordings, not AV1 — worth checking whether this is even needed for the target use case), or (c) pull a newer libdav1d via a non-jammy source (e.g. a PPA or manual `.deb`). Whichever fix lands, Bug 7 should be fixed *first* — otherwise a fix here still won't be visible as a failure if it's wrong, and future ffmpeg version bumps will silently break the same way again.

### Bug 9 — ffmpeg 8.1's `nvenc.c` incompatible with the Video Codec SDK headers this package pulls

With Bugs 7 and 8 fixed (silent-failure now surfaces real errors; dav1d dropped), the ffmpeg build got substantially further — full `configure` + most of `libavformat`/`libavcodec` compiled — before failing on:

```
libavcodec/nvenc.c: In function 'nvenc_fill_time_code':
error: 'NV_ENC_CLOCK_TIMESTAMP_SET' has no member named 'countingType'; did you mean 'countingTypeLSB'?
```

Root cause confirmed via the installed header: `/usr/local/include/ffnvcodec/nvEncodeAPI.h` declares `NVENCAPI_MAJOR_VERSION 13` / `NVENCAPI_MINOR_VERSION 1`, and its `NV_ENC_CLOCK_TIMESTAMP_SET` struct has already split the single `countingType` field into `countingTypeLSB : 1` + `countingTypeMSB : 4` bitfields. `packages/multimedia/ffmpeg`'s `nvenc.c` (vendored at ffmpeg 8.1) was written against the pre-split single-field API and doesn't know about the new bitfield split — a genuine version-skew incompatibility between ffmpeg 8.1's NVENC code and the `video-codec-sdk` package's headers (pulled via `NV_CODEC_HEADERS_TAG="n13.0.19.0"`, whose internal API version is 13.1 despite the `13.0.19` tag name).

**Fix applied**: dropped `--enable-nvenc` from `build.sh`'s `./configure` flags, keeping `--enable-nvdec`/`--enable-cuvid`/`--enable-cuda`. torchcodec/lerobot's actual need is decoding recorded episode videos (NVDEC), not hardware encode — same reasoning as dropping `--enable-libdav1d` in Bug 8. NVENC support can be revisited separately (either patching `nvenc.c` for the new struct layout, or pinning `video-codec-sdk` to older, API-13.0-compatible headers) if hardware encode is ever needed for this project.

**Gotcha**: removing `--enable-nvenc` alone did **not** disable it — ffmpeg's `./configure` auto-enables NVENC by default whenever it detects the ffnvcodec headers present (`--enable-nvenc` is not opt-in, only `--disable-nvenc` actually turns it off). Confirmed by the identical compile error recurring on the next run even with the enable flag gone. Fixed by adding an explicit `--disable-nvenc \` after `--enable-cuvid` in `build.sh`.

**Suggested fix for `jetson-containers`**: either patch `nvenc.c` for the header's new bitfield layout, or pin `video-codec-sdk`'s `NV_CODEC_HEADERS_TAG` to a version whose `nvEncodeAPI.h` still matches what the currently-declared `ffmpeg` version's vendored `nvenc.c` expects — the two packages' versions need to move in lockstep, and right now nothing enforces that.

### Bug 10 — pytorch build fails linking `torch_nvshmem`, an irrelevant multi-node feature auto-enabled by CMake

With Bugs 2–9 fixed, the chain progressed through stage 14/15 (`pytorch:2.11`) — a ~80-minute compile (2769→2942 of 6458 objects) — before failing:

```
nvlink error   : Uncompress failed
nvlink fatal   : elfLink fatbinary error
ninja: build stopped: subcommand failed.
```

on `CMakeFiles/torch_nvshmem.dir/cmake_device_link.o`. `torch_nvshmem` links PyTorch's support for NVSHMEM (NVIDIA's multi-GPU/multi-**node** shared-memory library, built for clusters communicating over NVLink/InfiniBand) — entirely irrelevant to a single-GPU Jetson board. `packages/ml/pytorch/build.sh` sets `USE_DISTRIBUTED=1` and `USE_NCCL=1` (both reasonable — needed even for single-GPU DDP-style code paths) but never sets `USE_NVSHMEM`, so PyTorch's CMake auto-detects the `nvidia-nvshmem` pip package (visible in the failing command: `/opt/venv/.../nvidia/nvshmem/lib/libnvshmem_device.a`) and silently turns the feature on.

**Fix applied**: added `export USE_NVSHMEM=0` to `build.sh` alongside the existing `USE_FLASH_ATTENTION=0`/`USE_MEM_EFF_ATTENTION=0` lines. Same category of fix as Bugs 8/9 — drop an unneeded feature rather than debug the underlying toolchain issue (root cause of the `nvlink`/`Uncompress failed` error itself not investigated; plausibly an `nvcc`/`nvlink` compatibility gap on aarch64 with this specific nvshmem device library version, but moot once the feature's disabled).

**Suggested fix for `jetson-containers`**: explicitly set `USE_NVSHMEM=0` in `packages/ml/pytorch/build.sh` for Jetson targets — multi-node shared memory has no use case on a single-board device, and letting it auto-enable via pip-dependency detection makes every from-scratch pytorch build on Jetson fail at the very end of a multi-hour compile.

### Bug 11 — pytorch's pip fast-path pollutes the environment with CUDA-13 packages even though it ultimately fails and falls back to source build

Major milestone first: with Bugs 2–10 fixed, both `pytorch:2.11` (stage 14, ~138 minutes, survived one mid-build reboot via `ccache`) and `torchcodec:0.11.0-pytorch:2.11` (stage 15) **built successfully**. This confirms the core question Section 12.2 set out to answer — cp312 / JetPack 6.2 / CUDA 12.6 / torch 2.11 / torchcodec 0.11.0 **is buildable** via `jetson-containers` once the above bugs are worked around. Only the post-build *test* step failed:

```
OSError: Could not load this library: .../torchcodec/libtorchcodec_core4.so
OSError: libnvrtc.so.13: cannot open shared object file: No such file or directory
```

This system has CUDA 12.6 (JetPack 6.2); `libnvrtc.so.13` is CUDA 13's runtime-compiler library, so something in the image is CUDA-13-tagged. Confirmed via `pip list` inside the built image: the *entire* `nvidia-*` package set is CUDA 13 — `nvidia-cuda-nvrtc 13.0.88`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`, `cuda-toolkit 13.0.2`, etc. None of these should be present at all — this build compiles PyTorch from source against the system's own CUDA 12.6 toolkit and shouldn't need any pip-distributed CUDA runtime packages.

Root cause: `packages/ml/pytorch/install.sh` runs *before* `build.sh` (the Dockerfile's `RUN` is `install.sh || build.sh`) and attempts a pip-based fast path first — `uv pip install torch==${TORCH_VERSION} || uv pip install --prerelease=allow "torch>=${PYTORCH_BUILD_VERSION}.dev,<=${PYTORCH_BUILD_VERSION}"`. Before Bug 6 was fixed, both of these always failed outright (dead/exclusive index), so `install.sh` predictably exited non-zero and `build.sh` always ran. Now that `UV_EXTRA_INDEX_URL` correctly falls back to real PyPI, the second `uv pip install` can actually **resolve and install** a real, generic (non-Jetson, CUDA-13-tagged) `torch==2.11.0` wheel from PyPI — pulling in the full `nvidia-*-cu13` dependency stack as a side effect. `install.sh` then fails its own later sanity check (`python3 -c 'import torch; ... torch.cuda.device_count()'`, since that generic wheel presumably can't actually use this GPU/CUDA version correctly) and *does* correctly fall through to `build.sh` — which clones, compiles, and installs the correct source-built `torch` wheel — but pip/`uv` never rolled back or removed the already-installed stray `nvidia-*-cu13` packages from the failed first attempt, since `uv pip install /opt/torch*.whl` (installing a specific local wheel file) doesn't audit or clean up unrelated already-present packages. Those stray packages sat in `site-packages` for the rest of the build, and `torchcodec`'s own build (stage 15) appears to have picked one up when resolving its CUDA link target, baking in a reference to `libnvrtc.so.13` that the actual (correct, CUDA-12.6) runtime environment can't satisfy.

**Fix applied**: the package already ships a purpose-built escape hatch for exactly this — `config.py:34` sets the `FORCE_BUILD` Dockerfile build-arg to `'on'` if the host env var `PYTORCH_FORCE_BUILD=on` is set at `jetson-containers build` invocation time, and `install.sh`'s very first check (`if [ "$FORCE_BUILD" == "on" ]; then exit 1; fi`) makes it exit immediately, skipping the risky pip attempt entirely and always going straight to the known-good source build. Reran with `PYTORCH_FORCE_BUILD=on jetson-containers build ...`. This invalidates the cached pytorch layer (build-arg change) but `ccache` should still replay the bulk of the ~138-minute compile quickly, then rebuild `torchcodec` on top of a clean, correctly-CUDA-12.6-only environment.

**Suggested fix for `jetson-containers`**: either (a) have `install.sh`'s pip fast path pin `--index-url` to a Jetson-specific wheel source only (never fall through to generic PyPI for this specific install, unlike the general `UV_EXTRA_INDEX_URL` fallback which is otherwise correct and needed — see Bug 6), or (b) have it `uv pip uninstall` everything it just installed before falling through to `build.sh` on failure, so a failed fast-path attempt never leaves stray packages behind for later build stages to accidentally pick up.

### Bug 12 — same pip-fast-path problem recurs in `torchcodec`'s own `install.sh`, and the package config already anticipated it with an unused `-builder` variant

After Bug 11's `PYTORCH_FORCE_BUILD=on` fix, the pytorch stage rebuilt cleanly — `pip list` inside the image now shows only `numpy`/`onnx`/`optree`/`torch` (no stray `nvidia-*-cu13` packages), and the test output confirms a fully working build: `torch==2.11.0`, CUDA 12.6, cuDNN 9.3, device `Orin`, every sub-test (`LAPACK`, `cuDNN`, tensor ops) passing, ending in `PyTorch OK`.

But `torchcodec`'s stage still failed the identical `libnvrtc.so.13` error as before — and the giveaway was timing: the `RUN` step for this stage completed in **2.3 seconds**, nowhere near enough to actually clone and compile a C++/CUDA extension from source. `packages/ml/pytorch/torchcodec/install.sh` has the exact same pip-fast-path pattern as pytorch's (`uv pip install torchcodec~=${TORCHCODEC_VERSION} || uv pip install --prerelease=allow ...`) — with Bug 6's PyPI fallback now working, it silently resolved and installed a real, generic (non-Jetson, CUDA-13-linked) `torchcodec` wheel from PyPI instead of building from source, exactly like Bug 11's pytorch case.

Checking `torchcodec`'s own `config.py` (`packages/ml/pytorch/torchcodec/config.py:6-38`) shows the package authors already anticipated this exact problem: the `torchcodec()` config function returns **two** package variants — the default (pip-first, what we'd been building) and a second `-builder`-suffixed twin with `builder['build_args']['FORCE_BUILD'] = 'on'` hardcoded. We had been building the wrong one (the default) this whole time without realizing a source-forcing variant already existed.

**Fix applied**: rather than hunt for the exact `-builder` package alias name, forced it directly via the `jetson-containers build` CLI's `--build-args` flag (format: comma-separated `KEY:VALUE` pairs, parsed at `jetson_containers/build.py:106-114`): `--build-args "FORCE_BUILD:on"`. This applies to every stage in the chain — the already-cached, already-correct pytorch layer is unaffected (same build-arg value as the `PYTORCH_FORCE_BUILD=on` env-var-driven rebuild), while torchcodec's `install.sh` now hits its own `FORCE_BUILD` check and skips straight to a real source build.

**Suggested fix for `jetson-containers`**: since a `-builder` variant already exists specifically to avoid this, either make the default `torchcodec` package alias point at the builder variant on platforms where no real matching prebuilt wheel is expected to exist (e.g. Jetson/aarch64), or apply the same fix as Bug 11's suggestion directly in `install.sh` (constrain the pip fast path to a Jetson-specific index only, or clean up after a failed attempt) rather than relying on callers to know to pass `FORCE_BUILD:on` themselves.

### Bug 13 — torchcodec's `build.sh` apt-installs a `python3.12-dev` package that doesn't exist on jammy, when headers are already present

With Bug 12's `FORCE_BUILD:on` applied, `install.sh` now correctly short-circuits ("Forcing build of torchcodec 0.11.0") and hands off to `build.sh` — confirming the fix works. New failure, immediately after:

```
apt-get install -y --no-install-recommends git pkg-config libffi-dev libsndfile1 python3.12-dev
E: Unable to locate package python3.12-dev
```

Ubuntu 22.04 (jammy)'s apt repos only carry `python3.10-dev`/`python3.11-dev` (confirmed via `apt-cache search`) — never `3.12`. But this system's Python 3.12 isn't from apt at all; it's a standalone build installed by `uv` (`packages/build/python/install.sh`, `UV_PYTHON_INSTALL_DIR=/opt/python`) — and those uv-managed builds bundle their own headers. Confirmed via `docker run --rm <pytorch-stage image> find /opt/python -iname Python.h`: `/opt/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12/Python.h` exists already. The apt install was never necessary on this platform, and CMake/setuptools locate the headers directly through the active interpreter's own `sysconfig`, not through a `python3-config` binary (confirmed absent: `python3-config: command not found` inside the image) or an apt dev package.

**Fix applied**: removed `python${PYTHON_VERSION}-dev` from `build.sh`'s `apt-get install` line entirely.

**Suggested fix for `jetson-containers`**: same fix upstream — drop the `python${PYTHON_VERSION}-dev` apt dependency from `packages/ml/pytorch/torchcodec/build.sh`, since the `build-essential`/`python` package chain that precedes it already guarantees working headers via the uv-managed interpreter on any platform (Jetson or otherwise) using that same base. If some other platform genuinely needs the apt package, make it conditional/best-effort (`|| true`) rather than a hard requirement.

**Addendum — removing the apt package wasn't sufficient on its own.** With `python3.12-dev` gone, `cmake`'s *first* `find_package(Python3)` (invoked from PyTorch's own bundled `TorchConfig.cmake`) succeeded fine (`Found Python: /opt/venv/bin/python3.12 ... Development.Module Development.Embed`), but torchcodec's *own* `CMakeLists.txt` makes a second, stricter call — `find_package(Python3 ... EXACT "3.12" COMPONENTS ... Development ...)` — which failed: `Could NOT find Python3 (missing: Python3_INCLUDE_DIRS Python3_LIBRARIES Development Development.Module Development.Embed)`, even though `/opt/python/cpython-3.12.13-linux-aarch64-gnu/lib/libpython3.12.so` genuinely exists. `python3.12-dev` would have placed a matching header dir and lib symlink under CMake's default system search paths (`/usr/include`, `/usr/lib/aarch64-linux-gnu`); the uv-managed build's files sit under the non-standard `/opt/python/...` prefix, which CMake's `FindPython3` module doesn't search by default even when the same interpreter is correctly resolved via `PATH`.

**Fix applied**: added two symlinks to `build.sh`, right after the `apt-get clean` — pointing `/usr/include/python3.12` and `/usr/lib/aarch64-linux-gnu/libpython3.12.so*` at the real uv-managed files (resolved dynamically via `PY_ROOT=$(dirname $(dirname $(realpath $(which python3))))`, so it isn't hardcoded to this exact python patch version). This recreates exactly what the apt package would have provided, without needing the (unavailable) package itself.

**Suggested fix for `jetson-containers`** (updated): rather than just dropping the apt dependency, the `python` base package (`packages/build/python/install.sh`) should do this symlinking once, globally, for any uv-managed standalone install — every downstream package building a Python C-extension via CMake's `FindPython3` module hits the same non-standard-prefix problem, not just `torchcodec`.

## Result (2026-07-31): full build green

```
testing torchcodec...
torchcodec version: 0.11.0
torchcodec OK

✅ `jetson-containers build torchcodec-test` (torchcodec-test:r36.4.tegra-aarch64-cp312-cu126-22.04)
⏱️  Total build time: 251.9 seconds (4.2 minutes)
```

All 15/15 stages built and passed their tests, including the pytorch (~74min once `ccache` warm; ~138min cold) and torchcodec (~4min) stages that were the actual point of Section 12.2. **Confirmed: `torchcodec:0.11.0-pytorch:2.11` (cp312, JetPack 6.2, CUDA 12.6) builds successfully via `jetson-containers`** — the original open question from `JETSON_ORIN_BUILD_GUIDE.md` §9 is answered: yes, once the 13 bugs documented above are worked around.

**Final invocation that worked, in full:**
```bash
docker buildx use default   # docker driver, not docker-container (Bug 4)
docker apt-get install --only-upgrade -y docker-buildx-plugin   # v0.19.3 -> v0.35.0 (Bug 2, only needed if attempting the entitlement path at all)
export PYTORCH_FORCE_BUILD=on   # Bug 11
PYTHON_VERSION=3.12 jetson-containers build \
  --buildkit-device=off \
  --name torchcodec-test \
  --build-args "FORCE_BUILD:on" \
  torchcodec:0.11.0-pytorch:2.11
```
Plus local source patches (not yet upstreamed) to: `packages/cuda/cuda/Dockerfile.pip` (Bug 6), `packages/cuda/cuda/config.py` (Bug 6), `packages/multimedia/ffmpeg/install.sh` (Bug 7), `packages/multimedia/ffmpeg/build.sh` (Bugs 8, 9), `packages/ml/pytorch/build.sh` (Bug 10), `packages/ml/pytorch/torchcodec/build.sh` (Bugs 12, 13).

## Environment note for this update

- `docker-buildx-plugin`: 0.19.3 → upgraded to 0.35.0 (apt, `download.docker.com/linux/ubuntu jammy/stable`)
- `buildkitd` (inside `docker-container` builder): unchanged, v0.31.2
- Tested on the same Jetson Orin NX 16GB / JetPack 6.2 / CUDA 12.6 system as the original report
- Section 12.2 test (`torchcodec:0.11.0-pytorch:2.11`, `--buildkit-device=off --name torchcodec-test`) reached stage 8/15 (`ffmpeg:8.1`) before hitting Bug 8 (masked by Bug 7) — cp312/JP6.2/torch2.11/torchcodec0.11.0 buildability still unconfirmed pending a dav1d fix
- Local patches made so far, all on top of upstream `jetson-containers` main (commit `70c149ae` base): `packages/cuda/cuda/Dockerfile.pip` (`UV_EXTRA_INDEX_URL` wiring), `packages/cuda/cuda/config.py` (`UV_EXTRA_INDEX_URL` value → `https://pypi.org/simple`). None of these have been upstreamed yet.
