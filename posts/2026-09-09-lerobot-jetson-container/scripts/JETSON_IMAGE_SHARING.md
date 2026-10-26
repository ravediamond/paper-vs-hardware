# Sharing `lerobot-jetson` — context + drafts

Context doc for iterating on Discord/LinkedIn posts announcing the Jetson image.
Written so a fresh session has everything needed without re-deriving history from
`JETSON_ORIN_BUILD_GUIDE.md` / `ROADMAP_CONTRIBUTION_OPPORTUNITIES.md`.

## What this is

[`ravediamond/lerobot-jetson`](https://github.com/ravediamond/lerobot-jetson) — a
Docker image that runs LeRobot on NVIDIA Jetson Orin with real GPU acceleration
(CUDA torch, CUDA torchcodec video decode). Published to GHCR, publicly pullable.

## Why it matters (the actual gap, not marketing)

- **No official or community channel ships a cp312 PyTorch wheel for Jetson.**
  Checked both NVIDIA's own redist and the jetson-ai-lab community index
  (`pypi.jetson-ai-lab.io`) — the community index has torch 2.11.0 but only for
  cp310, and its torchcodec ceiling (0.10.0) is below lerobot's floor
  (`torchcodec>=0.11.0`). This blocks `pip install lerobot` on Jetson entirely.
- **The usual fallback, `jetson-containers`, is effectively unmaintained** — original
  maintainer stepped away mid-2025 (confirmed via that repo's own "no longer
  maintained" issue thread), no human-merged PR there since June 2026, its
  `lerobot` package itself is stale (missing `torchcodec` from `depends`, a
  `pi0`→`pi` extra rename never applied, dead example CLI commands).
- **Without a working torchcodec, lerobot silently falls back to `pyav`** — no crash,
  no warning that matters to most users, just no GPU-accelerated video decode. This
  is also documented in lerobot's own `installation.mdx` platform-support note
  (Linux ARM/aarch64 falls back to `pyav`) — the gap this image closes.

## What was actually built and verified (not just claimed)

Two Dockerfiles in the repo:
- `Dockerfile` — full from-source build (torch, torchcodec, torchvision, ffmpeg
  w/ NVDEC, lerobot itself), ~23.9GB image.
- `Dockerfile.fast` — reuses already-built wheel artifacts for faster iteration,
  ~12.7GB image.

**2026-08-31 full from-scratch rebuild, verified live on real Jetson Orin NX 16GB**
(not emulated, not claimed from memory — checked via `docker run` inside the built
image):
- `torch.cuda.is_available()` → `True`, device `Orin`.
- `torchcodec`'s `VideoDecoder` produces a real `.cuda()` tensor — actual GPU decode,
  not a silent CPU fallback.
- `torchvision`'s compiled ops load correctly. (The generic PyPI wheel is silently
  ABI-incompatible with a from-source torch build here —
  `RuntimeError: operator torchvision::nms does not exist` — despite installing and
  importing cleanly. Built from source too, same as torch/torchcodec.)
- `import lerobot` + a real policy import (`SmolVLAPolicy`) succeed.
- `torch.distributed.is_available()` / `is_gloo_available()` → `True` — real module,
  not a stub. (`USE_DISTRIBUTED=1 USE_GLOO=1` — needed because `lerobot-train`'s
  `is_main_process()` check crashes on the stub otherwise, even on a single-GPU
  board.)
- `torch.backends.cuda.mem_efficient_sdp_enabled()` → `True`.

**Total build time: 5h 33m.** PyTorch compilation alone is **92% of that (~5h 6m)** —
everything else (ffmpeg, torchcodec, torchvision, lerobot install) finishes in under
15 minutes combined. Full per-step timing table in `JETSON_ORIN_BUILD_GUIDE.md`.

**Published to GHCR 2026-09-07**: `ghcr.io/ravediamond/lerobot-jetson:public` and
`:latest` (same image, both tags), confirmed publicly pullable (package visibility
flipped from GitHub's private-by-default). Usage:

```bash
docker pull ghcr.io/ravediamond/lerobot-jetson:latest
docker run --runtime nvidia -it -e HF_TOKEN=<your-hf-token> ghcr.io/ravediamond/lerobot-jetson:latest
```

**Ongoing maintenance commitment**: plan is to keep rebuilding for new lerobot
releases, not a one-off. This is the actual differentiator vs. a random one-time
build — worth stating explicitly in any post, since it's the reason referencing it
anywhere durable (docs, a pinned Discord message) makes sense.

## What's already been posted publicly

- **[huggingface/lerobot#819](https://github.com/huggingface/lerobot/issues/819)**
  ("support for running on Jetson?") — the original issue, maintainer
  `imstevenpmwork` (Steve, a personal contact from lerobot/HF's interview process)
  invited investigation there originally. Posted the full result
  ([comment](https://github.com/huggingface/lerobot/issues/819#issuecomment-5568289926)):
  working image, verified numbers, pull instructions.
- **[#2363](https://github.com/huggingface/lerobot/issues/2363)** ("no gpu used to
  train policy" on Jetson AGX Orin) — same root cause, cross-linked with the same
  fix ([comment](https://github.com/huggingface/lerobot/issues/2363#issuecomment-5568300295)).
- **Docs PR staged but not opened**: one `[!NOTE]` line in `docs/source/installation.mdx`'s
  TorchCodec platform-support callout, on branch `docs/jetson-lerobot-jetson-pointer`
  in this local clone. Deliberately not committed/pushed yet — asked Steve directly
  on #819
  ([comment](https://github.com/huggingface/lerobot/issues/819#issuecomment-5569789343))
  whether referencing it in core docs makes sense, given the "keep updating it"
  commitment, or whether he'd rather keep that out of core docs. **Waiting on his
  answer before opening the PR** — deliberate, not an oversight; he was given the
  option to say no to a core-docs reference.
- Decided explicitly **against opening a new GitHub issue** for "how do I integrate
  this" — #819 already is that issue, and a second one restating the same problem
  would read as padding activity across threads rather than reporting real results.

## What's NOT done / open threads (don't conflate with the image announcement)

- **#3636** (SmolVLA wrong inference on Jetson) — separate, unresolved thread. Ran
  two of `shaoningyu1231`'s three suggested experiments on real Orin hardware;
  results are real but inconclusive (confounded comparisons, not clean evidence
  either way). Asked them for their reference tensor/script for the third experiment
  (no x86+CUDA hardware available to generate one independently). **Not resolved —
  don't imply it is in any public post.**
- **TurboVLA SO-101 training** — separate track (`lerobot-policy-turbovla-so101`),
  in progress via HF Jobs (T4, `ravediamond/pick_cups_all` dataset) as of this
  writing. Not the same repo/image as `lerobot-jetson` — don't merge the two
  narratives in a single post; they're different artifacts on different timelines.

## Framing guidance for both posts

- **Lead with the mechanism, not "look what I built."** The #4381 LinkedIn post
  (VLA-JEPA future-frame-leak fix) worked well precisely because it was framed
  around the bug and how it was caught, not around the fact of contributing. Same
  instinct here: lead with the actual gap (no cp312 wheel exists, jetson-containers
  is dead) and what it took to close it, not "I made a Docker image."
- **State the real numbers, not vibes.** 5h33m build, 92% is torch compile, verified
  `.cuda()` tensors, real bugs found and fixed (torchvision ABI mismatch, distributed
  stub crash) — these are the credible parts. Don't round up or oversell.
- **Don't imply core lerobot endorsement yet.** The docs-PR question to Steve is
  still open. Fine to say "posted on #819" (true, verifiable), not fine to imply
  it's already referenced in official docs.
- **Discord vs LinkedIn are different registers.** Discord: casual, technical,
  peer-to-peer, drop the link and the headline numbers, let people ask questions.
  LinkedIn: slightly more narrative, can afford 2-3 sentences of "why this needed
  solving" context that Discord doesn't need.

---

## Draft: Discord (lerobot server)

> Built a Docker image for running LeRobot on Jetson Orin with real GPU
> acceleration (CUDA torch + CUDA torchcodec) — no official or community channel
> ships a cp312 torch wheel for Jetson yet, so both are built from source.
> `jetson-containers` (the usual fallback) has been unmaintained since its
> maintainer stepped away in 2025, so this ships independently instead of waiting
> on that.
>
> Verified on a real Orin NX 16GB: GPU visible to torch, torchcodec does real CUDA
> decode (not a silent CPU fallback), `import lerobot` + a policy import (SmolVLA)
> both work. Full build ~5.5h from scratch (source-compiling torch is 92% of that),
> published to GHCR so you don't have to:
>
> ```
> docker pull ghcr.io/ravediamond/lerobot-jetson:latest
> ```
>
> Repo + build notes: https://github.com/ravediamond/lerobot-jetson
>
> Plan to keep it updated for new lerobot releases. Happy to answer questions if
> anyone's hitting the same wall.

## Draft: LinkedIn

> No official or community channel ships a PyTorch wheel with CUDA support for
> Jetson + Python 3.12 — not NVIDIA's own redist, not the community index. That
> silently blocks `pip install lerobot` on Jetson hardware, and the usual fallback
> project for Jetson containers has been unmaintained since its maintainer stepped
> away in 2025.
>
> Spent [time] building torch and torchcodec from source for Jetson Orin (JetPack
> 6.2, CUDA 12.6, Python 3.12) — packaged as a Docker image so nobody else has to
> rediscover the same build gotchas (OOM'ing flash-attention kernels, a broken
> `torch.distributed` stub that crashes training on a single-GPU board, a
> torchvision ABI mismatch that installs cleanly but silently fails at runtime).
>
> Verified on real hardware, not just built: GPU visible to torch, real CUDA video
> decode, LeRobot itself and a real policy (SmolVLA) both import and run. Full
> build is ~5.5 hours, 92% of that is compiling PyTorch alone.
>
> Published to GHCR, plan to keep it updated for new LeRobot releases:
> https://github.com/ravediamond/lerobot-jetson
>
> [optional: mention posted on huggingface/lerobot#819, if it feels natural rather
> than tacked-on]
