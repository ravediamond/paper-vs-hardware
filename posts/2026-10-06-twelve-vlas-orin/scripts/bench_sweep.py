#!/usr/bin/env python3
"""Orin NX benchmark sweep following docs/benchmark/TEMPLATE.md.

Per model: download + sha256 check, full protocol at the defaults (3 processes x
3 warmups + 20 reps, keep the lowest mean, peak RSS over all three), flag screen
(2 warmups + 5 reps, one process per set), full re-run of the fastest set, and the
model file is deleted afterwards (disk holds one model at a time).

Results are appended as JSON lines to RESULTS; a model already present there is skipped,
so the sweep can be restarted after an interruption.
"""
import hashlib, json, os, re, resource, subprocess, sys, time, urllib.request
from pathlib import Path

BENCH = Path.home() / "vla.cpp/build-bench/vla-bench"
MODELS_DIR = Path.home() / "vla-bench-models"
RESULTS = Path.home() / "vla-bench-results/orin_nx.jsonl"
LOG = Path.home() / "vla-bench-results/orin_nx.log"
PHASES = Path.home() / "vla-bench-results/orin_nx.phases.jsonl"
CURRENT = [""]  # model being measured, for the phase markers

# name, repo, file, sha256 prefix, views, size, extra args, env
MODELS = [
    ("Octo-Small",  "octo-small-libero-gguf", "octo-small-libero-f32.gguf", "9b4e719b8011", 2, 256, [], {"VLA_OCTO_UNNORM_DATASET": "libero_object"}),
    ("TurboVLA",    "turbovla-libero-gguf", "turbovla-libero-f32.gguf", "6f44aa37f65d", 2, 256, [], {}),
    ("SmolVLA",     "smolvla-libero-gguf", "smolvla-libero.gguf", "6fb2d475c98b", 2, 512, [], {}),
    ("Evo-1",       "evo1-libero-gguf", "evo1-libero.gguf", "ef61ec7c4041", 1, 448, [], {}),
    ("BitVLA",      "bitvla-libero-gguf", "libero_object/bitvla-libero-object.gguf", "3a1dac4a1b56", 1, 224, [], {}),
    ("VLA-Adapter", "vla-adapter-libero-gguf", "libero_object/vla-adapter-libero-object.gguf", "d26b57644ce2", 1, 224, [], {}),
    ("VLA-JEPA",    "vla-jepa-libero", "vla-jepa.gguf", "c65a52613ed6", 1, 256, ["--extra-token", "151697", "--extra-count", "32"], {}),
    ("GR00T N1.7",  "gr00tn1d7-libero-gguf", "libero_object/gr00tn1d7-libero-object.gguf", "616c19a51035", 1, 256, [], {"VLA_GR00T_EMBODIMENT": "libero_sim"}),
    ("π0",          "pi0-libero-finetuned-v044-gguf", "pi0-libero-finetuned-v044.gguf", "7d211cae7098", 2, 224, [], {}),
    ("GR00T N1.5",  "gr00tn1d5-libero-object-gguf", "gr00tn1d5-libero-object.gguf", "d12d3ecffae4", 1, 224, [], {"VLA_GR00T_EMBODIMENT": "new_embodiment"}),
    ("GR00T N1.6",  "gr00tn1d6-libero-gguf", "gr00tn1d6-libero.gguf", "f93052789fb4", 1, 224, [], {"VLA_GR00T_EMBODIMENT": "libero_panda"}),
    ("π0.5",        "pi05-libero-gguf", "pi05-libero.gguf", "f9b585b12cbe", 2, 224, [], {}),
]

SCREEN = [
    [],
    ["--flash-attn"],
    ["--mm-prec", "default"],
    ["--flash-attn", "--mm-prec", "default"],
    ["--weight-dtype", "bf16"],
    ["--weight-dtype", "bf16", "--flash-attn"],
    ["--weight-dtype", "f16"],
    ["--weight-dtype", "f16", "--flash-attn"],
]
ACT_BF16 = ["--act-dtype", "bf16", "--flash-attn"]  # pi0 and Evo-1 on CUDA only

LINE = re.compile(r"min ([\d.]+) ms\s+mean ([\d.]+) ms\s+p50 ([\d.]+) ms\s+p90 ([\d.]+) ms\s+vision ([\d.]+) ms")


def log(msg):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a") as f:
        f.write(f"{stamp} {msg}\n")
    print(f"{stamp} {msg}", flush=True)


def run_once(path, views, size, extra, env, flags, warmup, reps, kind="bench"):
    """One vla-bench process; returns parsed timings plus this process's peak RSS in MiB."""
    cmd = [str(BENCH), "--ckpt", str(path), "--images", str(views), "--size", str(size),
           "--warmup", str(warmup), "--reps", str(reps), *extra, *flags]
    # fork a tiny python child per run so RUSAGE_CHILDREN reports this process only
    probe = ("import resource,subprocess,sys,json;"
             "p=subprocess.run(sys.argv[1:],capture_output=True,text=True);"
             "r=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss;"
             "print(json.dumps({'rc':p.returncode,'out':p.stdout,'err':p.stderr,'rss_kib':r}))")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", probe, *cmd], capture_output=True, text=True,
                       env={**os.environ, **env})
    t1 = time.time()
    res = json.loads(p.stdout)
    m = LINE.search(res["out"])
    if res["rc"] != 0 or not m:
        tail = (res["err"] or res["out"]).strip().splitlines()[-8:]
        return {"ok": False, "cmd": " ".join(cmd), "rc": res["rc"], "err": tail}
    mn, mean, p50, p90, vis = map(float, m.groups())
    # phase window for tegra_summary.py: the timed reps are the last reps*mean before exit
    with PHASES.open("a") as f:
        f.write(json.dumps({"model": CURRENT[0], "kind": kind, "flags": flags, "warmup": warmup, "reps": reps,
                            "mean_ms": mean, "t0": t0, "t1": t1,
                            "timed_from": t1 - reps * mean / 1000.0}, ensure_ascii=False) + "\n")
    return {"ok": True, "min": mn, "mean": mean, "p50": p50, "p90": p90, "vision": vis,
            "rss_mib": round(res["rss_kib"] / 1024), "flags": flags}


def full(path, views, size, extra, env, flags):
    runs = [run_once(path, views, size, extra, env, flags, 3, 20) for _ in range(3)]
    ok = [r for r in runs if r["ok"]]
    if len(ok) < 3:
        return {"ok": False, "runs": runs}
    best = min(ok, key=lambda r: r["mean"])
    return {**best, "rss_mib": max(r["rss_mib"] for r in ok), "runs": runs}


def profile(path, views, size, extra, env, flags, mean_ms):
    """Separate run for the tegrastats summary only: ~10 s of steady predict() calls.
    Not part of the reported latencies, which follow the template protocol."""
    reps = max(20, int(10_000 / max(mean_ms, 1.0)))
    log(f"{CURRENT[0]}: profile {' '.join(flags) or '(defaults)'} reps={reps}")
    return run_once(path, views, size, extra, env, flags, 3, reps, kind="profile")


def fetch(repo, fname, sha_prefix):
    if repo is None:  # local file: measured in place, never deleted
        return Path(fname), None
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_DIR / Path(fname).name
    if not dst.exists():
        url = f"https://huggingface.co/vrfai/{repo}/resolve/main/{fname}"
        log(f"download {url}")
        tmp = dst.with_suffix(".part")
        subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(tmp), url], check=True)
        tmp.rename(dst)
    h = hashlib.sha256()
    with dst.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if not digest.startswith(sha_prefix):
        raise SystemExit(f"sha256 mismatch for {fname}: {digest}")
    return dst, digest


def main():
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS.exists():
        done = {json.loads(l)["model"] for l in RESULTS.read_text().splitlines() if l.strip()}
    only = set(sys.argv[1:])
    # an extra model outside the vrfai set, e.g. EXTRA="SmolVLA SO-101=/path/model.gguf:2:512"
    if os.environ.get("EXTRA"):
        label, spec = os.environ["EXTRA"].split("=", 1)
        gguf, views, size = spec.rsplit(":", 2)
        MODELS.insert(0, (label, None, gguf, None, int(views), int(size), [], {}))
    for name, repo, fname, sha, views, size, extra, env in MODELS:
        if (only and name not in only) or name in done:
            continue
        rec = {"model": name, "repo": repo, "file": fname, "views": views, "size": size,
               "extra": extra, "env": env, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        CURRENT[0] = name
        try:
            path, rec["sha256"] = fetch(repo, fname, sha)
            log(f"{name}: defaults, full protocol")
            rec["defaults"] = full(path, views, size, extra, env, [])
            if rec["defaults"].get("ok"):
                rec["profile"] = profile(path, views, size, extra, env, [], rec["defaults"]["mean"])
            sets = SCREEN + ([ACT_BF16] if name in ("π0", "Evo-1") else [])
            rec["screen"] = []
            for flags in sets:
                log(f"{name}: screen {' '.join(flags) or '(defaults)'}")
                rec["screen"].append(run_once(path, views, size, extra, env, flags, 2, 5))
            ok = [s for s in rec["screen"] if s["ok"]]
            base = next((s for s in ok if not s["flags"]), None)
            best = min(ok, key=lambda s: s["mean"]) if ok else None
            rec["fastest"] = None
            if base and best and best["flags"] and best["mean"] < 0.97 * base["mean"]:
                log(f"{name}: full re-run of {' '.join(best['flags'])}")
                fr = full(path, views, size, extra, env, best["flags"])
                rec["fastest_rerun"] = fr
                d = rec["defaults"]
                if fr["ok"] and d["ok"] and fr["mean"] < 0.97 * d["mean"]:
                    rec["fastest"] = fr
                    rec["profile_fastest"] = profile(path, views, size, extra, env, fr["flags"], fr["mean"])
        except Exception as e:  # keep going; the row records why
            rec["error"] = repr(e)
            log(f"{name}: ERROR {e!r}")
        rec["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with RESULTS.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if repo is not None:
            for p in MODELS_DIR.glob(Path(fname).name + "*"):
                p.unlink()
        log(f"{name}: done")
    log("sweep finished")


if __name__ == "__main__":
    main()
