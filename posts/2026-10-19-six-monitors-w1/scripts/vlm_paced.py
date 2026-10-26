"""Run a repo script with the OpenAI-compatible backend paced and retried on 429/5xx (free-tier Gemini).

  python paced.py <repo root> <min seconds between calls> <api log .jsonl> <module> -- <script args>

Pacing sleeps happen before VLMJudge.judge starts its clock (and before each label call), so the latency the script
reports is the API time plus any 429 retries, not the pacing. Every API call's time goes to the .jsonl log.
"""
import json
import os
import sys
import time
import urllib.error

root, gap, log_path, module = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
args = sys.argv[sys.argv.index("--") + 1:]
sys.path[:0] = [root, root + "/src", root + "/scripts"]

from agent.supervision import vlm_judge  # noqa: E402

orig_call = vlm_judge.OpenAICompatBackend.__call__
orig_judge = vlm_judge.VLMJudge.judge
last = [0.0]
MAX_TOKENS = 4096


def pace():
    wait = gap - (time.time() - last[0])
    if wait > 0:
        time.sleep(wait)


def paced_call(self, image_b64, prompt):
    self.max_tokens = max(self.max_tokens, MAX_TOKENS)  # Gemini 3.x counts thinking tokens in max_tokens
    self.timeout_s = max(self.timeout_s, 180.0)  # thinking calls take 7-40 s
    for attempt in range(8):
        pace()
        last[0] = t0 = time.time()
        try:
            out = orig_call(self, image_b64, prompt)
            with open(log_path, "a") as f:
                f.write(json.dumps({"ms": round((time.time() - t0) * 1000), "attempt": attempt}) + "\n")
            return out
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                abort(f"HTTP {exc.code}: {exc.read()[:300]!r}")
            err = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            err = repr(exc)
        back = min(120, 10 * 2 ** attempt)
        print(f"    {err}, retry in {back}s", flush=True)
        time.sleep(back)
    abort("gave up after 8 retries")


def abort(msg):
    # VLMJudge turns any backend exception into a non-verdict (scored as a pass): stop instead, both scripts resume
    print(f"ABORT: {msg}", flush=True)
    os._exit(3)


def paced_judge(self, frames, telemetry):
    pace()
    return orig_judge(self, frames, telemetry)


vlm_judge.OpenAICompatBackend.__call__ = paced_call
vlm_judge.VLMJudge.judge = paced_judge

mod = __import__(module)
raise SystemExit(mod.main(args))
