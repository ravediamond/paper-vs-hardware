"""WS-050 gate 0, step 1: G9c-3's frozen judge on the 69 published rollouts -> judge frame + verdict per episode.

    python3 surveyor_judge.py ~/robot-data/so101-bowls-policy-rollouts <results dir>

Same per-episode logic as judge/g9c3_test.py (judge_frame3, hand scan, episode_score at the judge frame), same frozen
g9c_config.json / g9c2_rule.json. Writes <results>/judge.csv: episode, judge_frame, why, s, invalid, invalid_cause,
verdict (success iff s >= tau). No parameter is changed.
"""

import sys
from pathlib import Path

import pandas as pd

JUDGE = Path(__file__).resolve().parents[2] / "vla-jepa-world-model/judge"
sys.path.insert(0, str(JUDGE))
from g9c2_endrule import load_states  # noqa: E402
from g9c3_test import judge_frame3  # noqa: E402
from g9c_io import Episodes  # noqa: E402
from g9c_judge import episode_score, load_config  # noqa: E402


def main():
    root, out = sys.argv[1], Path(sys.argv[2])
    cfg, rule = load_config(str(JUDGE / "g9c_config.json")), load_config(str(JUDGE / "g9c2_rule.json"))
    E, states = Episodes(root), load_states(root)
    rows = []
    for ep in sorted(states):
        jf, info = judge_frame3(E, ep, states, rule, cfg)
        first, win, _ = E.frames(ep, cfg["max_window_s"], cfg["step_s"], end_frame=jf + 1)
        s, _ = episode_score(first, win, cfg)
        cause = "hand" if info["hand_frame"] is not None else ("undecidable" if s is None else None)
        verdict = None if cause else ("success" if s >= cfg["tau"] else "failure")
        rows.append({"episode": ep, "judge_frame": jf, "why": info["why"], "n_frames": info["n_frames"], "s": s,
                     "invalid": cause is not None, "invalid_cause": cause, "verdict": verdict})
        print(ep, jf, info["why"], verdict, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "judge.csv", index=False)


if __name__ == "__main__":
    main()
