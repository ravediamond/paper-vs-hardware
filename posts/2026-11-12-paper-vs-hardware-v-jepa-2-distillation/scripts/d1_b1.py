"""WS-110 B1 decision (D1_PROTOCOL.md §3, §7 G4): D1 student vs random-teacher student vs W1-scratch on the §30 term, one shared episode bootstrap.

Reads the per-anchor predictions the two instances returned (d1_preds.npz: W1, W1_random_init, D1_best, D1_final; rt_preds.npz: same
W1 rows + RT_best, RT_final), checks that the shared W1 rows are identical, and runs d1_compare.pooled (2,000 resamples, seed 0, percentile
2.5 / 97.5, the max inside each resample) over all rows at once.
B1 (as written): some student beats W1-scratch with CI lower bound > 0 AND beats the random-teacher control on the same probe (lower bound > 0).
Decision row = best-val checkpoint (O2); the final-step pairs are reported, labelled exploratory.

    python d1_b1.py --d1 results/d1/d1_preds.npz --rt results/d1/rt_preds.npz --out results/d1/b1.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1_compare as C  # noqa: E402
from gap5_compare import per_from_preds  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d1", type=Path, required=True)
    ap.add_argument("--rt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    per, sd = per_from_preds(a.d1)
    per_rt, sd_rt = per_from_preds(a.rt)
    assert np.allclose(sd, sd_rt)
    gap = {}
    for enc in ("W1", "W1_random_init"):          # same frames, stored rows: ridge is BLAS-sensitive (notes), so report the drift, bound it
        d = max(float(np.abs(per[enc][w][e][f] - per_rt[enc][w][e][f]).max()) for w in per[enc] for e in per[enc][w] for f in ("pred",))
        eps0 = sorted(per[enc]["ii"])
        t1, t2 = C.term_of(per[enc], sd, eps0), C.term_of(per_rt[enc], sd, eps0)
        gap[enc] = {"max_abs_pred_diff": d, "term_d1_instance": t1, "term_rt_instance": t2}
        assert abs(t1 - t2) < 0.02, (enc, t1, t2)
    per.update({k: v for k, v in per_rt.items() if k.startswith("RT_")})
    pairs = [("D1_best", "W1"), ("RT_best", "W1"), ("D1_best", "RT_best"), ("D1_best", "W1_random_init"), ("RT_best", "W1_random_init"),
             ("D1_final", "W1"), ("RT_final", "W1"), ("D1_final", "RT_final")]
    pp = C.pooled(per, sd, pairs)
    eps = sorted(per["W1"]["ii"])
    terms = {e: C.term_of(per[e], sd, eps) for e in per}
    lo = {k: v["ci95"][0] for k, v in pp.items()}
    b1_best = lo["D1_best_minus_W1"] > 0 and lo["D1_best_minus_RT_best"] > 0
    res = {"w1_rows_across_instances": gap, "terms": terms, "pairs": pp, "val_episodes": len(eps),
           "B1_decision_row_best": {"D1_beats_W1": lo["D1_best_minus_W1"] > 0, "D1_beats_random_teacher_student": lo["D1_best_minus_RT_best"] > 0,
                                    "B1_met": bool(b1_best)},
           "exploratory_final_checkpoint": {"D1_final_beats_W1": lo["D1_final_minus_W1"] > 0,
                                            "D1_final_beats_RT_final": lo["D1_final_minus_RT_final"] > 0}}
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
