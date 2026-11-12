"""WS-107 alignment metric and bootstrap, exactly PROTOCOL.md §4-§5 (mutual 10-NN, same-episode exclusion, ZCA whitening, seed pairing,
episode bootstrap). Pure numpy; no model code.

Frames: one row per (episode, raw frame); `elig` marks the common eligible set (5 fps index >= 2, i.e. raw frame >= 12).
Whitening uses every row (the full eval set N), neighbours are searched among eligible rows only.
"""
import numpy as np

K = 10
EPS = 1e-6


def zca(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """ZCA whitening, mean and covariance from x itself, eps added to the eigenvalues (§4)."""
    x = x.astype(np.float64)
    xc = x - x.mean(0)
    cov = xc.T @ xc / (len(x) - 1)
    w, v = np.linalg.eigh(cov)
    return xc @ (v @ np.diag(1.0 / np.sqrt(np.maximum(w, 0) + eps)) @ v.T)


def cos_sim(z: np.ndarray) -> np.ndarray:
    z = z / np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-12)
    return z @ z.T


def topk_idx(sim: np.ndarray, excl: np.ndarray, k: int = K) -> np.ndarray:
    """sim (M, M) cosine similarity, excl (M, M) bool True = not a candidate. Returns (M, k) column indices of the k most similar
    candidates, ties broken by column index (rows/columns are ordered by (episode copy, frame), so by frame index)."""
    s = np.where(excl, -np.inf, sim)
    m = s.shape[1]
    c = min(m, 4 * k + 30)
    cand = np.sort(np.argpartition(-s, c - 1, axis=1)[:, :c], axis=1)  # by column index, so the stable sort below breaks ties by index
    val = np.take_along_axis(s, cand, 1)
    order = np.argsort(-val, axis=1, kind="stable")
    return np.take_along_axis(cand, order, 1)[:, :k]


def mutual_scores(sim_a: np.ndarray, sim_b: np.ndarray, excl: np.ndarray, k: int = K) -> np.ndarray:
    """score_i = |NN_A(i) ∩ NN_B(i)| / k for every row i."""
    a, b = topk_idx(sim_a, excl, k), topk_idx(sim_b, excl, k)
    return np.array([len(set(x) & set(y)) for x, y in zip(a, b)]) / k


class Space:
    """One model's embeddings on the eligible frames, whitened with the full-set statistics. Keeps, per row, the candidates ranked by
    cosine similarity (ties by column index = frame order), first L of them. ep_ids = compact episode index (0..n_ep-1) of each eligible row."""

    def __init__(self, emb_all: np.ndarray, elig: np.ndarray, ep_ids: np.ndarray | None = None, L: int = 400):
        self.sim = cos_sim(zca(emb_all)[elig])
        if ep_ids is not None:
            self.ep = ep_ids
            self.R = np.argsort(-self.sim, axis=1, kind="stable")[:, :L].astype(np.int32)
            self.epR = ep_ids[self.R]
            self.own = self.epR == ep_ids[:, None]


def taken(sp: "Space", w: np.ndarray, k: int = K) -> np.ndarray:
    """Dense (M, M) int8: how many copies of candidate column j are among row i's k nearest in the resample whose episode multiplicities are w
    (w[e] = number of copies of episode e; the rows' own episode, all copies, is never a candidate). Equivalent to the §4 rule on the
    multiset of rows, ties between copies of one frame broken by copy order, the same order in both spaces."""
    W = w[sp.epR] * (~sp.own)
    cum = np.cumsum(W, axis=1)
    assert (cum[:, -1] >= k).all(), "increase L"
    t = np.minimum(W, np.maximum(0, k - (cum - W))).astype(np.int8)
    out = np.zeros(sp.sim.shape, np.int8)
    np.put_along_axis(out, sp.R, t, 1)
    return out


def row_scores(ta: np.ndarray, tb: np.ndarray, k: int = K) -> np.ndarray:
    """score_i = |NN_A(i) ∩ NN_B(i)| / k for every eligible row i (the same for all copies of the row)."""
    return np.minimum(ta, tb).sum(1) / k


def pair_scores(sa: Space, sb: Space, ep: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
    """Per-row mutual 10-NN score for the rows `idx` of the eligible set (default all; repeats = bootstrap copies)."""
    if idx is None:
        idx = np.arange(len(ep))
    e = ep[idx]
    excl = e[:, None] == e[None, :]  # a duplicated episode is a distinct copy, excluded by its original id
    return mutual_scores(sa.sim[np.ix_(idx, idx)], sb.sim[np.ix_(idx, idx)], excl)


def bootstrap_index(ep: np.ndarray, draw_eps: np.ndarray) -> np.ndarray:
    """Row indices of the resample whose episodes are `draw_eps` (with repeats), rows ordered by (draw position, frame)."""
    by = {int(e): np.nonzero(ep == e)[0] for e in np.unique(ep)}
    return np.concatenate([by[int(e)] for e in draw_eps])


def ladder_value(pairs, ep, idx=None):
    """Mean over seed-matched pairs (list of (Space, Space)) of the mean per-row score."""
    return float(np.mean([pair_scores(a, b, ep, idx).mean() for a, b in pairs]))


def episode_bootstrap(fn, n_ep, n_boot=2000, seed=0):
    """fn(w) -> dict of named values for the resample whose episode multiplicities are w (length n_ep); one resample of the n_ep episodes
    with replacement per draw, shared by all values and seeds. numpy default_rng(seed); episodes indexed in sorted id order."""
    rng = np.random.default_rng(seed)
    out = {}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_ep, n_ep), minlength=n_ep)
        for k, v in fn(w).items():
            out.setdefault(k, []).append(v)
    return {k: np.array(v) for k, v in out.items()}


def ci95(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def verdict(ci, one_sided_pass_positive=True):
    """pass: CI entirely > 0; fail: entirely < 0; not proven: CI contains 0 (§5, B1 and B2 on the paired difference)."""
    lo, hi = ci
    return "pass" if lo > 0 else ("fail" if hi < 0 else "not proven")
