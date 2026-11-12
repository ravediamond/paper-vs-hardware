"""PROTOCOL.md §6.2: hand-computed examples for the mutual 10-NN, the exclusion mask and the bootstrap plumbing. python3 -I test_align.py"""
import numpy as np
import align as A

rng = np.random.default_rng(1)


def space(x, elig=None, ep=None):
    elig = np.ones(len(x), bool) if elig is None else elig
    return A.Space(x, elig, ep)


def test_known_neighbours_hand_computed():
    # 12 points on a line in 1-D embedded in 3-D: neighbours of the end point are the next ones along the line, excluded rows removed
    x = np.zeros((12, 3)); x[:, 0] = np.arange(12) ** 1.5 + 1; x[:, 1] = 1.0
    sim = A.cos_sim(x)
    excl = np.eye(12, dtype=bool)
    nn = A.topk_idx(sim, excl, 3)
    # cosine of (a, 1, 0) is increasing in a, so for the largest point the nearest are the next largest
    assert list(nn[11]) == [10, 9, 8]
    assert 0 not in nn[5]  # not a trivial row
    # exclusion: forbid row 10 and 9 for row 11 -> next are 8, 7, 6
    ex2 = excl.copy(); ex2[11, 10] = ex2[11, 9] = True
    assert list(A.topk_idx(sim, ex2, 3)[11]) == [8, 7, 6]


def test_identity_rotation_and_independent():
    n, d = 400, 16
    ep = np.repeat(np.arange(20), 20); w1 = np.ones(20, int)
    x = rng.normal(size=(n, d)) @ rng.normal(size=(d, d))
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    a = space(x, ep=ep)

    def score(sb):
        return A.row_scores(A.taken(a, w1), A.taken(sb, w1)).mean()
    assert np.isclose(score(space(x, ep=ep)), 1.0)
    assert np.isclose(score(space(x @ q, ep=ep)), 1.0)  # whitening makes it rotation invariant
    ind = score(space(rng.normal(size=(n, d)), ep=ep))
    chance = 10 / (n - 20)
    assert abs(ind - chance) < 0.05, (ind, chance)


def test_same_episode_never_a_neighbour():
    n = 120
    ep = np.repeat(np.arange(6), 20)
    x = rng.normal(size=(n, 8))
    a = space(x)
    excl = ep[:, None] == ep[None, :]
    nn = A.topk_idx(a.sim, excl, 10)
    assert all(ep[nn[i]].tolist().count(ep[i]) == 0 for i in range(n))
    # two episodes: rows of an episode copy
    idx = A.bootstrap_index(ep, np.array([2, 2, 5]))
    e = ep[idx]
    nn2 = A.topk_idx(a.sim[np.ix_(idx, idx)], e[:, None] == e[None, :], 10)
    assert all((e[nn2[i]] != e[i]).all() for i in range(len(idx)))


def test_multiplicity_matches_brute_force_on_copies():
    ep = np.repeat(np.arange(8), 15)
    xa, xb = rng.normal(size=(120, 6)), rng.normal(size=(120, 6))
    xb = xa @ rng.normal(size=(6, 6)) * 0.5 + xb * 0.5
    sa, sb = space(xa, ep=ep), space(xb, ep=ep)
    w = np.array([2, 0, 1, 3, 0, 1, 1, 0])
    brute_idx = A.bootstrap_index(ep, np.repeat(np.arange(8), w))
    e = ep[brute_idx]; excl = e[:, None] == e[None, :]
    nA = A.topk_idx(sa.sim[np.ix_(brute_idx, brute_idx)], excl); nB = A.topk_idx(sb.sim[np.ix_(brute_idx, brute_idx)], excl)
    brute = np.array([len(set(x) & set(y)) for x, y in zip(nA, nB)]) / 10
    fast = A.row_scores(A.taken(sa, w), A.taken(sb, w))
    assert np.allclose(brute, fast[brute_idx]), (brute.mean(), fast[brute_idx].mean())


def test_bootstrap_shapes_and_seed():
    ep = np.repeat(np.arange(5), 6)
    out1 = A.episode_bootstrap(lambda w: {"n": int(w.sum())}, 5, n_boot=20, seed=0)
    out2 = A.episode_bootstrap(lambda w: {"n": int(w.sum())}, 5, n_boot=20, seed=0)
    assert (out1["n"] == out2["n"]).all() and (out1["n"] == 5).all()
    assert A.verdict([0.1, 0.3]) == "pass" and A.verdict([-0.3, -0.1]) == "fail" and A.verdict([-0.1, 0.3]) == "not proven"


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
