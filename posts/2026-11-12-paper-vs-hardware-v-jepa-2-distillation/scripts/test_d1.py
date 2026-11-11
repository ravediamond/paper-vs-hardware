"""Tests for d1_train.py pure pieces (run on the Orin: no hydra / LeWM needed)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1_train as D  # noqa: E402


def test_pooled_target_is_layer_normed_mean():
    z = np.random.default_rng(0).normal(3, 5, (7, 16, 1024)).astype(np.float16)
    t = D.pooled_target(z).astype(np.float32)
    assert t.shape == (7, 1024)
    assert np.allclose(t.mean(1), 0, atol=5e-3) and np.allclose(t.std(1), 1, atol=5e-3)
    # layer norm of the token mean, not the mean of layer-normed tokens
    ref = torch.nn.functional.layer_norm(torch.from_numpy(z).float().mean(1), (1024,)).numpy()
    assert np.allclose(t, ref, atol=2e-3)


def test_loss_is_cos_plus_mse_one_to_one():
    t = torch.randn(4, 2, 1024)
    assert D.d1_loss(t, t).item() < 1e-6
    p = -t
    assert abs(D.d1_loss(p, t).item() - (2.0 + 4 * t.pow(2).mean().item())) < 0.05
    p = torch.randn(4, 2, 1024)
    cos = (1 - torch.nn.functional.cosine_similarity(p, t, dim=-1)).mean()
    mse = (p - t).pow(2).mean()
    assert abs(D.d1_loss(p, t).item() - (cos + mse).item()) < 1e-5


def test_schedule_and_batches():
    assert D.lr_at(0, 81100) < D.LR * 0.01 and abs(D.lr_at(810, 81100) - D.LR) < 1e-9 and D.lr_at(81099, 81100) < 1e-8
    b = D.epoch_batches(1000, 40, 3072)
    assert len(b) == 40 and all(len(x) == 64 for x in b)
    assert len(set(np.concatenate(b[:15]))) == 15 * 64          # no repeats inside an epoch (drop_last)
    assert all((x == y).all() for x, y in zip(b, D.epoch_batches(1000, 40, 3072)))


def test_cache_alignment(tmp_path):
    n = 10
    mm = {v: np.memmap(tmp_path / f"{v}.u8", dtype=np.uint8, mode="w+", shape=(n, 3, 224, 224)) for v in D.CAMS}
    for j, v in enumerate(D.CAMS):
        for i in range(n):
            mm[v][i] = 10 * j + i
        mm[v].flush()
    (tmp_path / "index.json").write_text(json.dumps({"n": n, "episodes": {"4": [0, 6], "9": [6, 4]}}))
    c = D.Cache(tmp_path)
    f = c.frames(np.array([7, 2]))
    assert f.shape == (2, 2, 3, 224, 224) and f[0, 0, 0, 0, 0] == 7 and f[0, 1, 0, 0, 0] == 17 and f[1, 0, 0, 0, 0] == 2
    assert c.eps == {4: (0, 6), 9: (6, 4)}
    x = D.to_input(f, "cpu")
    assert abs(x[0, 0, 0, 0, 0].item() - ((7 / 255 - 0.485) / 0.229)) < 1e-5
    tg = np.arange(n * 2 * 1024, dtype=np.float16).reshape(n, 2, 1024)
    u8, t = D.BatchData(c, [np.array([3, 5])], tg)[0]
    assert (t.numpy() == tg[[3, 5]]).all() and u8[0, 0, 0, 0, 0] == 3
