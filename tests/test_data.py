"""Each information-flow guard must fire on a bad input, and the windows must sit where they should."""
import numpy as np
import pandas as pd
import pytest
import torch

from stgnn.config import CONFIG
from stgnn.data import Scaler, build_splits, split_masks, synthetic, window_ends
from stgnn.graph import AdaptiveAdjacency, HybridAdjacency, normalize_fixed
from stgnn.model import STModel
from stgnn.train import make_model, train_step


@pytest.fixture(scope="module")
def data():
    flow, ts, ids, adj = synthetic(num_nodes=20)
    cfg = dict(CONFIG, num_nodes=20)
    splits, scaler = build_splits(flow, ts, "cpu", cfg)
    return flow, ts, adj, cfg, splits, scaler


def test_window_counts_and_edges(data):
    flow, ts, adj, cfg, splits, scaler = data
    assert [len(splits[s].ends) for s in ("train", "val", "test")] == [35125, 8629, 26389]
    first = splits["train"].ends[0].item()
    assert pd.Timestamp(splits["train"].timestamps[first - 7]) == pd.Timestamp("2020-01-01 00:00")
    last = splits["test"].ends[-1].item()
    assert pd.Timestamp(splits["test"].timestamps[last + 4]) == pd.Timestamp("2021-12-31 23:45")


def test_window_contents(data):
    flow, ts, adj, cfg, splits, scaler = data
    s = splits["val"]
    t = s.ends[:3]
    x, y, m = s.windows(t)
    assert x.shape == (3, 8, 20) and y.shape == (3, 4, 20)
    raw = flow[split_masks(ts)["val"]]
    i = t[0].item()
    expect_y = np.nan_to_num(raw[i + 1:i + 5], nan=0.0)
    np.testing.assert_allclose(y[0].numpy(), expect_y)          # targets start after the last input
    np.testing.assert_allclose(m[0].numpy(), ~np.isnan(raw[i + 1:i + 5]))
    expect_x = np.nan_to_num(scaler.transform(raw[i - 7:i + 1]), nan=0.0)
    np.testing.assert_allclose(x[0].numpy(), expect_x, rtol=1e-5, atol=1e-5)


def test_split_lengths_guard(data):
    flow, ts, *_ = data
    with pytest.raises(ValueError):
        split_masks(ts[:-1])


def test_window_boundary_guard():
    assert window_ends(20)[0] == 7 and window_ends(20)[-1] == 15
    with pytest.raises(ValueError):
        window_ends(11)  # a split too short for one full window


def test_scaler_train_only(data):
    flow, ts, *_ = data
    with pytest.raises(ValueError):
        Scaler().fit(flow, ts)  # includes 2021
    with pytest.raises(RuntimeError):
        Scaler().transform(flow)


def test_scaler_statistics(data):
    flow, ts, adj, cfg, splits, scaler = data
    train = flow[split_masks(ts)["train"]]
    z = scaler.transform(train)
    assert abs(np.nanmean(z)) < 1e-3 and abs(np.nanstd(z) - 1) < 1e-3


def test_no_update_outside_train(data):
    flow, ts, adj, cfg, splits, scaler = data
    model, _ = make_model("hybrid", adj, cfg)
    opt = torch.optim.Adam(model.parameters())
    with pytest.raises(RuntimeError):
        train_step(model, opt, splits["val"], splits["val"].ends[:4], scaler)
    train_step(model, opt, splits["train"], splits["train"].ends[:4], scaler)


def test_graphs_row_stochastic(data):
    adj = data[2]
    a_hat, kind = normalize_fixed(adj)
    assert kind == "row"
    np.testing.assert_allclose(a_hat.sum(1), 1, atol=1e-5)
    sym = (adj + adj.T) / 2
    a_sym, kind = normalize_fixed(sym)
    assert kind == "symmetric" and np.abs(np.linalg.eigvalsh(a_sym)).max() <= 1 + 1e-5
    ada = AdaptiveAdjacency(20, 10)()
    torch.testing.assert_close(ada.sum(1), torch.ones(20))
    hyb = HybridAdjacency(a_hat, 20, 10)
    torch.testing.assert_close(hyb().sum(1), torch.ones(20))
    assert hyb.alpha.item() == 0.5


def test_self_loop_not_doubled():
    adj = np.array([[1.0, 0.5], [0.0, 1.0]])
    a_hat, _ = normalize_fixed(adj)
    np.testing.assert_allclose(a_hat, [[1 / 1.5, 0.5 / 1.5], [0, 1]], rtol=1e-6)


def test_receptive_field_and_equal_parameters(data):
    adj, cfg = data[2], data[3]
    temporal, _ = make_model("temporal", adj, cfg)
    fixed, _ = make_model("fixed", adj, cfg)
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(temporal) == count(fixed)
    x = torch.randn(1, 8, 20)
    z0 = temporal.encoder(x)
    x[0, 0, 5] += 1.0  # perturb the oldest input at sensor 5
    z1 = temporal.encoder(x)
    assert not torch.allclose(z0[0, 5], z1[0, 5]) and torch.allclose(z0[0, 4], z1[0, 4])
    assert STModel(32)(torch.randn(2, 8, 20)).shape == (2, 4, 20)
