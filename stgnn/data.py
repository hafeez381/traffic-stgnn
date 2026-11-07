"""Load the San Diego tensor, split it by date, normalise with training statistics, gather windows.

The assignment's information-flow rules are checked here at run time:
  - the three splits are the stated date ranges, with the expected number of steps
  - no window crosses a split boundary (windows are built inside each split)
  - normalisation statistics come from training timestamps only
"""
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from stgnn.config import CONFIG


def load(path):
    d = np.load(path)
    return d["flow"].astype(np.float32), d["timestamps"].astype(np.int64), d["ids"]


def split_masks(timestamps, cfg=CONFIG):
    masks = {}
    for name in ("train", "val", "test"):
        a, b = (pd.Timestamp(s).value for s in cfg[name])
        masks[name] = (timestamps >= a) & (timestamps < b)
        n = int(masks[name].sum())
        if n != cfg["split_steps"][name]:
            raise ValueError(f"{name} split has {n} steps, expected {cfg['split_steps'][name]}")
    covered = masks["train"].astype(int) + masks["val"] + masks["test"]
    if covered.max() > 1 or covered.sum() != len(timestamps):
        raise ValueError("splits overlap or leave steps unassigned")
    return masks


class Scaler:
    """One mean and standard deviation over all observed training values."""

    def __init__(self):
        self.mean = self.std = None

    def fit(self, values, timestamps, train_end=CONFIG["train"][1]):
        if timestamps.max() >= pd.Timestamp(train_end).value:
            raise ValueError("normalisation statistics must come from the training period only")
        self.mean = float(np.nanmean(values))
        self.std = float(np.nanstd(values))
        return self

    def transform(self, x):
        if self.mean is None:
            raise RuntimeError("Scaler used before fit")
        return (x - self.mean) / self.std

    def inverse(self, x):
        if self.mean is None:
            raise RuntimeError("Scaler used before fit")
        return x * self.std + self.mean


@dataclass
class Split:
    name: str
    x: torch.Tensor        # (T, N) normalised inputs, missing -> 0 (the training mean)
    y: torch.Tensor        # (T, N) original-unit targets, missing -> 0 (masked out)
    mask: torch.Tensor     # (T, N) bool, True where the reading exists
    ends: torch.Tensor     # (W,) window end index t, so inputs are t-7..t and targets t+1..t+4
    timestamps: np.ndarray

    def windows(self, t, in_len=CONFIG["in_len"], out_len=CONFIG["out_len"]):
        """Gather a batch of windows for end indices t (B,): X (B, 8, N), Y and M (B, 4, N)."""
        past = t[:, None] + torch.arange(-in_len + 1, 1, device=t.device)
        future = t[:, None] + torch.arange(1, out_len + 1, device=t.device)
        return self.x[past], self.y[future], self.mask[future]

    def ends_sha256(self):
        return hashlib.sha256(self.timestamps[self.ends.cpu().numpy()].tobytes()).hexdigest()


def window_ends(n_steps, in_len=CONFIG["in_len"], out_len=CONFIG["out_len"]):
    ends = np.arange(in_len - 1, n_steps - out_len)
    if len(ends) == 0 or ends[0] - (in_len - 1) < 0 or ends[-1] + out_len > n_steps - 1:
        raise ValueError("a window crosses its split boundary")
    return ends


def build_splits(flow, timestamps, device="cpu", cfg=CONFIG):
    masks = split_masks(timestamps, cfg)
    scaler = Scaler().fit(flow[masks["train"]], timestamps[masks["train"]], cfg["train"][1])
    splits = {}
    for name, m in masks.items():
        raw = flow[m]
        observed = ~np.isnan(raw)
        x = np.where(observed, scaler.transform(raw), 0.0).astype(np.float32)
        y = np.where(observed, raw, 0.0).astype(np.float32)
        ends = window_ends(len(raw), cfg["in_len"], cfg["out_len"])
        splits[name] = Split(name, torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
                             torch.from_numpy(observed).to(device), torch.from_numpy(ends).to(device),
                             timestamps[m])
    return splits, scaler


def synthetic(num_nodes=CONFIG["num_nodes"], seed=0):
    """SD-shaped stand-in for smoke runs: daily cycles, noise, 5% missing, sparse asymmetric graph."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", "2021-12-31 23:45", freq="15min").as_unit("ns").asi8
    tod = (timestamps // (15 * 60 * 10**9)) % 96
    level = rng.uniform(100, 1500, num_nodes)
    flow = level * (0.6 + 0.4 * np.sin(2 * np.pi * tod / 96))[:, None] + rng.normal(0, 30, (len(tod), num_nodes))
    flow = flow.astype(np.float32)
    flow[rng.random(flow.shape) < 0.05] = np.nan
    adj = (rng.random((num_nodes, num_nodes)) < 0.03) * rng.uniform(0.01, 1, (num_nodes, num_nodes))
    np.fill_diagonal(adj, 1.0)
    return flow, timestamps, np.array([str(i) for i in range(num_nodes)]), adj.astype(np.float32)
