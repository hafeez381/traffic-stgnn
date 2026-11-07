"""Masked errors: missing readings never enter the loss or the reported metrics."""
import math

import torch


def masked_mae(pred, target, mask):
    m = mask.float()
    return (torch.abs(pred - target) * m).sum() / m.sum().clamp(min=1)


class ErrorSums:
    """Accumulates absolute and squared errors over a whole split, so RMSE takes one root at the end."""

    def __init__(self):
        self.abs = self.sq = 0.0
        self.count = 0

    def add(self, pred, target, mask):
        m = mask.float()
        err = (pred - target) * m
        self.abs += err.abs().sum().item()
        self.sq += (err ** 2).sum().item()
        self.count += int(m.sum().item())

    @property
    def mae(self):
        return self.abs / self.count

    @property
    def rmse(self):
        return math.sqrt(self.sq / self.count)
