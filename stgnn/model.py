"""Temporal convolution per sensor -> one graph convolution across sensors -> linear forecast head."""
import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """Three unpadded convolutions over the 8 input steps: lengths 8 -> 6 -> 2 -> 1.

    Kernel 3, then kernel 3 with dilation 2, then kernel 2: the receptive field is exactly the 8
    inputs, with no padding. Weights are shared by all sensors.
    """

    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, 3), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, dilation=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 2), nn.ReLU(),
        )

    def forward(self, x):  # x (B, 8, N) -> Z (B, N, d)
        b, t, n = x.shape
        z = self.net(x.permute(0, 2, 1).reshape(b * n, 1, t))
        assert z.shape[-1] == 1, z.shape
        return z.reshape(b, n, -1)


class STModel(nn.Module):
    """graph=None gives the temporal-only model: same encoder, same W_g, same head, no neighbours."""

    def __init__(self, hidden, graph=None, out_len=4):
        super().__init__()
        self.encoder = TemporalEncoder(hidden)
        self.w_g = nn.Linear(hidden, hidden)
        self.graph = graph
        self.head = nn.Linear(hidden, out_len)

    def forward(self, x):  # x (B, 8, N) -> (B, 4, N)
        z = self.w_g(self.encoder(x))
        if self.graph is not None:
            # H = ReLU(A_hat Z W_g): the product runs over the node axis, separately for each batch item
            z = torch.einsum("ij,bjd->bid", self.graph(), z)
        return self.head(torch.relu(z)).permute(0, 2, 1)
