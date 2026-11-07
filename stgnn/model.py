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
    """Graph layer H = ReLU(A_hat Z W_g + Z W_s), then a linear head.

    The self term Z W_s keeps each sensor's own features at full weight. Without it, row
    normalisation leaves a sensor a median weight of 0.14 on itself in the SD road graph (and
    about 1/716 in a dense learned graph), so its forecast is pulled toward its neighbours' level.
    graph=None gives the temporal-only model: the same layers with no neighbour term, i.e.
    H = ReLU(Z W_g + Z W_s), so every model has the same encoder, weights and head.
    """

    def __init__(self, hidden, graph=None, out_len=4):
        super().__init__()
        self.encoder = TemporalEncoder(hidden)
        self.w_g = nn.Linear(hidden, hidden)
        self.w_s = nn.Linear(hidden, hidden, bias=False)
        self.graph = graph
        self.head = nn.Linear(hidden, out_len)

    def forward(self, x):  # x (B, 8, N) -> (B, 4, N)
        z = self.encoder(x)
        neighbours = self.w_g(z)
        if self.graph is not None:
            # A_hat Z W_g: the product runs over the node axis, separately for each batch item
            neighbours = torch.einsum("ij,bjd->bid", self.graph(), neighbours)
        return self.head(torch.relu(neighbours + self.w_s(z))).permute(0, 2, 1)
