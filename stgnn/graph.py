"""The three graph constructions: fixed road graph, learned (adaptive) graph, and their mixture.

All three return a row-stochastic N x N matrix (every row sums to 1), so the graph layer always
takes a weighted average over a sensor's neighbours and mixing two of them stays row-stochastic.
"""
import numpy as np
import torch
from torch import nn


def normalize_fixed(adj):
    """Normalise the road-distance adjacency.

    LargeST stores nonzero self-entries, so the diagonal is cleared before adding the identity;
    otherwise each sensor would count itself twice. A symmetric graph gets D^-1/2 (A+I) D^-1/2.
    The SD graph is directed (driving distance i->j need not equal j->i), so it gets the row
    normalisation D_out^-1 (A+I): row i averages sensor i and the sensors in its row.
    """
    a = np.array(adj, dtype=np.float64)
    np.fill_diagonal(a, 0.0)
    a = a + np.eye(len(a))
    deg = a.sum(axis=1)
    if np.allclose(a, a.T):
        d = 1.0 / np.sqrt(deg)
        return (d[:, None] * a * d[None, :]).astype(np.float32), "symmetric"
    return (a / deg[:, None]).astype(np.float32), "row"


class FixedAdjacency(nn.Module):
    def __init__(self, a_hat):
        super().__init__()
        self.register_buffer("a_hat", torch.as_tensor(a_hat))  # a buffer: moved to the device, never trained

    def forward(self):
        return self.a_hat


class AdaptiveAdjacency(nn.Module):
    """A = row-softmax(ReLU(E1 E2^T)), learned from data.

    ReLU drops negative similarities; the row softmax makes each row a probability distribution
    over all 716 sensors. Two different embeddings let the learned graph be asymmetric.
    """

    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.e2 = nn.Parameter(torch.randn(num_nodes, embed_dim))

    def forward(self):
        return torch.softmax(torch.relu(self.e1 @ self.e2.T), dim=1)


class HybridAdjacency(nn.Module):
    """alpha * fixed + (1 - alpha) * adaptive, with alpha = sigmoid(theta) learned, starting at 0.5."""

    def __init__(self, a_hat, num_nodes, embed_dim):
        super().__init__()
        self.fixed = FixedAdjacency(a_hat)
        self.adaptive = AdaptiveAdjacency(num_nodes, embed_dim)
        self.theta = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self):
        return torch.sigmoid(self.theta)

    def forward(self):
        return self.alpha * self.fixed() + (1 - self.alpha) * self.adaptive()
