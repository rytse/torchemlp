import torch
import torch.nn as nn

import functorch

from torchemlp.groups import Group, O
from torchemlp.reps import Rep, Vector, Scalar, T


class GroupAugmentation(nn.Module):
    """
    Convenience model that wraps a nn.Module with its Representation structure.
    """

    def __init__(self, network: nn.Module, repin: Rep, repout: Rep, G: Group):
        super().__init__()
        self.network = network
        self.repin = repin
        self.repout = repout
        self.G = G

        self.rho_in = functorch.vmap(self.repin.rho)
        self.rho_out = functorch.vmap(self.repout.rho)

    def forward(self, x, training=True):
        if training:
            gs = self.G.samples(x.shape[0])
            rhout_inv = torch.linalg.inv(self.rho_out(gs))
            model_in = (self.rho_in(gs) @ x[..., None])[..., None]
            model_out = self.network(model_in, training)
            return (rhout_inv @ model_out)[..., 0]
        return self.network(x, training)


class Inertia(object):
    def __init__(self, N=1024, k=5):
        self.dim = (1 + 3) * k

        self.X = torch.randn(N, self.dim)

        self.X[:, :k] = torch.log(1 + torch.exp(self.X[:, :k]))  # masses
        mi = self.X[:, :k]
        ri = self.X[:, k:].reshape(-1, k, 3)

        I = torch.eye(3)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (
            mi[:, :, None, None] * (r2 * I - ri[..., None] * ri[..., None, :])
        ).sum(1)
        self.Y = inertia.reshape(-1, 9)

        self.repin = k * Scalar + k * Vector
        self.repout = T(2)
        self.symmetry = O(3)

        Xmean = self.X.mean(0)
        Xmean[k:] = 0.0
        Xstd = torch.zeros_like(Xmean)
        Xstd[:k] = torch.abs(self.X[:, :k]).mean(0)
        Xstd[k:] = (
            torch.abs(
                self.X[:, k:].reshape(N, k, 3).mean((0, 2))[:, None]
                + torch.zeros((k, 3))
            )
        ).reshape(k * 3)

        Ymean = 0 * self.Y.mean(0)
        Ystd = torch.abs(self.Y - Ymean).mean((0, 1)) + torch.zeros_like(Ymean)

        self.stats = [0.0, 1.0, 0.0, 1.0]

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def default_aug(self, model):
        return GroupAugmentation(model, self.repin, self.repout, self.symmetry)
