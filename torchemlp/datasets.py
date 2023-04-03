from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import functorch

from torchemlp.groups import Group, O, SO
from torchemlp.reps import Rep, Vector, Scalar, T
from torchemlp.utils import DEFAULT_DEVICE


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


class Dataset(ABC):
    """
    Abstract class for a dataset.
    """

    def __init__(
        self,
        dim: int,
        G: Group,
        repin: Rep,
        repout: Rep,
        X: torch.Tensor,
        Y: torch.Tensor,
        stats: list[float],
    ):
        self.G = G
        self.repin = repin
        self.repout = repout
        self.dim = dim
        self.X = X
        self.Y = Y
        self.stats = stats

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def default_aug(self, model):
        return GroupAugmentation(model, self.repin, self.repout, self.G)


class Inertia(Dataset):
    """
    Euler equations for inertial rotations

    J = Sum_i m_i(x_i^T x_i I - x_i x_i^T)

    O(3)-invariant.
    """

    def __init__(self, N=1024, k=5, device: torch.device = DEFAULT_DEVICE):
        dim = (1 + 3) * k

        repin = k * Scalar + k * Vector
        repout = T(2)
        G = O(3)

        X = torch.randn(N, dim, device=device)
        X[:, :k] = torch.log(1 + torch.exp(X[:, :k]))  # masses
        mi = X[:, :k]
        ri = X[:, k:].reshape(-1, k, 3)

        I = torch.eye(3, device=device)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (
            mi[:, :, None, None] * (r2 * I - ri[..., None] * ri[..., None, :])
        ).sum(1)
        Y = inertia.reshape(-1, 9)

        # Xmean = self.X.mean(0)
        # Xmean[k:] = 0.0
        # Xstd = torch.zeros_like(Xmean, dtype=Xmean.dtype, device=device)
        # Xstd[:k] = torch.abs(self.X[:, :k]).mean(0)
        # Xstd[k:] = (
        # torch.abs(
        # self.X[:, k:].reshape(N, k, 3).mean((0, 2))[:, None]
        # + torch.zeros((k, 3), dtype=Xstd.dtype, device=device)
        # )
        # ).reshape(k * 3)

        # Ymean = 0 * self.Y.mean(0)
        # Ystd = torch.abs(self.Y - Ymean).mean((0, 1)) + torch.zeros_like(
        # Ymean, dtype=Ymean.dtype, device=device
        # )

        stats = [0.0, 1.0, 0.0, 1.0]

        super().__init__(dim, G, repin, repout, X, Y, stats)


class O5Synthetic(Dataset):
    """
    Dataset representing trajectories of

    f(x_1, x_2) = sin(||x_1||) - 1/2||x_2||^3 + 〈x_1, x_2〉/ (||x_1|| ||x_2||)

    O(5)-invariant.
    """

    def __init__(self, N=1024, device: torch.device = DEFAULT_DEVICE):
        d = 5
        dim = 2 * d

        repin = 2 * Vector
        repout = Scalar
        G = O(d)

        X = torch.randn(N, dim, device=device)
        ri = X.reshape(-1, 2, d)
        r1 = ri[:, 0, :]
        r2 = ri[:, 1, :]
        r1n = torch.norm(r1, dim=1)
        r2n = torch.norm(r2, dim=1)
        Y = (
            torch.sin(r1n)
            - 0.5 * torch.pow(r2n, 3)
            + torch.sum(r1 * r2, dim=1) / r1n / r2n
        )
        Y = Y[..., None]

        Xmean = X.mean(0)
        Xscale = (
            torch.sqrt((X.reshape(N, 2, d) ** 2).mean((0, 2)))[:, None] + 0 * ri[0]
        ).reshape(dim)

        stats = [Xmean, Xscale, Y.mean(dim=0), Y.std(dim=0)]

        super().__init__(dim, G, repin, repout, X, Y, stats)


class Radius(Dataset):
    def __init__(self, N=1024, device: torch.device = DEFAULT_DEVICE):
        d = 3

        repin = Vector
        repout = Scalar
        G = SO(d)

        X = 100 * torch.randn(N, d, device=device)
        Y = torch.norm(X, dim=1)
        Y = Y[..., None]

        stats = [X.mean(0), X.std(dim=0), Y.mean(dim=0), Y.std(dim=0)]

        super().__init__(d, G, repin, repout, X, Y, stats)
