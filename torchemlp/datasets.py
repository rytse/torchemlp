from abc import ABC
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from functorch import vmap  # type: ignore
from torch.utils.data import Dataset

from torchemlp.groups import SO, Group, O
from torchemlp.reps import Rep, Scalar, T, Vector
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

        self.rho_in = vmap(self.repin.rho)
        self.rho_out = vmap(self.repout.rho)

    def forward(self, x, training=True):
        if training:
            gs = self.G.samples(x.shape[0])
            rhout_inv = torch.linalg.inv(self.rho_out(gs))
            model_in = (self.rho_in(gs) @ x[..., None])[..., None]
            model_out = self.network(model_in, training)
            return (rhout_inv @ model_out)[..., 0]
        return self.network(x, training)


class SymDataset(Dataset, ABC):
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


class Inertia(SymDataset):
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

        stats = [0.0, 1.0, 0.0, 1.0]

        super().__init__(dim, G, repin, repout, X, Y, stats)


class O5Synthetic(SymDataset):
    """
    Dataset representing trajectories of

    f(x_1, x_2) = sin(||x_1||) - 1/2||x_2||^3 + 〈x_1, x_2〉/ (||x_1|| ||x_2||)

    O(5)-invariant
    """

    def __init__(self, N=1024, device: torch.device = DEFAULT_DEVICE):
        d = 5
        dim = 2 * d

        repin = 2 * Vector
        repout = Scalar
        G = O(d)

        X, Y = self.generate_data(N, device)

        stats = [X.mean(dim=0), X.std(dim=0), Y.mean(dim=0), Y.std(dim=0)]

        super().__init__(dim, G, repin, repout, X, Y, stats)

    def f(self, x1: npt.NDArray, x2: npt.NDArray) -> npt.NDArray:
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return (
            np.sin(norm_x1)
            - 0.5 * (norm_x2**3)
            + np.dot(x1, x2) / (norm_x1 * norm_x2)
        )

    def generate_data(
        self, num_samples: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_np = np.random.rand(num_samples, 10)
        Y_np = np.array([self.f(sample[:5], sample[5:]) for sample in X_np])

        X = torch.tensor(X_np, dtype=torch.float32).to(device)
        Y = torch.tensor(Y_np, dtype=torch.float32).to(device)

        return X, Y


class Radius(SymDataset):
    """
    Dataset representing the radius of a 3D vector.

    SO(3)-invariant
    """

    def __init__(self, N=1024, device: torch.device = DEFAULT_DEVICE):
        d = 3

        repin = Vector
        repout = Scalar
        G = SO(d)

        X, Y = self.generate_data(N, device)

        stats = [X.mean(dim=0), X.std(dim=0), Y.mean(dim=0), Y.std(dim=0)]

        super().__init__(d, G, repin, repout, X, Y, stats)

    def generate_data(
        self, num_samples: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X_np = np.random.uniform(-5, 5, (num_samples, 3))
        Y_np = np.sqrt(np.sum(np.square(X_np), axis=1)).reshape(-1, 1)

        X_torch = torch.tensor(X_np, dtype=torch.float32).to(device)
        Y_torch = torch.tensor(Y_np, dtype=torch.float32).to(device)

        return X_torch, Y_torch
