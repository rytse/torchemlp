from typing import Callable

import torch
import torch.nn as nn
from torch.autograd import grad
import torch.autograd.functional as F

from torchemlp.nn.equivariant import EMLP


class Hamiltonian(nn.Module):
    def __init__(self, H: nn.Module):
        super().__init__()
        self.H = H

    def forward(self, t: torch.Tensor, z: torch.Tensor):
        return hamiltonian_dynamics(self.H, z, t)


def hamiltonian_dynamics(
    H: nn.Module,
    z: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Given a nn.Module representing the Hamiltonian of a system, compute the
    dynamics function z' = J∇H(z)
    """
    d = z.shape[-1] // 2
    J = torch.zeros((2 * d, 2 * d)).to(z)
    J[:d, d:] = torch.eye(d)
    J[d:, :d] = -torch.eye(d)

    z_in = z.clone().detach().requires_grad_(True)
    t_in = t.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        H_val = H(t_in, z_in)

    dHdz = grad(H_val, z_in, grad_outputs=torch.ones_like(H_val), create_graph=True)[0]

    return torch.einsum("ij,bj->bi", J, dHdz)
