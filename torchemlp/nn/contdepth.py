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
        zp = hamiltonian_dynamics(self.H, z, t)
        breakpoint()
        return zp


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
    J = J.requires_grad_(True)

    with torch.enable_grad():
        H_val = H(t, z)

    dHdz = grad(
        H_val,
        z,
        grad_outputs=torch.ones_like(H_val),
        create_graph=True,
        retain_graph=True,
    )[0]

    # zp = torch.cat((-dHdz[:, d:], dHdz[:, :d]), dim=1)
    # zp = torch.matmul(dHdz.unsqueeze(1), J.T).squeeze(1)

    zp = torch.matmul(dHdz, J.t())

    # if z.shape[0] == 500:
    # breakpoint()

    return zp


def hamiltonian_dynamics_nograd(H: nn.Module, z: torch.Tensor, t: torch.Tensor):
    print("hamiltonian_dynamics_nograd")
    z = z.requires_grad_(True)
    t = t.requires_grad_(True)
    return hamiltonian_dynamics(H, z, t).detach()
