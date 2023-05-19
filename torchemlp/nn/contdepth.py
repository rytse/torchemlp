from typing import Callable

import torch
import torch.autograd.functional as F

from torchemlp.nn.equivariant import EMLP


class EMLPH(EMLP):
    def H(self, x: torch.Tensor):
        return torch.sum(self.network(x))

    def __call__(self, x: torch.Tensor):
        return self.H(x)


def hamiltonian_dynamics(
    H: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    t: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Given a nn.Module representing the Hamiltonian of a system, compute the
    dynamics function z' = J∇H(z)
    """

    d = z.shape[-1] // 2
    J = torch.zeros((2 * d, 2 * d), device=device)
    J[:d, d:] = torch.eye(d, device=device)
    J[d:, :d] = -torch.eye(d, device=device)

    z_in = z.clone().detach().requires_grad_(True)
    t_in = t.clone().detach().requires_grad_(True)
    H_val = H(z_in, t_in)
    H_val.backward()
    H_grad = z_in.grad

    return torch.mm(H_grad, J.t())
