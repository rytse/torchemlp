import torch
import torch.nn as nn
from torch.autograd import grad, Function


class HamiltonianDynamicsFunction(Function):
    @staticmethod
    def forward(ctx, H, t, z):
        d = z.shape[-1] // 2

        J = torch.zeros((2 * d, 2 * d), device=z.device)
        J[:d, d:] = torch.eye(d)
        J[d:, :d] = -torch.eye(d)

        try:
            with torch.enable_grad():
                t.requires_grad_()
                z.requires_grad_()
                H_val = H(t, z)
                (dHdz,) = grad(H_val.sum(), z, create_graph=True)
        except Exception as e:
            breakpoint()
            raise e

        dynamics = torch.matmul(dHdz.unsqueeze(0), J.t()).squeeze(0)

        # Save variables required for the backward pass
        ctx.save_for_backward(t, z, J, dHdz)
        ctx.H = H
        return dynamics

    @staticmethod
    def backward(ctx, grad_output):
        t, z, J, dHdz = ctx.saved_tensors
        H = ctx.H

        with torch.enable_grad():
            t = t.clone().requires_grad_()
            z = z.clone().requires_grad_()
            H_val = H(t, z)
            H_val_sum = H_val.sum()
            grad_output_J = grad_output @ J
            grad_z, grad_t = grad(
                H_val_sum,
                (z, t),
                grad_outputs=(grad_output_J, None),
                create_graph=True,
                allow_unused=True,
            )

        grad_params = grad(
            H_val_sum, H.parameters(), grad_outputs=dHdz, retain_graph=True
        )

        return (None, grad_t, grad_z, *grad_params)


class Hamiltonian(nn.Module):
    def __init__(self, H: nn.Module):
        super().__init__()
        self.H = H

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return HamiltonianDynamicsFunction.apply(self.H, t, z)


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

    z.requires_grad_()
    t.requires_grad_()

    J = torch.zeros((2 * d, 2 * d)).to(z)
    J[:d, d:] = torch.eye(d)
    J[d:, :d] = -torch.eye(d)
    J.requires_grad_()

    with torch.enable_grad():
        H_val = H(t, z)

    dHdz = grad(
        H_val,
        z,
        grad_outputs=torch.ones_like(H_val),
        create_graph=True,
        retain_graph=True,
    )[0]

    return torch.matmul(dHdz, J.t())
