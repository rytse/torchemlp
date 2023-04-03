import matplotlib.pyplot as plt

import torch


DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def merge_torch_types(dtype1, dtype2, device: torch.device = DEFAULT_DEVICE):
    return (
        torch.ones(1, dtype=dtype1, device=device)
        * torch.ones(1, dtype=dtype2, device=device)
    ).dtype


def rel_rms_diff(
    A: torch.Tensor, B: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Relative root-mean-squared difference between two tensors.
    """
    mad = torch.mean(torch.abs(A - B))  # mean abs diff
    ama = torch.mean(torch.abs(A))  # a mean abs
    bma = torch.mean(torch.abs(B))  # b mean abs
    return mad / (ama + bma + epsilon)


def vis_basis(basis, shape):
    device = basis.device

    Q = basis @ torch.eye(
        basis.shape[-1], device=device
    )  # convert to a dense matrix if necessary

    v = torch.randn(Q.shape[0], device=device)  # sample random vector
    v = Q @ (Q.T @ v)  # project onto equivariant subspace

    if v is not None:
        plt.imshow(torch.abs(v).detach().cpu().numpy().reshape(shape))
        plt.axis("off")
    else:
        print("Q @ (Q.T @ v) failed")


def vis(repin, repout):
    Q = (
        (repin >> repout).equivariant_basis().dense(device=repin.device)
    )  # compute the equivariant basis
    vis_basis(Q, (repout.size, repin.size))  # visualize it


def lambertW(ch: int, d: int) -> int:
    """
    Solve x * d^x = ch rounded down to the nearest integer.
    """
    max_rank = 0
    while (max_rank + 1) * d**max_rank <= ch:
        max_rank += 1
    return max_rank - 1


def binom(n, k):
    if not 0 <= k <= n:
        return 0

    b = 1
    for t in range(min(k, n - k)):
        b *= n
        b /= t + 1
        n -= 1

    return int(b)
