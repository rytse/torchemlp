import numpy as np
import matplotlib.pyplot as plt
import scipy.special

import torch


def merge_torch_types(dtype1, dtype2):
    return (torch.ones(1, dtype=dtype1) * torch.ones(1, dtype=dtype2)).dtype


def vis_basis(basis, shape, cluster=True):
    Q = basis @ np.eye(basis.shape[-1])  # convert to a dense matrix if necessary
    v = np.random.randn(Q.shape[0])  # sample random vector
    v = Q @ (Q.T @ v)  # project onto equivariant subspace
    if v is not None:
        plt.imshow(v.reshape(shape))
        plt.axis("off")
    else:
        print("Q @ (Q.T @ v) failed")


def vis(repin, repout, cluster=True):
    Q = (repin >> repout).equivariant_basis().dense  # compute the equivariant basis
    vis_basis(Q, (repout.size, repin.size), cluster)  # visualize it


def lambertW(ch, d) -> int:
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
