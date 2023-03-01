import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import torch


def merge_torch_types(dtype1, dtype2):
    return (torch.ones(1, dtype=dtype1) * torch.ones(1, dtype=dtype2)).dtype


def vis_basis(basis, shape, cluster=True):
    Q = basis @ np.eye(basis.shape[-1])  # convert to a dense matrix if necessary
    v = np.random.randn(Q.shape[0])  # sample random vector
    v = Q @ (Q.T @ v)  # project onto equivariant subspace
    if cluster:  # cluster nearby values for better color separation in plot
        v = KMeans(n_clusters=Q.shape[-1]).fit(v.reshape(-1, 1)).labels_
    if v is not None:
        plt.imshow(v.reshape(shape))
        plt.axis("off")
    else:
        print("Q @ (Q.T @ v) failed")


def vis(repin, repout, cluster=True):
    Q = (repin >> repout).equivariant_basis().dense  # compute the equivariant basis
    vis_basis(Q, (repout.size, repin.size), cluster)  # visualize it
