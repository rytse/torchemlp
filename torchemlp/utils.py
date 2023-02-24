import torch


def is_scalar(x):
    return (
        (isinstance(x, torch.Tensor) and x.dim() == 0)
        or isinstance(x, float)
        or isinstance(x, int)
    )


def is_vector(x):
    return isinstance(x, torch.Tensor) and len(x.shape) == 1


def is_matrix(x):
    return isinstance(x, torch.Tensor) and len(x.shape) == 2
