from functools import reduce

import torch

from linop_base import LinearOperator, Lazy


def product(L):
    return reduce(lambda a, b: a * b, L)


def lazify(op):
    if isinstance(op, LinearOperator):
        return op
    elif isinstance(op, torch.Tensor):
        return Lazy(op)
    else:
        return NotImplemented


def densify(op):
    if isinstance(op, LinearOperator):
        return op.dense
    elif isinstance(op, torch.Tensor):
        return op
    else:
        return NotImplemented


def kronsum(A, B):
    return torch.kron(A, torch.eye(B.shape[-1])) + torch.kron(torch.eye(A.shape[-1]), B)


def lazy_direct_matmat(v, Ms, mults):
    n = v.shape[0]
    k = v.shape[1] if len(v.shape) > 1 else 1

    i = 0
    y = []
    for M, multiplicity in zip(Ms, mults):
        i_end = i + multiplicity * M.shape[-1]
        elems = M @ v[i:i_end].T.reshape(k * multiplicity, M.shape[-1]).T
        y.append(elems.T.reshape(k, multiplicity * M.shape[0]).T)
        i = i_end

    return torch.cat(y, dim=0)
