from functools import reduce
from typing import Union, List, Any

import torch

from .linop_base import LinearOperator, Lazy


def product(L: Any) -> Any:
    return reduce(lambda a, b: a * b, L)


def lazify(op: Union[LinearOperator, torch.Tensor]) -> LinearOperator:
    match op:
        case LinearOperator():
            return op
        case torch.Tensor():
            return Lazy(op)
    return NotImplemented


def densify(op: Union[LinearOperator, torch.Tensor]) -> torch.Tensor:
    match op:
        case LinearOperator():
            return op.dense
        case torch.Tensor():
            return op
    return NotImplemented


def kronsum(A_dense: torch.Tensor, B_dense: torch.Tensor) -> torch.Tensor:
    return torch.kron(A_dense, torch.eye(B_dense.shape[-1])) + torch.kron(
        torch.eye(A_dense.shape[-1]), B_dense
    )


def lazy_direct_matmat(
    v: torch.Tensor, Ms: List[LinearOperator], mults: Union[List, torch.Tensor]
):
    k = v.shape[1] if len(v.shape) > 1 else 1

    i = 0
    y = []
    for M, multiplicity in zip(Ms, mults):
        i_end = i + multiplicity * M.shape[-1]
        elems = M @ v[i:i_end].T.reshape(k * int(multiplicity), M.shape[-1]).T
        y.append(elems.T.reshape(k, int(multiplicity) * M.shape[0]).T)
        i = i_end

    return torch.cat(y, dim=0)
