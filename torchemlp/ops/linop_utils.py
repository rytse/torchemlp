from functools import reduce
from typing import Any

import torch

from .linop_base import LinearOperator, Lazy


def product(L: Any) -> Any:
    return reduce(lambda a, b: a * b, L)


def lazify(op: LinearOperator | torch.Tensor) -> LinearOperator:
    match op:
        case torch.Tensor():
            return Lazy(op)
        case _:
            return op


def densify(op: LinearOperator | torch.Tensor) -> torch.Tensor:
    match op:
        case torch.Tensor():
            return op
        case _:
            return op.dense()


def kronsum(A_dense: torch.Tensor, B_dense: torch.Tensor) -> torch.Tensor:
    return torch.kron(
        A_dense,
        torch.eye(B_dense.shape[-1], dtype=B_dense.dtype, device=B_dense.device),
    ) + torch.kron(
        torch.eye(A_dense.shape[-1], dtype=A_dense.dtype, device=A_dense.device),
        B_dense,
    )


def lazy_direct_matmat(
    v: torch.Tensor, Ms: list[LinearOperator], mults: list | torch.Tensor
):
    k = v.shape[1] if len(v.shape) > 1 else 1

    i = 0
    y = []
    for M, mult in zip(Ms, mults):
        i_end = i + mult * M.shape[-1]
        v_slice = v[i:i_end]

        if v_slice.ndim == 2:
            v_slice = v_slice.T

        v_slice = v_slice.reshape(k * int(mult), M.shape[-1]).T
        elems = M @ v_slice

        y.append(elems.T.reshape(k, int(mult) * M.shape[0]).T)
        i = i_end

    return torch.cat(y, dim=0)
