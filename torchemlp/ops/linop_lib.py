from functools import reduce
from typing import Callable

import torch
import torch.autograd.functional as F

from .linop_base import LinearOperator, InvertibleLinearOperator
from .linop_utils import product, kronsum, lazy_direct_matmat


class LazyKron(InvertibleLinearOperator):
    """
    A lazy implementation of the Kronecker product of many linear operators.
    """

    def __init__(self, Ms: list[LinearOperator]):
        shape = product([Mi.shape[0] for Mi in Ms]), product([Mi.shape[1] for Mi in Ms])
        super().__init__(Ms[0].dtype, shape)
        self.Ms = Ms

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.matmat(v).reshape(-1)

    # TODO there's gotta be a cleaner way to do this ...
    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        ev = B.reshape(*[Mi.shape[-1] for Mi in self.Ms], -1)
        for i, M in enumerate(self.Ms):
            ev_front = torch.moveaxis(ev, i, 0)
            Mev_front = M @ ev_front.reshape(M.dense().shape[-1], -1)
            match Mev_front:
                case torch.Tensor():
                    Mev_front_dense = Mev_front
                case LinearOperator():
                    Mev_front_dense = Mev_front.dense()
            ev = torch.moveaxis(
                Mev_front_dense.reshape(M.dense().shape[0], *ev_front.shape[1:]), 0, i
            )
        return ev.reshape(self.shape[0], ev.shape[-1])

    def adjoint(self) -> LinearOperator:
        return LazyKron([Mi.H for Mi in self.Ms])

    def invT(self) -> InvertibleLinearOperator:
        invs = []
        for M in self.Ms:
            if isinstance(M, InvertibleLinearOperator):
                invs.append(M.invT())
            else:
                raise NotImplementedError(
                    "Cannot invert non-invertible linear operator."
                )
        return LazyKron(invs)

    def dense(self) -> torch.Tensor:
        Ms_dense = [M.dense() if isinstance(M, LinearOperator) else M for M in self.Ms]
        return reduce(torch.kron, Ms_dense)  # reducing via kronecker product


class LazyKronsum(LinearOperator):
    """
    A lazy implementation of the Kronecker sum of many linear operators.
    """

    def __init__(self, Ms: list[LinearOperator]):
        shape = product([Mi.shape[0] for Mi in Ms]), product([Mi.shape[1] for Mi in Ms])
        super().__init__(Ms[0].dtype, shape)
        self.Ms = Ms

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.matmat(v).reshape(-1)

    # TODO there's gotta be a cleaner way to do this ...
    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        ev = B.reshape(*[Mi.shape[-1] for Mi in self.Ms], -1)
        out = 0 * ev
        for i, M in enumerate(self.Ms):
            ev_front = torch.moveaxis(ev, i, 0)
            Mev_front = M @ ev_front.reshape(M.shape[-1], -1)
            match Mev_front:
                case torch.Tensor():
                    Mev_front_dense = Mev_front
                case LinearOperator():
                    Mev_front_dense = Mev_front.dense()
            out += torch.moveaxis(
                Mev_front_dense.reshape(M.shape[0], *ev_front.shape[1:]), 0, i
            )
        return out.reshape(self.shape[0], out.shape[-1])

    def adjoint(self) -> LinearOperator:
        return LazyKronsum([Mi.H for Mi in self.Ms])

    def dense(self) -> torch.Tensor:
        Ms_dense = [M.dense() if isinstance(M, LinearOperator) else M for M in self.Ms]
        return reduce(kronsum, Ms_dense)  # reducing via kronecker sum


class LazyJVP(LinearOperator):
    """
    Lazy implementation of a linear operator that performs the jacobian-vector
    product of the input linear operator.
    """

    def __init__(self, op_fn: Callable, X: torch.Tensor, TX: torch.Tensor):
        """ "
        Args:
        ----------
            op_fn:  torch function whose Jacobian defines this operation

            X:      the point in op_fn's domain manifold at which to compute
                    the JVP (i.e. the x in J(x) @ v)

            TX:     the direction in op_fn's domain manifold's tangent space in
                    which to compute the JVP (i.e. the v in J(x) @ v)
        """
        shape = op_fn(X).shape
        self.vjp = lambda v: torch.cat(F.jvp(lambda x: op_fn(x) @ v, X, TX)[1], dim=0)
        self.vjp_H = lambda v: torch.cat(
            F.jvp(lambda x: op_fn(x).H @ v, X, TX)[1], dim=0
        )
        self.vjp_T = lambda v: torch.cat(
            F.jvp(lambda x: op_fn(x).T @ v, X, TX)[1], dim=0
        )
        super().__init__(op_fn(X).dtype, shape)

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.vjp(v)

    def rmatvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.vjp_H(v)

    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        return self.vjp(B)

    def rmatmat(self, B: torch.Tensor) -> torch.Tensor:
        return self.vjp_H(B)


class LazyConcat(LinearOperator):
    """
    Lazy implementation of a linear operator that concatenates its input
    linear operators.
    """

    def __init__(self, Ms: list[LinearOperator]):
        if not all(M.shape[0] == Ms[0].shape[0] for M in Ms):
            raise ValueError("All input operators must have the same shape")
        shape = (sum(M.shape[0] for M in Ms), Ms[0].shape[1])
        self.Ms = Ms

        super().__init__(Ms[0].dtype, shape)

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        Mvs_ambig = [M @ v for M in self.Ms]
        Mvs_tensor = []
        for Mv in Mvs_ambig:
            match Mv:
                case LinearOperator():
                    Mvs_tensor.append(Mv.dense())
                case torch.Tensor():
                    Mvs_tensor.append(Mv)
        return torch.cat(Mvs_tensor, dim=0)

    def rmatvec(self, v: torch.Tensor) -> torch.Tensor:
        return torch.cat([M.H @ v for M in self.Ms], dim=0)

    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        MBs_ambig = [M @ B for M in self.Ms]
        MBs_tensor = []
        for MB in MBs_ambig:
            match MB:
                case LinearOperator():
                    MBs_tensor.append(MB.dense())
                case torch.Tensor():
                    MBs_tensor.append(MB)
        return torch.cat(MBs_tensor, dim=0)

    def rmatmat(self, B: torch.Tensor) -> torch.Tensor:
        Bs = torch.split(B, len(self.Ms))
        MHBs = [self.Ms[i].H @ Bs[i] for i in range(len(self.Ms))]
        return reduce(lambda x, y: x + y, MHBs)

    def dense(self) -> torch.Tensor:
        Ms_dense = [M.dense() if isinstance(M, LinearOperator) else M for M in self.Ms]
        return torch.cat(Ms_dense, dim=0)


class LazyDirectSum(InvertibleLinearOperator):
    """
    Lazy implementation of a linear operator that is the direct sum of its
    input linear operators.
    """

    def __init__(self, Ms: list[LinearOperator], mults: list[int] = []):
        self.Ms = Ms
        self.mults = [1 for _ in self.Ms] if mults is None else mults
        dim = sum(Mi.shape[0] * m for Mi, m in zip(self.Ms, self.mults))
        shape = (dim, dim)
        super().__init__(Ms[0].dtype, shape)

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return lazy_direct_matmat(v, self.Ms, self.mults)

    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        return lazy_direct_matmat(B, self.Ms, self.mults)

    def adjoint(self) -> LinearOperator:
        return LazyDirectSum([Mi.H for Mi in self.Ms], self.mults)

    def invT(self) -> InvertibleLinearOperator:
        invs = []
        for M in self.Ms:
            if isinstance(M, InvertibleLinearOperator):
                invs.append(M.invT())
            else:
                raise ValueError("Cannot invert non-invertible linear operator")
        return LazyDirectSum(invs, self.mults)

    def dense(self) -> torch.Tensor:
        Ms_all = [M for M, c in zip(self.Ms, self.mults) for _ in range(c)]
        Ms_dense = [
            Mi.dense() if isinstance(Mi, LinearOperator) else Mi for Mi in Ms_all
        ]
        return torch.block_diag(*Ms_dense)


class LazyPerm(InvertibleLinearOperator):
    """
    Lazy implementation of a linear operator that is the permutation of its
    input linear operators.
    """

    def __init__(self, perm: torch.Tensor):
        super().__init__(None, (len(perm), len(perm)))
        self.perm = perm

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return v[self.perm]

    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        return B[self.perm]

    def adjoint(self) -> LinearOperator:
        return LazyPerm(torch.argsort(self.perm))

    def invT(self) -> InvertibleLinearOperator:
        return self


class LazyShift(InvertibleLinearOperator):
    """
    Lazy implementation of a linear operator that is the shift of its input's
    elements or rows.
    """

    def __init__(self, n: int, k: int = 1):
        super().__init__(None, (n, n))
        self.n = n
        self.k = k

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return torch.roll(v, self.k, dims=0)

    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        return torch.roll(B, self.k, dims=0)

    def adjoint(self) -> LinearOperator:
        return LazyShift(self.n, -self.k)

    def invT(self) -> InvertibleLinearOperator:
        return self


class SwapMatrix(InvertibleLinearOperator):
    """
    Lazy implementation of a linear operator that is the swap of its input's
    elements or rows.
    """

    def __init__(self, swaps: torch.Tensor | list[int], n: int):
        super().__init__(None, (n, n))
        self.swaps = swaps
        self.n = n

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.matmat(v)

    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        out = torch.tensor(B)
        out[self.swaps[::-1]] = B[self.swaps]
        return out

    def adjoint(self) -> LinearOperator:
        return self

    def invT(self) -> InvertibleLinearOperator:
        return self


class Rot90(InvertibleLinearOperator):
    """
    Lazy implementation of a linear operator that performs a given number of
    quarter turns on its input vector(s).
    """

    def __init__(self, n: int, k: int):
        super().__init__(None, (n * n, n * n))
        self.n = n
        self.k = k

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return torch.rot90(v.reshape((self.n, self.n, -1)), k=self.k).reshape(v.shape)

    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        return torch.rot90(B.reshape((self.n, self.n, -1)), k=self.k).reshape(B.shape)

    def invT(self) -> InvertibleLinearOperator:
        return self
