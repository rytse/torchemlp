from functools import reduce

import torch

from linop_base import LinearOperator
from linop_utils import product, kronsum, lazy_direct_matmat


class LazyKron(LinearOperator):
    """
    A lazy implementation of the Kronecker product of two linear operators.
    """

    def __init__(self, Ms):
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
            Mev_front = (M @ ev_front.reshape(M.shape[-1], -1)).reshape(
                M.shape[0], *ev_front.shape[1:]
            )
            ev = torch.moveaxis(Mev_front, 0, i)
        return ev.reshape(self.shape[0], ev.shape[-1])

    def adjoint(self) -> LinearOperator:
        return LazyKron([Mi.H for Mi in self.Ms])

    def invT(self) -> LinearOperator:
        return LazyKron([Mi.invT() for Mi in self.Ms])

    def dense(self) -> torch.Tensor:
        Ms_dense = [M.dense if isinstance(M, LinearOperator) else M for M in self.Ms]
        return reduce(torch.kron, Ms_dense)  # reducing via kronecker product


class LazyKronsum(LinearOperator):
    """
    A lazy implementation of the Kronecker sum of two linear operators.
    """

    def __init__(self, Ms):
        shape = product([Mi.shape[0] for Mi in Ms]), product([Mi.shape[1] for Mi in Ms])
        super().__init__(Ms[0].dtype, shape)
        self.Ms = Ms

    def matvec(self, v):
        return self.matmat(v).reshape(-1)

    # TODO there's gotta be a cleaner way to do this ...
    def matmat(self, B):
        ev = B.reshape(*[Mi.shape[-1] for Mi in self.Ms], -1)
        out = 0 * ev
        for i, M in enumerate(self.Ms):
            ev_front = torch.moveaxis(ev, i, 0)
            Mev_front = (M @ ev_front.reshape(M.shape[-1], -1)).reshape(
                M.shape[0], *ev_front.shape[1:]
            )
            out += torch.moveaxis(Mev_front, 0, i)
        return out.reshape(self.shape[0], out.shape[-1])

    def adjoint(self):
        return LazyKronsum([Mi.H for Mi in self.Ms])

    def dense(self):
        Ms_dense = [M.dense if isinstance(M, LinearOperator) else M for M in self.Ms]
        return reduce(kronsum, Ms_dense)  # reducing via kronecker sum


class LazyJVP(LinearOperator):
    """
    Lazy implementation of a linear operator that performs the jacobian-vector
    product of the input linear operator.
    """

    def __init__(self, op_fn, X, TX):
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
        self.vjp = lambda v: F.jvp(lambda x: op_fn(x) @ v, X, TX)[1]
        self.vjp_H = lambda v: F.jvp(lambda x: op_fn(x).H @ v, X, TX)[1]
        self.vjp_T = lambda v: F.jvp(lambda x: op_fn(x).T @ v, X, TX)[1]
        super().__init__(op_fn(X).dtype, shape)

    def matvec(self, v):
        return self.vjp(v)

    def rmatvec(self, v):
        return self.vjp_H(v)

    def matmat(self, B):
        return self.vjp(B)

    def rmatmat(self, B):
        return self.vjp_H(B)


class LazyConcat(LinearOperator):
    """
    Lazy implementation of a linear operator that concatenates its input
    linear operators.
    """

    def __init__(self, Ms):
        if not all(M.shape[0] == Ms[0].shape[0] for M in Ms):
            raise ValueError("All input operators must have the same shape")
        shape = (sum(M.shape[0] for M in Ms), Ms[0].shape[1])
        super().__init__(Ms[0].dtype, shape)
        self.Ms = Ms

    def matvec(self, v):
        return torch.cat([M @ v for M in self.Ms], dim=0)

    def rmatvec(self, v):
        return torch.cat([M.H @ v for M in self.Ms], dim=0)

    def matmat(self, B):
        return torch.cat([M @ B for M in self.Ms], dim=0)

    def rmatmat(self, B):
        Bs = torch.split(B, len(self.Ms))
        return sum([Ms[i].H @ Bs[i] for i in len(self.Ms)])

    def dense(self):
        Ms_dense = [M.dense if isinstance(M, LinearOperator) else M for M in self.Ms]
        return torch.cat(Ms_dense, dim=0)


class LazyDirectSum(LinearOperator):
    """
    Lazy implementation of a linear operator that is the direct sum of its
    input linear operators.
    """

    def __init__(self, Ms, mults=None):
        self.Ms = Ms
        self.mults = [1 for M in self.Ms] if mults is None else mults
        dim = sum(Mi.shape[0] * m for Mi, m in zip(Ms, mults))
        shape = (dim, dim)
        super().__init__(Ms[0].dtype, shape)

    def matvec(self, v):
        return lazy_direct_matmat(v, self.Ms, self.mults)

    def matmat(self, B):
        return lazy_direct_matmat(B, self.Ms, self.mults)

    def adjoint(self):
        return LazyDirectSum([Mi.H for Mi in self.Ms], self.mults)

    def invT(self):
        return LazyDirectSum([Mi.invT() for Mi in self.Ms], self.mults)

    def dense(self):
        Ms_all = [M for M, c in zip(self.Ms, self.mults) for _ in range(c)]
        Ms_dense = [Mi.dense if isinstance(Mi, LinearOperator) else Mi for Mi in Ms_all]
        return torch.block_diag(*Ms_dense)


class LazyPerm(LinearOperator):
    """
    Lazy implementation of a linear operator that is the permutation of its
    input linear operators.
    """

    def __init__(self, perm):
        super().__init__(None, (len(perm), len(perm)))
        self.perm = perm

    def matvec(self, v):
        return v[self.perm]

    def matmat(self, B):
        return B[self.perm]

    def adjoint(self):
        return LazyPerm(torch.argsort(self.perm))

    def invT(self):
        return self


class LazyShift(LinearOperator):
    """
    Lazy implementation of a linear operator that is the shift of its input's
    elements or rows.
    """

    def __init__(self, n, k=1):
        super().__init__(None, (n, n))
        self.n = n
        self.k = k

    def matvec(self, v):
        return torch.roll(v, self.k, axis=0)

    def matmat(self, B):
        return torch.roll(B, self.k, axis=0)

    def adjoint(self):
        return LazyShift(self.n, -self.k)

    def invT(self):
        return self


class SwapMatrix(LinearOperator):
    """
    Lazy implementation of a linear operator that is the swap of its input's
    elements or rows.
    """

    def __init__(self, swaps, n):
        super().__init__(None, (n, n))
        self.swaps = swaps
        self.n = n

    def matvec(self, v):
        return self.matmat(v)

    def matmat(self, B):
        out = torch.tensor(B)
        out[self.swaps[::-1]] = B[self.swaps]
        return out

    def adjoint(self):
        return self

    def invT(self):
        return self


class Rot90(LinearOperator):
    """
    Lazy implementation of a linear operator that performs a given number of
    quarter turns on its input vector(s).
    """

    def __init__(self, n, k):
        super().__init__(None, (n * n, n * n))
        self.n = n
        self.k = k

    def matvec(self, v):
        return torch.rot90(v.reshape((self.n, self.n, -1)), k=self.k).reshape(v.shape)

    def matmat(self, B):
        return torch.rot90(B.reshape((self.n, self.n, -1)), k=self.k).reshape(B.shape)

    def invT(self):
        return self
