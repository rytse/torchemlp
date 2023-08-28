from abc import ABC, abstractmethod
from typing import Any, Callable, Union

import torch

import torchemlp.ops
import torchemlp.utils
from torchemlp.utils import DEFAULT_DEVICE


class LinearOperator(ABC):
    def __init__(self, dtype, shape: tuple):
        self.dtype = dtype
        self.shape = shape

    @abstractmethod
    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform y = Ax where A is the linear operator. Overload this with a
        fast custom implementation given the structure of the operator.
        """
        pass

    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        """
        Perform Y = A B where A is self's linear operator and B is another
        linear operator (or matrix). This will call a vectorized version of
        matvec unless you overload it with something faster.
        """
        return torch.hstack([self.matvec(col.reshape(-1, 1)) for col in B.T])

    def adjoint(self) -> "LinearOperator":
        return _AdjointLinearOperator(self)

    H = property(adjoint)

    def transpose(self) -> "LinearOperator":
        return _TransposedLinearOperator(self)

    T = property(transpose)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform adjoint y = A^H x where A is the linear operator. Creates a new
        instance of _AdjointLinearOperator(self) every time this is called.

        If you need performance, overload this with a custom implemntation that
        doesn't recompute self.H every time.
        """
        if type(self).adjoint == LinearOperator.adjoint:
            raise NotImplementedError  # infinite recursion
        return self.H.matvec(x)

    def rmatmat(self, B: torch.Tensor) -> torch.Tensor:
        """
        Perform Y = A^H B where A is self's linear operator and B is another
        linear operator (or matrix). This will call a vectorized version of
        rmatvec unless you overload it with something faster.
        """
        if type(self).adjoint == LinearOperator.adjoint:
            return torch.hstack([self.rmatvec(col.reshape(-1, 1)) for col in B.T])
        return self.H.matmat(B)

    def __mul__(
        self, x: Union["LinearOperator", torch.Tensor, int, float]
    ) -> Union["LinearOperator", torch.Tensor]:
        """
        This python operator overloads the * operator and performs a different
        function depending on the type of x.

        If x is a LinearOperator:
            Construct a new linear operator via the product (composition) of
            the two operators in the group of linear operators on the
            self.shape dimensional vector space.

        If x is a scalar (float or singleton torch.Tensor):
            Construct a new linear operator via the linear scaling in the
            vector space of linear operators on the self.shape dimensional
            vector space.

        If x is a torch.Tensor representing a vector or a matrix:
            Apply the linear operator to x.

        Several other dunder methods are overloaded to perform the same task.
        The notation is mirrored in several places because in every case of
        self * x, we are performing a multiplication-like group operation,
        which has several different notations across different fields of math.
        """
        match x:
            case LinearOperator():
                return _ProductLinearOperator(self, x)
            case torch.Tensor():
                if x.ndim == 0:
                    return _ScaledLinearOperator(self, x)
                elif x.ndim == 1:
                    return self.matvec(x)
                elif x.ndim == 2:
                    return self.matmat(x)
                else:
                    raise NotImplementedError
            case int() | float():
                return _ScaledLinearOperator(self, x)

    def __call__(
        self, x: Union["LinearOperator", torch.Tensor, float]
    ) -> Union["LinearOperator", torch.Tensor]:
        return self * x

    def __matmul__(
        self, x: Union[torch.Tensor, "LinearOperator"]
    ) -> Union[torch.Tensor, "LinearOperator"]:
        return self * x

    def __rmul__(self, x: torch.Tensor | int | float) -> "LinearOperator":
        """
        This python operator overloads right-application (to handle
        non-commutative operators). This only works for LinearOperators and
        scalars, as right-application to a vector is not always defined by
        self.matvec. You can overload this function to cover this case!

        Several other dunder methods are overloaded to perform the same task.
        """
        match x:
            case torch.Tensor():
                if x.ndim == 0:
                    return _ScaledLinearOperator(self, x)
                else:
                    raise NotImplementedError
            case int() | float():
                return _ScaledLinearOperator(self, x)

    def __rmatmul__(
        self, x: Union[torch.Tensor, "LinearOperator"]
    ) -> Union[torch.Tensor, "LinearOperator"]:
        return x * self

    def __pow__(self, p: Union[int, torch.Tensor]):
        """
        This python operator overloads the ** operator for scalar p.
        """
        match p:
            case int():
                return _PowerLinearOperator(self, p)
            case torch.Tensor():
                if p.ndim == 0:
                    return _PowerLinearOperator(self, p)
                else:
                    raise NotImplementedError

    def __add__(self, x: Union["LinearOperator", torch.Tensor]) -> "LinearOperator":
        """
        This python operator overloads the + operator. It creates a new linear
        operator via addition in the vector space of linear operators. This
        method works on LinearOperators and torch matrices.

        Several other dunder methods are overloaded to perform the same task.
        """
        match x:
            case LinearOperator():
                return _SumLinearOperator(self, x)
            case torch.Tensor():
                if x.ndim == 2:
                    return _SumLinearOperator(self, Lazy(x))
                else:
                    raise NotImplementedError

    def __radd__(self, x: Union["LinearOperator", torch.Tensor]) -> "LinearOperator":
        return self.__add__(x)

    def __neg__(self) -> "LinearOperator":
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x: Union["LinearOperator", torch.Tensor]) -> "LinearOperator":
        return self.__add__(-x)

    def __repr__(self) -> str:
        M, N = self.shape
        if self.dtype is None:
            dt = "unspecified dtype"
        else:
            dt = f"dtype={self.dtype}"
        return f"<{M}x{N} {self.__class__.__name__} with {dt}>"

    def dense(self, device: torch.device = DEFAULT_DEVICE) -> torch.Tensor:
        """
        Convert the lazy implementation of the linear operator to a matrix
        representing the operator in R^n.
        """
        res = self @ torch.eye(self.shape[-1], device=device)
        match res:
            case LinearOperator():
                return res.dense()
            case torch.Tensor():
                return res


class InvertibleLinearOperator(LinearOperator, ABC):
    """
    Invertible linear operator.
    """

    @abstractmethod
    def invT(self) -> "InvertibleLinearOperator":
        """
        Inverse transpose of the linear operator.
        """
        pass


class _AdjointLinearOperator(LinearOperator):
    """
    Adjoint of a linear operator.
    """

    def __init__(self, A: LinearOperator):
        super(_AdjointLinearOperator, self).__init__(A.dtype, (A.shape[1], A.shape[0]))
        self.A = A
        self.args = (A,)

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.A.rmatvec(v)

    def rmatvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.A.matvec(v)

    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        return self.A.rmatmat(B)

    def rmatmat(self, B: torch.Tensor) -> torch.Tensor:
        return self.A.matmat(B)


class _TransposedLinearOperator(LinearOperator):
    """
    Transpose of a linear operator.
    """

    def __init__(self, A: LinearOperator):
        super(_TransposedLinearOperator, self).__init__(
            A.dtype, (A.shape[1], A.shape[0])
        )
        self.A = A
        self.args = (A,)

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.A.rmatvec(torch.conj(v)))

    def rmatvec(self, v: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.A.matvec(torch.conj(v)))

    def matmat(self, B: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.A.rmatmat(torch.conj(B)))

    def rmatmat(self, B: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.A.matmat(torch.conj(B)))


class _SumLinearOperator(InvertibleLinearOperator):
    """
    Sum of two linear operators.
    """

    def __init__(self, A: LinearOperator, B: LinearOperator):
        if A.shape != B.shape:
            raise ValueError("A and B must have the same shape")

        sum_dtype = torchemlp.utils.merge_torch_types(A.dtype, B.dtype)
        super(_SumLinearOperator, self).__init__(sum_dtype, A.shape)

        self.A = A
        self.B = B
        self.args = (A, B)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.matvec(x) + self.B.matvec(x)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.rmatvec(x) + self.B.rmatvec(x)

    def matmat(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.matmat(x) + self.B.matmat(x)

    def rmatmat(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.rmatmat(x) + self.B.rmatmat(x)

    def adjoint(self) -> LinearOperator:
        return self.A.H + self.B.H

    def invT(self) -> LinearOperator:
        if isinstance(self.A, InvertibleLinearOperator) and isinstance(
            self.B, InvertibleLinearOperator
        ):
            return self.A.invT() + self.B.invT()
        raise NotImplementedError


class _ProductLinearOperator(InvertibleLinearOperator):
    """
    Product (composition) of two linear operators.
    """

    def __init__(self, A: LinearOperator, B: LinearOperator):
        if A.shape[1] != B.shape[0]:
            raise ValueError("A and B must have composable shapes")

        prod_dtype = torchemlp.utils.merge_torch_types(A.dtype, B.dtype)
        super(_ProductLinearOperator, self).__init__(
            prod_dtype, (A.shape[0], B.shape[1])
        )

        self.A = A
        self.B = B
        self.args = (A, B)

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.A.matvec(self.B.matvec(v))

    def rmatvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.B.rmatvec(self.A.rmatvec(v))

    def matmat(self, C: torch.Tensor) -> torch.Tensor:
        return self.A.matmat(self.B.matmat(C))

    def rmatmat(self, C: torch.Tensor) -> torch.Tensor:
        return self.B.rmatmat(self.A.rmatmat(C))

    def adjoint(self) -> LinearOperator:
        return self.B.H * self.A.H

    def invT(self) -> LinearOperator:
        if isinstance(self.A, InvertibleLinearOperator) and isinstance(
            self.B, InvertibleLinearOperator
        ):
            res = self.A.invT() * self.B.invT()
            match res:
                case LinearOperator():
                    return res
                case torch.Tensor():
                    return torchemlp.ops.lazify(res)
        raise NotImplementedError

    def dense(self) -> torch.Tensor:
        return self.A.dense() @ self.B.dense()


class _ScaledLinearOperator(InvertibleLinearOperator):
    """
    Scaling of a linear operator by a scalar constant
    """

    def __init__(self, A: LinearOperator, c: Union[float, int, torch.Tensor]):
        super(_ScaledLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A

        match c:
            case torch.Tensor():
                if c.ndim != 0:
                    raise ValueError("c must be a scalar")
                self.c = c
            case int() | float():
                self.c = torch.tensor(c, dtype=A.dtype, device=DEFAULT_DEVICE)

        self.args = (A, c)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.c * self.A.matvec(x)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.c) * self.A.rmatvec(x)

    def matmat(self, x: torch.Tensor) -> torch.Tensor:
        return self.c * self.A.matmat(x)

    def rmatmat(self, x: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.c) * self.A.rmatmat(x)

    def adjoint(self) -> LinearOperator:
        return torch.conj(self.c) * self.A.H

    def invT(self) -> InvertibleLinearOperator:
        return 1.0 / self.c * self.A.T

    def dense(self) -> torch.Tensor:
        return self.c * self.A.dense()


class _PowerLinearOperator(InvertibleLinearOperator):
    """
    Power of a linear operator by a scalar constant.
    """

    def __init__(self, A: LinearOperator, p: Union[int, torch.Tensor]):
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square operator")
        if p < 0:
            raise ValueError("p must be non-negative")

        super(_PowerLinearOperator, self).__init__(A.dtype, A.shape)
        self.p = int(p)
        self.A = A
        self.args = (A, p)

    def power(self, fun: Callable, x: Any) -> Any:
        rep = x.copy()
        for _ in range(self.p):
            rep = fun(rep)
        return rep

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.power(self.A.matvec, x)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.power(self.A.rmatvec, x)

    def matmat(self, x: torch.Tensor) -> torch.Tensor:
        return self.power(self.A.matmat, x)

    def rmatmat(self, x: torch.Tensor) -> torch.Tensor:
        return self.power(self.A.rmatmat, x)

    def adjoint(self) -> LinearOperator:
        return self.A.H**self.p

    def invT(self) -> InvertibleLinearOperator:
        if isinstance(self.A, InvertibleLinearOperator):
            return self.A.invT() ** self.p
        raise NotImplementedError


class MatrixLinearOperator(InvertibleLinearOperator):
    """
    A linear operator that wraps a matrix.
    """

    def __init__(self, A: torch.Tensor):
        super(MatrixLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A
        self.args = (A,)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.A @ x

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.H @ x

    def matmat(self, x: torch.Tensor) -> torch.Tensor:
        return self.A @ x

    def rmatmat(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.H @ x

    def adjoint(self) -> InvertibleLinearOperator:
        return MatrixLinearOperator(self.A.H)

    def invT(self) -> InvertibleLinearOperator:
        return MatrixLinearOperator(torch.inverse(self.A).T)


class ZeroOperator(InvertibleLinearOperator):
    def __init__(self):
        super(ZeroOperator, self).__init__(None, (0,))

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x, dtype=x.dtype, device=x.device)

    def matmat(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x, dtype=x.dtype, device=x.device)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x, dtype=x.dtype, device=x.device)

    def rmatmat(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x, dtype=x.dtype, device=x.device)

    def adjoint(self) -> InvertibleLinearOperator:
        return self

    def invT(self) -> InvertibleLinearOperator:
        return self


class IdentityOperator(InvertibleLinearOperator):
    """
    A linear operator that is the identity operator.
    """

    def __init__(self, dim: int):
        super(IdentityOperator, self).__init__(None, (dim, dim))
        self.args = (dim,)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def matmat(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def rmatmat(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def adjoint(self) -> LinearOperator:
        return self

    def invT(self) -> InvertibleLinearOperator:
        return self


class I(IdentityOperator):
    """
    Alias to IdentityOperator.
    """

    pass


class Lazy(InvertibleLinearOperator):
    """
    A lazy wrapper on MatrixLinearOperator.
    """

    def __init__(self, A: torch.Tensor):
        super(Lazy, self).__init__(A.dtype, A.shape)
        self.A = A
        self.args = (A,)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.A @ x

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.H @ x

    def matmat(self, x: torch.Tensor) -> torch.Tensor:
        return self.A @ x

    def rmatmat(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.H @ x

    def dense(self, device: torch.device = DEFAULT_DEVICE) -> torch.Tensor:
        return self.A.to(device)

    def invT(self) -> InvertibleLinearOperator:
        return Lazy(torch.linalg.inv(self.A).H)
