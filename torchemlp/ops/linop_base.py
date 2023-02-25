from typing import Union, Callable, Any
from abc import ABC, abstractmethod

import torch
import functorch

from utils import is_scalar, is_vector, is_matrix


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
        return functorch.vmap(self.matvec)(B.T).T

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
        return self.H.matvec(x)

    def rmatmat(self, B: torch.Tensor) -> torch.Tensor:
        """
        Perform Y = A^H B where A is self's linear operator and B is another
        linear operator (or matrix). This will call a vectorized version of
        rmatvec unless you overload it with something faster.
        """
        return functorch.vmap(self.rmatvec)(B.T).T

    def __mul__(
        self, x: Union["LinearOperator", torch.Tensor, float]
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
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(self, x)
        elif is_scalar(x):
            return _ScaledLinearOperator(self, x)
        elif is_vector(x):
            return self.matvec(x)
        elif is_matrix(x):
            return self.matmat(x)
        raise ValueError("Unsupported type {}".format(type(x)))

    def __call__(
        self, x: Union["LinearOperator", torch.Tensor, float]
    ) -> Union["LinearOperator", torch.Tensor]:
        return self * x

    def __matmul__(
        self, x: Union["LinearOperator", torch.Tensor, float]
    ) -> Union["LinearOperator", torch.Tensor]:
        return self * x

    def __rmul__(
        self, x: Union["LinearOperator", torch.Tensor, float]
    ) -> Union["LinearOperator", torch.Tensor]:
        """
        This python operator overloads right-application (to handle
        non-commutative operators). This only works for LinearOperators and
        scalars, as right-application to a vector is not always defined by
        self.matvec. You can overload this function to cover this case!

        Several other dunder methods are overloaded to perform the same task.
        """
        if isinstance(x, LinearOperator):
            return _ProductLinearOperator(x, self)
        elif is_scalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            return NotImplemented

    def __rmatmul__(
        self, x: Union["LinearOperator", torch.Tensor, float]
    ) -> Union["LinearOperator", torch.Tensor]:
        return x * self

    def __pow__(self, p: Union[int, torch.Tensor]):
        """
        This python operator overloads the ** operator for scalar p.
        """
        if is_scalar(p):
            return _PowerLinearOperator(self, p)
        else:
            return NotImplemented

    def __add__(self, x: Union["LinearOperator", torch.Tensor]) -> "LinearOperator":
        """
        This python operator overloads the + operator. It creates a new linear
        operator via addition in the vector space of linear operators. This
        method works on LinearOperators and torch matrices.

        Several other dunder methods are overloaded to perform the same task.
        """
        if isinstance(x, LinearOperator):
            return _SumLinearOperator(self, x)
        elif is_matrix(x):
            return _SumLinearOperator(self, Lazy(x))
        else:
            return NotImplemented

    def __radd__(self, x: Union["LinearOperator", torch.Tensor]) -> "LinearOperator":
        return self + x

    def __neg__(self) -> "LinearOperator":
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x: Union["LinearOperator", torch.Tensor]) -> "LinearOperator":
        return self + (-x)

    def __repr__(self) -> str:
        M, N = self.shape
        if self.dtype is None:
            dt = "unspecified dtype"
        else:
            dt = f"dtype={self.dtype}"
        return f"<{M}x{N} {self.__class__.__name__} with {dt}>"

    @property
    def dense(self) -> torch.Tensor:
        """
        Convert the lazy implementation of the linear operator to a matrix
        representing the operator in R^n.
        """
        return self @ torch.eye(self.shape[-1])


class GroupElement(LinearOperator):
    """
    A linear operator that we think of as a group element.

    __brand_group_elem__ keeps group element types distinct from representation
    element types.
    """

    __brand_group_elem__: bool = True


class ReprElement(LinearOperator):
    """
    A linear operator that we think of as a representation of a group element.

    __brand_repr_elem__ keeps representation element types distinct from group
    element types.
    """

    __brand_repr_elem__: bool = True


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


class _SumLinearOperator(LinearOperator):
    """
    Sum of two linear operators.
    """

    def __init__(self, A: LinearOperator, B: LinearOperator):
        if A.shape != B.shape:
            raise ValueError("A and B must have the same shape")
        if A.dtype != B.dtype:
            raise ValueError("A and B must have the same dtype")

        super(_SumLinearOperator, self).__init__(A.dtype, A.shape)

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
        if hasattr(self.A, "invT") and hasattr(self.B, "invT"):
            return self.A.invT() + self.B.invT()
        raise NotImplementedError


class _ProductLinearOperator(LinearOperator):
    """
    Product (composition) of two linear operators.
    """

    def __init__(self, A: LinearOperator, B: LinearOperator):
        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise ValueError("A and B must be LinearOperators")
        if A.shape[1] != B.shape[0]:
            raise ValueError("A and B must have composable shapes")
        if A.dtype != B.dtype:
            raise ValueError("A and B must have the same dtype")

        super(_ProductLinearOperator, self).__init__(A.dtype, (A.shape[0], B.shape[1]))
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
        if hasattr(self.A, "invT") and hasattr(self.B, "invT"):
            return self.A.invT() * self.B.invT()
        raise NotImplementedError

    def to_dense(self) -> LinearOperator:
        return self.A.dense @ self.B.dense


class _ScaledLinearOperator(LinearOperator):
    """
    Scaling of a linear operator by a scalar constant
    """

    def __init__(self, A: LinearOperator, c: Union[float, int, torch.Tensor]):
        if not isinstance(A, LinearOperator):
            raise ValueError("A must be a LinearOperator")
        if not is_scalar(c):
            raise ValueError("c must be a scalar")

        super(_ScaledLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A
        if isinstance(c, torch.Tensor):
            self.c = c
        else:
            self.c = torch.Tensor(c)
        self.args = (A, c)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self.c * self.A.matvec(x)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.c) * self.A.rmatvec(x)

    def matmat(self, x: torch.Tensor) -> torch.Tensor:
        return self.c * self.A.matmat(x)

    def rmatmat(self, x: torch.Tensor) -> torch.Tensor:
        return torch.conj(self.c) * self.A.rmatmat(x)

    def adjoint(self):
        return torch.conj(self.c) * self.A.H

    def invT(self):
        return 1.0 / self.c * self.A.T

    def to_dense(self):
        return self.c * self.A.dense


class _PowerLinearOperator(LinearOperator):
    """
    Power of a linear operator by a scalar constant.
    """

    def __init__(self, A: LinearOperator, p: Union[int, torch.Tensor]):
        if not isinstance(A, LinearOperator):
            raise ValueError("A must be a LinearOperator")
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square operator")
        if not isinstance(p, int):
            raise ValueError("p must be an int")
        if p < 0:
            raise ValueError("p must be non-negative")

        super(_PowerLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A
        self.p = p
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

    def invT(self) -> LinearOperator:
        if hasattr(self.A, "invT"):
            return self.A.invT() ** self.p
        raise NotImplementedError


class MatrixLinearOperator(LinearOperator):
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

    def adjoint(self) -> LinearOperator:
        return MatrixLinearOperator(self.A.H)


class IdentityOperator(LinearOperator):
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


class I(IdentityOperator):
    """
    Alias to IdentityOperator.
    """

    pass


class Lazy(LinearOperator):
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

    def to_dense(self) -> torch.Tensor:
        return self.A

    def invT(self) -> LinearOperator:
        return Lazy(torch.linalg.inv(self.A).H)
