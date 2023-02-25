from abc import ABC, abstractmethod

import torch
import functorch

from utils import is_scalar, is_vector, is_matrix


class LinearOperator(ABC):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape

    @abstractmethod
    def matvec(self, x):
        """
        Perform y = Ax where A is the linear operator. Overload this with a
        fast custom implementation given the structure of the operator.
        """
        pass

    def matmat(self, B):
        """
        Perform Y = A B where A is self's linear operator and B is another
        linear operator (or matrix). This will call a vectorized version of
        matvec unless you overload it with something faster.
        """
        return functorch.vmap(self.matvec)(B.T).T

    def adjoint(self):
        return _AdjointLinearOperator(self)

    H = property(adjoint)

    def transpose(self):
        return _TransposedLinearOperator(self)

    T = property(transpose)

    def rmatvec(self, x):
        """
        Perform adjoint y = A^H x where A is the linear operator. Creates a new
        instance of _AdjointLinearOperator(self) every time this is called.

        If you need performance, overload this with a custom implemntation that
        doesn't recompute self.H every time.
        """
        return self.H.matvec(x)

    def rmatmat(self, B):
        """
        Perform Y = A^H B where A is self's linear operator and B is another
        linear operator (or matrix). This will call a vectorized version of
        rmatvec unless you overload it with something faster.
        """
        return functorch.vmap(self.rmatvec)(B.T).T

    def __mul__(self, x):
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

    def __call__(self, x):
        return self * x

    def __matmul__(self, x):
        return self * x

    def __rmul__(self, x):
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

    def __rmatmul__(self, x):
        return x * self

    def __pow__(self, p):
        """
        This python operator overloads the ** operator for scalar p.
        """
        if is_scalar(p):
            return _PowerLinearOperator(self, p)
        else:
            return NotImplemented

    def __add__(self, x):
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

    def __radd__(self, x):
        return self + x

    def __neg__(self):
        return _ScaledLinearOperator(self, -1)

    def __sub__(self, x):
        return self + (-x)

    def __repr__(self):
        M, N = self.shape
        if self.dtype is None:
            dt = "unspecified dtype"
        else:
            dt = f"dtype={self.dtype}"
        return f"<{M}x{N} {self.__class__.__name__} with {dt}>"

    @property
    def dense(self):
        """
        Convert the lazy implementation of the linear operator to a matrix
        representing the operator in R^n.
        """
        return self @ torch.eye(self.shape[-1])


class _AdjointLinearOperator(LinearOperator):
    """
    Adjoint of a linear operator.
    """

    def __init__(self, A):
        super(_AdjointLinearOperator, self).__init__(A.dtype, (A.shape[1], A.shape[0]))
        self.A = A
        self.args = (A,)

    def matvec(self, x):
        return self.A.rmatvec(x)

    def rmatvec(self, x):
        return self.A.matvec(x)

    def matmat(self, B):
        return self.A.rmatmat(B)

    def rmatmat(self, B):
        return self.A.matmat(B)


class _TransposedLinearOperator(LinearOperator):
    """
    Transpose of a linear operator.
    """

    def __init__(self, A):
        super(_TransposedLinearOperator, self).__init__(
            A.dtype, (A.shape[1], A.shape[0])
        )
        self.A = A
        self.args = (A,)

    def matvec(self, x):
        return torch.conj(self.A.rmatvec(torch.conj(x)))

    def rmatvec(self, x):
        return torch.conj(self.A.matvec(torch.conj(x)))

    def matmat(self, B):
        return torch.conj(self.A.rmatmat(torch.conj(B)))

    def rmatmat(self, B):
        return torch.conj(self.A.matmat(torch.conj(B)))


class _SumLinearOperator(LinearOperator):
    """
    Sum of two linear operators.
    """

    def __init__(self, A, B):
        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise ValueError("A and B must be LinearOperators")
        if A.shape != B.shape:
            raise ValueError("A and B must have the same shape")
        if A.dtype != B.dtype:
            raise ValueError("A and B must have the same dtype")

        super(_SumLinearOperator, self).__init__(A.dtype, A.shape)

        self.A = A
        self.B = B
        self.args = (A, B)

    def matvec(self, x):
        return self.A.matvec(x) + self.B.matvec(x)

    def rmatvec(self, x):
        return self.A.rmatvec(x) + self.B.rmatvec(x)

    def matmat(self, x):
        return self.A.matmat(x) + self.B.matmat(x)

    def rmatmat(self, x):
        return self.A.rmatmat(x) + self.B.rmatmat(x)

    def adjoint(self):
        return self.A.H + self.B.H

    def invT(self):
        return self.A.invT() + self.B.invT()


class _ProductLinearOperator(LinearOperator):
    """
    Product (composition) of two linear operators.
    """

    def __init__(self, A, B):
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

    def matvec(self, x):
        return self.A.matvec(self.B.matvec(x))

    def rmatvec(self, x):
        return self.B.rmatvec(self.A.rmatvec(x))

    def matmat(self, x):
        return self.A.matmat(self.B.matmat(x))

    def rmatmat(self, x):
        return self.B.rmatmat(self.A.rmatmat(x))

    def adjoint(self):
        return B.H * A.H

    def invT(self):
        return A.invT() * B.invT()

    def to_dense(self):
        Ad = A.dense if isinstance(A, LinearOperator) else A
        Bd = B.dense if isinstance(B, LinearOperator) else B
        return Ad @ Bd


class _ScaledLinearOperator(LinearOperator):
    """
    Scaling of a linear operator by a scalar constant
    """

    def __init__(self, A, c):
        if not isinstance(A, LinearOperator):
            raise ValueError("A must be a LinearOperator")
        if not is_scalar(c):
            raise ValueError("c must be a scalar")

        super(_ProductLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A
        self.c = c
        self.args = (A, c)

    def matvec(self, x):
        return self.c * self.A.matvec(x)

    def rmatvec(self, x):
        return torch.conj(self.c) * self.A.rmatvec(x)

    def matmat(self, x):
        return self.c * self.A.matmat(x)

    def rmatmat(self, x):
        return torch.conj(self.c) * self.A.rmatmat(x)

    def adjoint(self):
        return torch.conj(self.c) * A.H

    def invT(self):
        return 1.0 / self.c * A.T

    def to_dense(self):
        return self.c * self.A.dense


class _PowerLinearOperator(LinearOperator):
    """
    Power of a linear operator by a scalar constant.
    """

    def __init__(self, A, p):
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

    def power(self, fun, x):
        rep = x.copy()
        for _ in range(self.p):
            rep = fun(rep)
        return rep

    def matvec(self, x):
        return self.power(self.A.matvec, x)

    def rmatvec(self, x):
        return self.power(self.A.rmatvec, x)

    def matmat(self, x):
        return self.power(self.A.matmat, x)

    def rmatmat(self, x):
        return self.power(self.A.rmatmat, x)

    def adjoint(self):
        return self.A.H ** self.p

    def invT(self):
        return self.A.invT() ** self.p


class MatrixLinearOperator(LinearOperator):
    """
    A linear operator that wraps a matrix.
    """

    def __init__(self, A):
        super(MatrixLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A
        self.args = (A,)

    def matvec(self, x):
        return self.A @ x

    def rmatvec(self, x):
        return self.A.H @ x

    def matmat(self, x):
        return self.A @ x

    def rmatmat(self, x):
        return self.A.H @ x

    def adjoint(self):
        return MatrixLinearOperator(self.A.H)


class IdentityOperator(LinearOperator):
    """
    A linear operator that is the identity operator.
    """

    def __init__(self, dtype, shape):
        super(IdentityOperator, self).__init__(dtype, shape)
        self.args = (dtype, shape)

    def matvec(self, x):
        return x

    def rmatvec(self, x):
        return x

    def matmat(self, x):
        return x

    def rmatmat(self, x):
        return x

    def adjoint(self):
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

    def __init__(self, A):
        super(Lazy, self).__init__(A.dtype, A.shape)
        self.A = A
        self.args = (A,)

    def matvec(self, x):
        return self.A @ x

    def rmatvec(self, x):
        return self.A.H @ x

    def matmat(self, x):
        return self.A @ x

    def rmatvec(self, x):
        return self.A.H @ x

    def to_dense(self):
        return self.A

    def invT(self):
        return Lazy(torch.linalg.inv(self.A).H)
