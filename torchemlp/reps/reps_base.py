from typing import Union, Optional, Any, Literal, List, Tuple, Dict
from abc import ABC, abstractmethod

from torchemlp.utils import (
    GroupElem,
    LieAlgebraElem,
    ReprElem,
)

from torchemlp.groups import Group
from torchemlp.ops import (
    LinearOperator,
    ZeroOperator,
    I,
    LazyJVP,
    LazyConcat,
)
from torchemlp.ops import densify, lazify

from reps_utils import dictify_rep
from reps_solvers import orthogonal_complement, krylov_constraint_solve

from reps_algebra import SumRep, ProductRep, DeferredSumRep, DeferredProductRep

import torch


class Rep(ABC):
    """
    Abstract base class for a group representation meaning the vector space on
    which a group acts. Representations contain a set of discrete group
    generators and the Lie algebra, which can be considered as a set of
    continuous group generators. These types can be transformed via the direct
    sum, direct product, and dual operations. Rep objects must be immutable.
    """

    # Cache of canonicalized reps of the Rep class (used by the EMLP solver)
    solcache: Dict["Rep", LinearOperator] = dict()

    is_permutation: bool = False

    def __init__(self, G: Optional[Group] = None):
        """
        Initialize a rep object with the group it acts on.
        """
        self.G = G

    @abstractmethod
    def rho(self, g: GroupElem) -> ReprElem:
        """
        Calculate the discrete group representation of an input matrix M.
        """
        pass

    def drho(self, A: LieAlgebraElem) -> ReprElem:
        """
        Calculate the Lie algebra representation of an input matrix A.
        """
        if isinstance(A, torch.Tensor):
            A_dense = A
        elif isinstance(A, LinearOperator):
            A_dense = A.dense

        I = torch.eye(A.shape[0], dtype=A.dtype)
        return LazyJVP(self.rho, I, A_dense)

    @abstractmethod
    def __call__(self, G: Group) -> "Rep":
        """
        Construct the representation of a given group using the
        current representation.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        return repr(self)

    def __eq__(self, other: "Rep") -> bool:
        return dictify_rep(self) == dictify_rep(other)

    def __hash__(self) -> int:
        return hash((type(self), dictify_rep(self)))

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Returns the size of the vector space on which the group acts.
        """
        pass

    def canonicalize(self) -> Tuple["Rep", torch.Tensor]:
        """
        Return canonical form of representation. This enables you to reuse
        equivalent solutions in the EMLP solver. Should return the canonically
        ordered representation and the permutation used to go from the current
        representation to the cannonical representation.

        This implementation of this method assumes that the current
        representation is the canonical form. Overload this for non-canonical
        representations.
        """
        return self, torch.arange(self.size)

    def rho_dense(self, g: GroupElem) -> torch.Tensor:
        return densify(self.rho(g))

    def drho_dense(self, A: LieAlgebraElem) -> torch.Tensor:
        return densify(self.drho(A))

    def constraint_matrix(self) -> LinearOperator:
        """
        Get the equivariance constraint matrix.
        """
        n = self.size
        constraints = []

        if self.G is not None:
            for h in self.G.discrete_generators:
                constraints.append(lazify(self.rho(h) - I(n)))
            for A in self.G.lie_algebra:
                constraints.append(lazify(self.drho(A) - I(n)))

        if len(constraints) > 0:
            return LazyConcat(*constraints)
        return ZeroOperator()

    def equivariant_basis(self) -> LinearOperator:
        """
        Get the equivariant solution basis for the given representation via its
        canonicalization. Caches each canonicalization to a class variable.
        """

        canon_rep, perm = self.canonicalize()
        invperm = torch.argsort(perm)

        if canon_rep not in self.__class__.solcache:
            C = canon_rep.constraint_matrix()
            if C.shape[0] * C.shape[1] < 3e7:  # SVD
                result = orthogonal_complement(C.dense)
            else:  # too big for SVD, use iterative krylov solver
                result = krylov_constraint_solve(C)
            self.__class__.solcache[canon_rep] = lazify(result)

        if invperm == perm:
            return self.__class__.solcache[canon_rep]
        return lazify(self.__class__.solcache[canon_rep].dense[invperm])

    def equivariant_projector(self) -> LinearOperator:
        """
        Computes Q @ Q.H lazily to project onto the equivariant basis.
        """
        Q_lazy = self.equivariant_basis
        return Q_lazy @ Q_lazy.H

    def __add__(self, other: Union["Rep", int, torch.Tensor]) -> "Rep":
        """
        Compute the direct sum of representations.
        """

        match other:
            case Rep():
                if self.G is not None and other.G is not None:
                    return SumRep([self, other])
                return DeferredSumRep([self, other])
            case int():
                if other == 0:
                    return self
                return self + other * Scalar
            case torch.Tensor():
                if other.ndim == 0:
                    if other == 0:
                        return self
                    return self + int(other) * Scalar
                raise ValueError("Can only add reps, ints, and singleton int tensors")

    def __radd__(self, other: Union["Rep", int, torch.Tensor]) -> "Rep":
        """
        Compute the direct sum of representations in reverse order.
        """

        match other:
            case Rep():
                if self.G is not None and other.G is not None:
                    return SumRep([other, self])
                return DeferredSumRep([other, self])
            case int():
                if other == 0:
                    return self
                return other * Scalar + self
            case torch.Tensor():
                if other.ndim == 0:
                    if other == 0:
                        return self
                    return int(other) * Scalar + self
                raise ValueError("Can only add reps, ints, and singleton int tensors")

    def __mul__(self, x: Union["Rep", int]) -> "Rep":
        """
        If x is a rep, return the direct product of self and x.
        If x is an int, return the repeated tensor sum of self, x times
        """

        match x:
            case Rep():
                if self.G is not None and x.G is not None:
                    return ProductRep([self, x])
                return DeferredProductRep([self, x])

            case int():
                assert x >= 0, "Cannot multiply negative number of times"

                if x == 1:
                    return self
                elif x == 0:
                    return ZeroRep()
                elif self.G is not None:
                    return SumRep([self for _ in range(x)])
                return DeferredSumRep([self for _ in range(x)])

    def __rmul__(self, x: Union["Rep", int]) -> "Rep":
        """
        If x is a rep, return the direct product of self and x.
        If x is an int, return the repeated tensor sum of self, x times
        """

        match x:
            case Rep():
                if self.G is not None and x.G is not None:
                    return ProductRep([x, self])
                return DeferredProductRep([x, self])

            case int():
                assert x >= 0, "Cannot multiply negative number of times"

                if x == 1:
                    return self
                elif x == 0:
                    return ZeroRep()
                elif self.G is not None:
                    return SumRep([self for _ in range(x)])
                return DeferredSumRep([self for _ in range(x)])

    def __pow__(self, other: int) -> "Rep":
        """
        Compute the iterated tensor product of representations.
        """
        assert other >= 0, "Power only supported for non-negative integers"
        out = Scalar
        for _ in range(other):
            out = out * self
        return out

    def __rshift__(self, other: int) -> "Rep":
        """
        Compute the adjoint map from self->other, overloading the >> operator.
        """
        return other * self.T

    def __lshift__(self, other: "Rep") -> "Rep":
        """
        Compute the adjoint map from other->self, overloading the << operator.
        """
        return self * other.T

    def __lt__(self, other: "Rep") -> bool:
        """
        Arbitrarily defined order to sort representations, overloading the <
        operator. Sorts by group, then size, then hash.
        """
        match other:
            case ScalarRep():
                return False

        if self.G is not None and other.G is not None:
            if self.G < other.G:
                return True
            elif self.G > other.G:
                return False

        if self.size < other.size:
            return True
        elif self.size > other.size:
            return False

        return hash(self) < hash(other)

    def __mod__(self, other: "Rep") -> "Rep":
        """
        Compute the Wreath product of representations (not implemented yet)
        """
        return NotImplemented

    @property
    def T(self) -> "Rep":
        if self.G is not None and self.G.is_orthogonal:
            return self
        raise NotImplementedError


class ZeroRep(Rep):
    """
    Represents the zero vector.
    """

    def rho(self, g: GroupElem) -> ReprElem:
        return torch.zeros(g.shape, dtype=g.dtype)

    def drho(self, A: LieAlgebraElem) -> ReprElem:
        return torch.zeros(A.shape, dtype=A.dtype)

    def __call__(self, G: Group) -> Rep:
        # return self
        # TODO CHECK THAT ITS OK TO RETURN A NEW REPRESENTATION, NOT A COPY
        return ZeroRep()

    def __repr__(self) -> str:
        return "0V"

    @property
    def size(self) -> int:
        return 0

    @property
    def constraint_matrix(self) -> LinearOperator:
        return ZeroOperator()


class ScalarRep(Rep):
    def __init__(self, G: Optional[Group] = None):
        self.G = G
        self.is_permutation = True

    def __call__(self, G: Optional[Group]) -> Rep:
        # self.G = G
        # return self
        # TODO CHECK THAT ITS OK TO RETURN A NEW REPRESENTATION, NOT A COPY
        return ScalarRep(G)

    def size(self) -> int:
        return 1

    def __repr__(self) -> str:
        return "V⁰"

    def rho(self, g: GroupElem) -> ReprElem:
        return torch.eye(1)

    def drho(self, A: LieAlgebraElem) -> ReprElem:
        return 0 * torch.eye(1)

    @property
    def constraint_matrix(self) -> LinearOperator:
        raise NotImplementedError

    @property
    def equivariant_basis(self):
        return torch.ones((1, 1))

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: Rep) -> bool:
        return isinstance(other, ScalarRep)

    def __mul__(self, other: Rep | int) -> Rep:
        match other:
            case Rep():
                return other
            case int():
                return super().__mul__(other)

    def T(self) -> Rep:
        return self


class Base(Rep):
    """
    Base representation V of a group which will be used to construct other
    representations.
    """

    def __init__(self, G: Optional[Group] = None):
        self.G = G
        if G is not None:
            self.is_permutation = G.is_permutation

    def __call__(self, G: Group) -> "Base":
        # return self.__class__(G)
        return Base(G)

    def rho(self, g: GroupElem | Dict[Group, torch.Tensor]) -> ReprElem:
        if isinstance(g, GroupElem):
            return g
        if self.G is not None and isinstance(g, dict) and self.G in g:
            return g[self.G]
        raise ValueError("M must be a dictionary or a group element")

    def drho(self, A: LieAlgebraElem | Dict[Group, torch.Tensor]) -> ReprElem:
        if isinstance(A, LieAlgebraElem):
            return A
        if self.G is not None and isinstance(A, dict) and self.G in A:
            return A[self.G]
        raise ValueError("M must be a dictionary or a Lie algebra element")

    @property
    def constraint_matrix(self) -> LinearOperator:
        """
        Get the equivariance constraint matrix.
        """
        if self.G is None:
            raise NotImplementedError

        discrete_constraints = [
            lazify(self.rho(h)) - I(self.size) for h in self.G.discrete_generators
        ]
        continuous_constraints = [lazify(self.drho(A)) for A in self.G.lie_algebra]
        constraints = discrete_constraints + continuous_constraints

        return (
            LazyConcat(constraints)
            if constraints
            else lazify(torch.zeros((1, self.size)))
        )

    @property
    def size(self) -> int:
        if self.G is None:
            return 0
        return self.G.d

    @property
    def T(self) -> "Base":
        if self.G is not None and self.G.is_orthogonal:
            return self
        return Dual(self)

    def __repr__(self) -> str:
        return "V"

    def __hash__(self) -> int:
        return hash((type(self), self.G))

    def __eq__(self, other: Rep) -> bool:
        if isinstance(other, Base):
            return self.G == other.G
        return False

    def __lt__(self, other: Rep) -> bool:
        if isinstance(other, Dual):
            return True
        return super().__lt__(other)


class Dual(Base):
    """
    Dual representation to the input representation.
    """

    def __init__(self, rep: Base):
        self.rep = rep
        self.G = rep.G
        self.is_permutation = rep.is_permutation

    def __call__(self, G: Group) -> Base:
        return self.rep(G).T

    def rho(self, g: GroupElem) -> ReprElem:
        rr = self.rep.rho(g)
        match rr:
            case torch.Tensor():
                return torch.linalg.inv(rr).T
            case LinearOperator():
                return rr.invT()

    def drho(self, A: LieAlgebraElem) -> ReprElem:
        return -self.rep.drho(A).T

    def __repr__(self) -> str:
        return repr(self.rep) + "*"

    @property
    def T(self) -> Base:
        return self.rep

    def __eq__(self, other: Rep) -> bool:
        match other:
            case Dual():
                return self.rep == other.rep
        return False

    def __hash__(self) -> int:
        return hash((type(self), self.rep))

    def __lt__(self, other: Rep) -> bool:
        if other == self.rep:
            return False
        return super().__lt__(other)

    @property
    def size(self) -> int:
        return self.rep.size


# The base instances of the Vector and Scalar representations
V = Vector = Base()
Scalar = ScalarRep()
