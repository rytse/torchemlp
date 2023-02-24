from abc import ABC, abstractmethod

from torchemlp.utils import is_scalar

from torchemlp.ops import LinearOperator, LazyJVP, LazyConcat
from torchemlp.ops import densify, lazify

from reps_utils import dictify_rep, mul_reps
from product_sum_reps import SumRep, DeferredSumRep

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
    solcache = dict()

    is_permutation = False

    @abstractmethod
    def rho(self, M):
        """
        Calculate the discrete group representation of an input matrix M.
        """
        pass

    @abstractmethod
    def drho(self, A):
        """
        Calculate the Lie algebra representation of an input matrix A.
        """
        I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        return LazyJVP(self.rho, I, A)

    @abstractmethod
    def __call__(self, G):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return dictify_rep(self) == dictify_rep(other)

    def __hash__(self):
        return hash((type(self), dictify_rep(self)))

    @property
    def concrete(self):
        return hasattr(self, "G") and self.G is not None

    @property
    def size(self):
        """
        Returns the size of the vector space on which the group acts.
        """
        if self.concrete and self.G is not None:
            return self.rho(self.G.sample()).shape[-1]
        return NotImplemented

    def canonicalize(self):
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

    def rho_dense(self, M):
        return densify(self.rho(M))

    def drho_dense(self, A):
        return densify(self.drho(A))

    @property
    def constraint_matrix(self):
        """
        Get the equivariance constraint matrix.
        """
        if not self.concrete:
            raise ValueError("Representation does not have a group")
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
    def equivariant_basis(self):
        """
        Get the equivariant solution basis for the given representation via its
        canonicalization. Caches each canonicalization to a class variable.
        """

        if isinstance(self, Scalar):
            return torch.ones((1, 1))

        canon_rep, perm = self.canonicalize()
        invperm = torch.argsort(perm)

        if canon_rep not in self.__class__.solcache:
            C = canon_rep.constraint_matrix
            if C.shape[0] * C.shape[1] > 3e7:  # too large to use SVD
                result = krylov_constraint_solve(C)
            else:
                result = orthogonal_complement(C.dense)
            self.__class__.solcache[canon_rep] = result

        return self.__class__.solcache[canon_rep][invperm]

    @property
    def equivariant_projector(self):
        """
        Computes Q @ Q.H lazily to project onto the equivariant basis.
        """
        Q_lazy = lazify(self.equivariant_basis)
        return Q_lazy @ Q_lazy.H

    def __add__(self, other):
        """
        Compute the direct sum of representations.
        """
        if is_scalar(other):
            if other == 0:
                return self
            return self + other * Scalar

        if self.concrete and other.concrete:
            return SumRep(self, other)

        return DeferredSumRep(self, other)

    def __radd__(self, other):
        """
        Compute the direct sum of representations in reverse order.
        """
        if is_scalar(other):
            if other == 0:
                return self
            return other * Scalar + self

        if self.concrete and other.concrete:
            return SumRep(other, self)

        return DeferredSumRep(other, self)

    def __mul__(self, other):
        """
        Compute the tensor sum of representations.
        """
        return mul_reps(self, other)

    def __rmul__(self, other):
        """
        Compute the tensor sum of representations in reverse order.
        """
        return mul_reps(other, self)

    def __pow__(self, other):
        """
        Compute the iterated tensor product of representations.
        """
        assert (
            isinstance(other, int) and other >= 0
        ), "Power only supported for non-negative integers"
        out = Scalar
        for _ in range(other):
            out *= self
        return out

    def __rshift__(self, other):
        """
        Compute the adjoint map from self->other, overloading the >> operator.
        """
        return other * self.H

    def __lshift__(self, other):
        """
        Compute the adjoint map from other->self, overloading the << operator.
        """
        return self * other.H

    def __lt__(self, other):
        """
        Arbitrarily defined order to sort representations, overloading the <
        operator. Sorts by group, then size, then hash.
        """
        if other == Scalar:
            return False

        if self.concrete and other.concrete:
            if self.G < other.G:
                return True
            elif self.G > other.G:
                return False

        if self.size < other.size:
            return True
        elif self.size > other.size:
            return False

        return hash(self) < hash(other)

    def __mod__(self, other):
        """
        Compute the Wreath product of representations (not implemented yet)
        """
        return NotImplemented

    @property
    def T(self):
        if self.concrete and self.G.is_orthogonal:
            return self
        return Dual(self)


class ScalarRep(Rep):
    def __init__(self, G=None):
        self.G = G
        self.is_permutation = True

    def __call__(self, G):
        self.G = G
        return self

    def size(self):
        return 1

    def __repr__(self):
        return "V⁰"

    def rho(self, M):
        return torch.eye(1)

    def drho(self, A):
        return 0 * torch.eye(1)

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, ScalarRep)

    def __mul__(self, other):
        if isinstance(other, int):
            return super().__mul__(other)
        return other

    def __rmul__(self, other):
        if isinstance(other, int):
            return super().__rmul__(other)
        return other

    def concrete(self):
        return True

    def T(self):
        return self


class Base(Rep):
    """
    Base representation V of a group which will be used to construct other
    representations.
    """

    def __init__(self, G=None):
        self.G = G
        if G is not None:
            self.is_permutation = G.is_permutation

    def __call__(self, G):
        return self.__class__(G)

    def rho(self, M):
        if self.concrete and isinstance(M, dict):
            M = M[self.G]
        return M

    def drho(self, A):
        if self.concrete and isinstance(A, dict):
            A = A[self.G]
        return A

    def size(self):
        if self.G is None:
            raise ValueError("Must have G to find size")
        return self.G.d

    def __repr__(self):
        return "V"

    def __hash__(self):
        return hash((type(self), self.G))

    def __eq__(self, other):
        return type(other) == type(self) and self.G == other.G

    def __lt__(self, other):
        if isinstance(other, Dual):
            return True
        return super().__lt__(other)


class Dual(Rep):
    """
    Dual representation to the input representation.
    """

    def __init__(self, rep):
        self.rep = rep
        self.G = rep.G
        if hasattr(rep, "is_permutation"):
            self.is_permutation = rep.is_permutation

    def __call__(self, G):
        return self.rep(G).T

    def rho(self, M):
        rr = self.rep.rho(M)
        return rr.invT() if isinstance(rr, LinearOperator) else torch.linalg.inv(rr).T

    def drho(self, A):
        return -self.rep.drho(A).T

    def __repr__(self):
        return repr(self.rep) + "*"

    def T(self):
        return self.rep

    def __eq__(self, other):
        return type(other) == type(self) and self.rep == other.rep

    def __hash__(self):
        return hash((type(self), self.rep))

    def __lt__(self, other):
        if other == self.rep:
            return False
        return super().__lt__(other)

    def size(self):
        return self.rep.size


# The base instances of the Vector and Scalar representations
V = Vector = Base()
Scalar = ScalarRep()
