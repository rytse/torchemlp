from typing import Optional, Dict

import torch

from torchemlp.utils import (
    GroupElem,
    LieAlgebraElem,
    ReprElem,
)

from torchemlp.groups import Group
from torchemlp.ops import (
    LinearOperator,
    I,
    LazyConcat,
)
from torchemlp.ops import lazify

from .reps_base import Rep


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
