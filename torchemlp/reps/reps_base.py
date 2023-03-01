from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, Union, Generator
from functools import lru_cache, reduce
from collections import defaultdict
import itertools

import torch

# from torchemlp.utils import (
# GroupElem,
# LieAlgebraElem,
# ReprElem,
# )

from torchemlp.groups import Group
from torchemlp.ops import (
    lazy_direct_matmat,
    product,
    LinearOperator,
    ZeroOperator,
    I,
    LazyKron,
    LazyKronsum,
    LazyDirectSum,
    LazyJVP,
    LazyConcat,
    LazyPerm,
)
from torchemlp.ops import densify, lazify

from .reps_utils import dictify_rep
from .reps_solvers import orthogonal_complement, krylov_constraint_solve

# Elements of a group embedded as a tensor or linear operator.
# Has the same python type as LieAlgebraElem(s) and ReprElem(s), but we define
# them differently so that our methods' type signatures are more helpful.
GroupElem = Union[torch.Tensor, LinearOperator]

# Elements of a group's Lie algebra embedded as a tensor or linear operator.
# Has the same python type as GroupElem(s) and ReprElem(s), but we define them
# differently so that our methods' type signatures are more helpful.
LieAlgebraElem = Union[torch.Tensor, LinearOperator]

# Representation of a group element as a tensor or linear operator.
# Has the same python type as GroupElem(s) and LieAlgebraElem(s), but we define
# them differently so that our methods' type signatures are more helpful.
ReprElem = Union[torch.Tensor, LinearOperator]


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
        if self.G is None:
            return lazify(torch.zeros((1, self.size)))

        discrete_constraints = [
            lazify(self.rho(h)) - I(self.size) for h in self.G.discrete_generators
        ]
        continuous_constraints = [lazify(self.drho(A)) for A in self.G.lie_algebra]
        constraints = discrete_constraints + continuous_constraints

        return LazyConcat(constraints)

    def equivariant_basis(self) -> LinearOperator:
        """
        Get the equivariant solution basis for the given representation via its
        canonicalization. Caches each canonicalization to a class variable.
        """

        canon_rep, perm = self.canonicalize()
        invperm = torch.argsort(perm)

        if canon_rep not in self.solcache:
            C = canon_rep.constraint_matrix()
            if C.shape[0] * C.shape[1] < 3e7:  # SVD
                result = orthogonal_complement(C.dense)
            else:  # too big for SVD, use iterative krylov solver
                result = krylov_constraint_solve(C)
            self.solcache[canon_rep] = lazify(result)

        if all(invperm == perm):
            return self.__class__.solcache[canon_rep]
        return lazify(self.__class__.solcache[canon_rep].dense[invperm])

    def equivariant_projector(self) -> LinearOperator:
        """
        Computes Q @ Q.H lazily to project onto the equivariant basis.
        """
        Q_lazy = self.equivariant_basis()
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
                return self + other * ScalarRep(self.G)
            case torch.Tensor():
                if other.ndim == 0:
                    if other == 0:
                        return self
                    return self + int(other) * ScalarRep(self.G)
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
                return other * ScalarRep(self.G) + self
            case torch.Tensor():
                if other.ndim == 0:
                    if other == 0:
                        return self
                    return int(other) * ScalarRep(self.G) + self
                raise ValueError("Can only add reps, ints, and singleton int tensors")

    def __mul__(self, x: Union["Rep", int]) -> "Rep":
        """
        If x is a rep, return the direct product of self and x.
        If x is an int, return the repeated tensor sum of self, x times
        """

        # If we can distribute the tensor product, do so
        # TODO see if there's a more OOP-y / pythonic way to do this
        if isinstance(self, SumRep) and isinstance(x, Rep) or isinstance(x, SumRep):
            return distribute_product([self, x])

        match x:
            case ScalarRep():
                return x.__mul__(self)

            case Rep():
                if self.G is not None and x.G is not None:
                    return ProductRep([self, x])
                return DeferredProductRep([self, x])

            case int():
                assert x >= 0, "Cannot multiply negative number of times"

                if x == 1:
                    return self
                elif x == 0:
                    return ScalarRep(self.G)
                elif self.G is not None:
                    return SumRep([self for _ in range(x)])
                return DeferredSumRep([self for _ in range(x)])

    def __rmul__(self, x: Union["Rep", int]) -> "Rep":
        """
        If x is a rep, return the direct product of self and x.
        If x is an int, return the repeated tensor sum of self, x times
        """

        # If we can distribute the tensor product, do so
        # TODO see if there's a more OOP-y / pythonic way to do this
        if isinstance(self, SumRep) and isinstance(x, Rep) or isinstance(x, SumRep):
            return distribute_product([x, self])

        match x:
            case ScalarRep():
                return x.__rmul__(self)

            case Rep():
                if self.G is not None and x.G is not None:
                    return ProductRep([x, self])
                return DeferredProductRep([x, self])

            case int():
                assert x >= 0, "Cannot multiply negative number of times"

                if x == 1:
                    return self
                elif x == 0:
                    return ScalarRep(self.G)
                elif self.G is not None:
                    return SumRep([self for _ in range(x)])
                return DeferredSumRep([self for _ in range(x)])

    def __pow__(self, other: int) -> "Rep":
        """
        Compute the iterated tensor product of representations.
        """
        assert other >= 0, "Power only supported for non-negative integers"
        prodlist = [self for _ in range(other)]
        if len(prodlist) > 0:
            return reduce(lambda a, b: a * b, prodlist)
        return ScalarRep(self.G)

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
        # match other:
        # case ScalarRep():
        # return False

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
        return Dual(self)


# class ZeroRep(Rep):
# """
# Represents the zero vector.
# """

# def rho(self, g: GroupElem) -> ReprElem:
# return torch.zeros(g.shape, dtype=g.dtype)

# def drho(self, A: LieAlgebraElem) -> ReprElem:
# return torch.zeros(A.shape, dtype=A.dtype)

# def __call__(self, G: Group) -> Rep:
# # return self
# # TODO CHECK THAT ITS OK TO RETURN A NEW REPRESENTATION, NOT A COPY
# return ZeroRep()

# def __repr__(self) -> str:
# return "0V"

# @property
# def size(self) -> int:
# return 0

# def constraint_matrix(self) -> LinearOperator:
# return ZeroOperator()


class OpRep(Rep, ABC):
    """
    Abstract base class for operations on a list of representations.


    Complex represenations can be created by composing OpReps, and often times
    we will have repeated applications of OpReps to the same base Reps. In
    order to effectively exploit lazy operators, we keep track of the set of
    unique operators that occur in a composed OpRep and the places at which
    they occur. This information is saved in self.counters.

    Additionally, we perform many operations in terms of the cannonical
    representation of the Rep(s). Whenever an OpRep is constructed, we compute
    the canonicalized representation and save the permutations needed to return
    to the original (potentially non-canonical) represenation. We save the
    permutation for canonical->original in self.perms, and the permtation for
    original->canonical in self.invperms.
    """

    def __init__(
        self,
        reps: dict[Rep, int] | list[Rep],
        extra_perm: Optional[torch.Tensor] = None,
    ):
        """

        Compute the operation over a list of reps and apply any extra
        permutations specified.

        Args:
        ----------
            inreps: either a list of reps to compute the operation of or a dict
                    of reps and the number of times they should be repeated in
                    the tensor product in the tensor product.
            extra_perm: tensor to specify a non-canonical ordering
        """
        super().__init__(None)

        match reps:
            case dict():
                self.counters = reps

                match extra_perm:
                    case None:
                        perm = torch.arange(self.size)
                    case torch.Tensor():
                        perm = extra_perm
                    case _:  # TODO figure out why type checker doesn't understand "is None"
                        raise ValueError(
                            f"extra_perm must be None or torch.Tensor, not {type(extra_perm)}"
                        )

                self.counters, self.perm = self.__class__.compute_canonical(
                    [reps], (perm,)
                )

            case list():
                # Canonicalize each rep to be multiplied
                canreps, perms = zip(*[rep.canonicalize() for rep in reps])

                if not all(isinstance(perm, torch.Tensor) for perm in perms):
                    raise ValueError("Permutation must be a torch.Tensor")

                # Dict containing the set of unique Reps in the multiplication
                # and the number of occurances of each unique Rep
                in_counters = []
                for rep in canreps:
                    match rep:
                        case self.__class__():
                            in_counters.append(rep.counters)
                        case _:
                            in_counters.append({rep: 1})

                self.counters: dict[Rep, int] = dict()
                self.counters, perm = self.__class__.compute_canonical(
                    in_counters, perms
                )

                match extra_perm:
                    case None:
                        self.perm = perm
                    case torch.Tensor():
                        self.perm = extra_perm[perm]
                    case _:  # TODO figure out why type checker doesn't understand "is None"
                        raise ValueError(
                            f"extra_perm must be None or torch.Tensor, not {type(extra_perm)}"
                        )

        self.invperm = torch.argsort(self.perm)
        self.Gs = [rep.G for rep in self.counters.keys()]
        self.G = self.Gs[0] if len(set(self.Gs)) == 1 else None

        if all(self.Gs[0] == self.Gs[i] for i in range(len(self.Gs))):
            self.G = self.Gs[0]
        else:
            self.G = None

        self.canonical = torch.all(self.perm == self.invperm)
        self.is_permutation = all(rep.is_permutation for rep in self.counters.keys())

    @staticmethod
    @abstractmethod
    def compute_canonical(
        reps: list[dict[Rep, int]], perm: tuple[torch.Tensor]
    ) -> Tuple[dict[Rep, int], torch.Tensor]:
        """
        Compute the canonical order of the reps in the list of reps to sum over.

        Args:
        ----------
            counters: list of dicts of the uniqe Reps in each Rep to be summed
                      over and the number of occurances of each unique Rep
            perms:    permutation to be applied to the input Reps

        Returns:
        ----------
            reps: dict of unique Reps in the direct sum and the number of
                  occurances of each unique Rep
            perm: permutation to be applied to the collection of reps to get
                  the canonical order
        """
        pass

    @abstractmethod
    def drho(self, A: LieAlgebraElem | dict[Group, LieAlgebraElem]) -> ReprElem:
        """
        The default Rep implementation of rho -> drho will not work!
        """
        pass

    @property
    def T(self):
        """
        Swap to adjoint representation without reordering elements.
        """
        return self.__class__(
            [rep.T for rep, c in self.counters.items() for _ in range(c)], self.perm
        )

    def canonicalize(self) -> tuple["OpRep", torch.Tensor]:
        """
        Return the canonical form of the operation and the permutation it took
        to get there. The SumRep constructor takes care of canonicalizing,
        unless we specify an optional non-canonical ordering. Here we call the
        constructor again without the extra_perm argument to get the canonical
        ordering.
        """
        return self.__class__(self.counters), self.perm

    def __call__(self, G: Group) -> Rep:
        return self.__class__(
            {rep(G): c for rep, c in self.counters.items()}, self.perm
        )

    def __eq__(self, other: Rep) -> bool:
        match other:
            case self.__class__():
                return self.counters == other.counters and bool(
                    torch.all(self.perm == other.perm)
                )
        return False

    def __hash__(self) -> int:
        assert self.canonical
        return hash(tuple(self.counters.items()))

    def __iter__(self) -> Generator[Rep, None, None]:
        """
        Gets the reps in the self.counters dict. Ignores permutation order!
        """
        return (rep for rep, c in self.counters.items() for _ in range(c))


class SumRep(OpRep):
    """
    Direct sum of a list of reps.

    In order to keep track of compositions of SumReps and direct sums of
    equivalent Reps, we store a dictionary of all the unique Reps that
    show up in the merged/summed represenation, along with the number of
    occurences of each unique Rep. This dict is self.counters.

    In order to keep track of the canonical ordering of the Reps that comprise
    the direct sum, we store a torch.Tensor with the ordering of the reps,
    called self.perm. We can specify a non-canonical ordering by specifying the
    extra_perm tensor, which imposes further shuffling on the Reps. We also
    store self.perm's reverse in self.invperm to quickly revert to the
    canonical ordering.
    """

    @staticmethod
    def compute_canonical(
        counters: list[dict[Rep, int]], perms: tuple[torch.Tensor]
    ) -> tuple[dict[Rep, int], torch.Tensor]:
        """
        Compute the canonical order of the reps in the list of reps to sum over.

        Args:
        ----------
            counters: list of dicts of the uniqe Reps in each Rep to be summed
                      over and the number of occurances of each unique Rep
            perms:    permutation to be applied to the input Reps

        Returns:
        ----------
            reps: dict of unique Reps in the direct sum and the number of
                  occurances of each unique Rep
            perm: permutation to be applied to the collection of reps to get
                  the canonical order
        """

        # Compute the cannonical order of the reps that make up the new
        # summed/merged rep
        in_reps = [counter.keys() for counter in counters]
        merged_reps = reduce(lambda set1, set2: set1 | set2, in_reps)
        can_reps = sorted(merged_reps)

        # Compute the shifts for each perm in the naively summed/merged rep
        shifted_perms = []
        n = 0
        for perm in perms:
            shifted_perms.append(n + perm)
            n += len(perm)

        # Compute the canonical order of the reps
        ids = [0] * len(counters)
        permlist = []
        merged_counters = defaultdict(int)
        for rep in can_reps:
            for i in range(len(counters)):
                c = counters[i].get(rep, 0)
                new_perm = shifted_perms[i][ids[i] : ids[i] + c * rep.size]
                permlist.extend(new_perm.tolist())
                ids[i] += +c * rep.size
                merged_counters[rep] += c

        return merged_counters, torch.tensor(permlist)

    @property
    def size(self) -> int:
        return sum(rep.size * count for rep, count in self.counters.items())

    def rho(self, g: GroupElem) -> ReprElem:
        rhos = [lazify(rep.rho(g)) for rep in self.counters]
        mults = list(self.counters.values())
        return LazyPerm(self.invperm) @ LazyDirectSum(rhos, mults) @ LazyPerm(self.perm)

    def drho(self, A: LieAlgebraElem) -> ReprElem:
        drhos = [lazify(rep.drho(A)) for rep in self.counters]
        mults = list(self.counters.values())
        return (
            LazyPerm(self.invperm) @ LazyDirectSum(drhos, mults) @ LazyPerm(self.perm)
        )

    def equivariant_basis(self) -> LinearOperator:
        """
        Compute the equivariant basis for the sum of the reps, decomposing
        constraints across elements of the sum and lazifying them together.
        """
        Qs: dict[Rep, LinearOperator] = {
            rep: rep.equivariant_basis() for rep in self.counters.keys()
        }
        active_dims = sum([self.counters[rep] * Qs[rep].shape[-1] for rep in Qs.keys()])
        mults = self.counters.values()

        dtype = float
        shape = (self.size, active_dims)

        def lazy_Q(Q: torch.Tensor) -> torch.Tensor:
            return lazy_direct_matmat(Q, list(Qs.values()), list(mults))[self.invperm]

        class LazyQ(LinearOperator):
            def __init__(self):
                super().__init__(dtype=dtype, shape=shape)

            def matvec(self, v: torch.Tensor) -> torch.Tensor:
                return lazy_Q(v)

            def matmat(self, M: torch.Tensor) -> torch.Tensor:
                return lazy_Q(M)

        return LazyQ()

    def equivariant_projector(self) -> LinearOperator:
        """
        Compute the equivariant projector for the sum of the reps, decomposing
        constraints across elements of the sum and lazifying them together.
        """
        Ps: dict[Rep, LinearOperator] = {
            rep: rep.equivariant_projector() for rep in self.counters
        }
        mults = self.counters.values()

        dtype = float
        shape = (self.size, self.size)

        def lazy_P(P: torch.Tensor) -> torch.Tensor:
            return lazy_direct_matmat(P[self.perm], list(Ps.values()), list(mults))[
                self.invperm
            ]

        class LazyP(LinearOperator):
            def __init__(self):
                super().__init__(dtype=dtype, shape=shape)

            def matvec(self, v: torch.Tensor) -> torch.Tensor:
                return lazy_P(v)

            def matmat(self, M: torch.Tensor) -> torch.Tensor:
                return lazy_P(M)

        return LazyP()

    def __repr__(self) -> str:
        return "+".join(
            f"{count if count > 1 else ''}{repr(rep)}"
            for rep, count in self.counters.items()
        )

    def __len__(self) -> int:
        """
        Gets the number of reps in the sum, including duplicates.
        """
        return sum(mult for mult in self.counters.values())


def distribute_product(reps: list[Rep]) -> Rep:
    """
    Expands product of sums into sums of products,(ρ₁⊕ρ₂)⊗ρ₃ = (ρ₁⊗ρ₃)⊕(ρ₂⊗ρ₃).
    """
    can_reps, perms = zip(*[repsum.canonicalize() for repsum in reps])

    sum_reps = []
    for rep in can_reps:
        match rep:
            case SumRep():
                sum_reps.append(rep)
            case Rep():
                sum_reps.append(SumRep({rep: 1}))
    sum_reps = tuple(sum_reps)

    # Compute axis-wise permutation to canonical rep
    axis_sizes = tuple([len(perm) for perm in perms])
    # for perm in perms:
    # match perm:
    # case torch.Tensor():
    # axis_sizes.append(len(perm))
    # case _:
    # raise ValueError("Permutation must be a torch.Tensor")

    order = torch.arange(product(axis_sizes)).reshape(axis_sizes)
    for i, perm in enumerate(perms):
        order = torch.swapaxes(torch.swapaxes(order, 0, i)[perm, ...], 0, i)
    order = order.reshape(-1)

    # Compute permutation from multilinear map ordering -> vector ordering,
    # decomposing the blocks
    repsizes_all = []
    for rep in sum_reps:
        this_rep_sizes = []
        for r, c in rep.counters.items():
            this_rep_sizes.extend([c * r.size])
        repsizes_all.append(tuple(this_rep_sizes))
    block_perm = rep_permutation(tuple(repsizes_all))

    # Product order -> multiplicity grouped ordering
    ordered_reps = []
    each_perm = []
    i = 0
    for prod in itertools.product(*[rep.counters.items() for rep in sum_reps]):
        rs, cs = zip(*prod)
        prod_rep = product(cs) * product(rs)
        prod_rep, can_perm = prod_rep.canonicalize()

        ordered_reps.append(prod_rep)
        shape = []
        for r, c in prod:
            shape.extend([c, r.size])

        num = list(range(2 * len(prod)))
        evens = num[::2]
        odds = num[1::2]
        axis_perm = evens + odds
        mul_perm = torch.permute(
            torch.arange(len(can_perm)).reshape(shape), axis_perm
        ).reshape(-1)

        each_perm.append(mul_perm[can_perm] + i)
        i += len(can_perm)

    total_perm = order[block_perm[torch.cat(each_perm)]]

    return SumRep(ordered_reps, total_perm)


@lru_cache(maxsize=None)
def rep_permutation(repsizes_all) -> torch.Tensor:
    """
    Permutation from block ordering to flattened ordering.
    """
    size_cumsums = []
    for repsizes in repsizes_all:
        padded_sizes = torch.tensor([0] + [size for size in repsizes])
        # size_cumsums.append(torch.cumsum(torch.flatten(padded_sizes), 0))
        size_cumsums.append(torch.cumsum(padded_sizes, 0))
    permutation = torch.zeros([cumsum[-1] for cumsum in size_cumsums], dtype=torch.int)
    arange = torch.arange(reduce(lambda x, y: x * y, permutation.size()))
    indices_iter = itertools.product(
        *[range(len(repsizes)) for repsizes in repsizes_all]
    )

    i = 0
    for indices in indices_iter:
        slices = tuple(
            [
                slice(cumsum[idx], cumsum[idx + 1])
                for idx, cumsum in zip(indices, size_cumsums)
            ]
        )
        slice_lengths = [int(sl.stop - sl.start) for sl in slices]
        chunk_size = product(slice_lengths)
        permutation[slices] += arange[i : i + chunk_size].reshape(*slice_lengths)
        i += chunk_size

    return torch.argsort(permutation.reshape(-1))


class ProductRep(OpRep):
    """
    Tensor product of a list of reps.

    In order to keep track of compositions of ProductReps and tensor products
    of equivalent Reps, we store a dictionary of all the unique Reps that show
    up in the merged/summed represenation, along with the number of occurences
    of each unique Rep. This dict is self.counters.

    In order to keep track of the canonical ordering of the Reps that comprise
    the tensor product, we store a torch.Tensor with the ordering of the reps,
    called self.perm. We can specify a non-canonical ordering by specifying the
    extra_perm tensor, which imposes further shuffling on the Reps. We also
    store self.perm's reverse in self.invperm to quickly revert to the
    canonical ordering.
    """

    def __init__(
        self,
        inreps: dict[Rep, int] | list[Rep],
        extra_perm: Optional[torch.Tensor] = None,
    ):
        super().__init__(inreps, extra_perm)
        assert self.G is not None, "Can only take products of reps of the same group"

    @staticmethod
    def compute_canonical(
        counters: list[dict[Rep, int]], perms: tuple[torch.Tensor]
    ) -> tuple[dict[Rep, int], torch.Tensor]:
        """
        Compute the canonical order of the reps in the list of reps to multiply
        over.

        Args:
        ----------
            counters: list of dicts of the unique Reps in each Rep to be
                      multiplied over and the number of occurances of each
                      unique Rep
            perms:    permutation to be applied to the input Reps

        Returns:
        ----------
            reps: dict of unique Reps in the direct sum and the number of
                  occurances of each unique Rep
            perm: pemutation to be applied to the collection of reps to get the
                  canonical order
        """
        order = torch.arange(product(len(perm) for perm in perms)).reshape(
            tuple(len(perm) for perm in perms)
        )
        in_reps = [counter.keys() for counter in counters]
        merged_reps = reduce(lambda set1, set2: set1 | set2, in_reps)
        can_reps = sorted(merged_reps)

        # Canonicalize along each axis
        for i, perm in enumerate(perms):
            order = torch.moveaxis(torch.moveaxis(order, i, 0)[perm, ...], 0, i)

        # Get original axis ids
        ids = []
        n = 0
        for counter in counters:
            idsi = {}
            for rep, c in counter.items():
                idsi[rep] = n + torch.arange(c)
                n += c
            ids.append(idsi)

        # Compute the canonical order of the reps
        permlist = []
        merged_counters = defaultdict(int)
        for rep in can_reps:
            for i in range(len(perms)):
                c = counters[i].get(rep, 0)
                if c != 0:
                    permlist.extend(ids[i][rep].tolist())
                    merged_counters[rep] += c

        order = order.reshape(
            tuple(
                rep.size
                for counter in counters
                for rep, c in counter.items()
                for _ in range(c)
            )
        )

        order = torch.permute(order, permlist).reshape(-1)
        return merged_counters, order

    @property
    def size(self) -> int:
        return product([rep.size**count for rep, count in self.counters.items()])

    def rho(self, g: GroupElem | dict[Group, GroupElem]) -> ReprElem:
        match g:
            case dict():
                if self.G is not None:
                    gg = g[self.G]
                raise ValueError(
                    "Must specify a GroupElem for ProductReps without a group"
                )
            case _:
                gg = g

        rho_can = LazyKron(
            [lazify(rep.rho(gg)) for rep, c in self.counters.items() for _ in range(c)]
        )
        return LazyPerm(self.invperm) @ rho_can @ LazyPerm(self.perm)

    def drho(self, A: LieAlgebraElem | dict[Group, LieAlgebraElem]) -> ReprElem:
        match A:
            case dict():
                if self.G is not None:
                    AA = A[self.G]
                raise ValueError(
                    "Must specify a LieGroupElem for ProductReps without a group"
                )
            case _:
                AA = A
        drho_can = LazyKronsum(
            [lazify(rep.drho(AA)) for rep, c in self.counters.items() for _ in range(c)]
        )
        return LazyPerm(self.invperm) @ drho_can @ LazyPerm(self.perm)

    def __repr__(self):
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "⊗".join(
            [
                str(rep) + (f"{c}".translate(superscript) if c > 1 else "")
                for rep, c in self.counters.items()
            ]
        )


class DirectProduct(ProductRep):
    """
    Direct product of a list of reps.

    This is a generalization of the ProductRep where ρ=ρ₁⊗ρ₂ is a
    representation of G=G₁×G₂. In this case, the solutions for the
    sub-representations can be solved independently and reassembled via the
    Kronecker product, Q = Q₁⊗Q₂ and P = P₁⊗P₂.
    """

    def __init__(
        self,
        reps: dict[Rep, int] | list[Rep],
        extra_perm: Optional[torch.Tensor] = None,
    ):
        OpRep.__init__(self, reps, extra_perm)  # don't inherit from ProductRep
        assert all(count == 1 for count in self.counters.values())

    def rho(self, g: GroupElem) -> ReprElem:
        canonical_lazy = LazyKron(
            [lazify(rep.rho(g)) for rep, c in self.counters.items() for _ in range(c)]
        )
        return LazyPerm(self.invperm) @ canonical_lazy @ LazyPerm(self.perm)

    def drho(self, A: LieAlgebraElem) -> ReprElem:
        canonical_lazy = LazyKronsum(
            [lazify(rep.drho(A)) for rep, c in self.counters.items() for _ in range(c)]
        )
        return LazyPerm(self.invperm) @ canonical_lazy @ LazyPerm(self.perm)

    def equivariant_basis(self) -> LinearOperator:
        canon_Q = LazyKron([rep.equivariant_basis() for rep in self.counters.keys()])
        return lazify(LazyPerm(self.invperm) @ canon_Q)

    def equivariant_projector(self) -> LinearOperator:
        canon_P = LazyKron([p.equivariant_projector() for p in self.counters.keys()])
        return lazify(LazyPerm(self.invperm) @ canon_P @ LazyPerm(self.perm))

    def __repr__(self) -> str:
        return "⊗".join([str(rep) + f"_{rep.G}" for rep in self.counters.keys()])


class DeferredOpRep(Rep, ABC):
    """
    Abstract base class for placeholders of OpReps on an unspecified group. As
    soon as a DeferredOpRep is called on a group that isn't none, it's
    converted to an OpRep.

    The list of Reps to OpRep together is saved in self.to_op.
    """

    def __init__(self, reps: list[Rep]):
        self.G = None
        self.to_op = []

        for rep in reps:
            match rep:
                case self.__class__():
                    self.to_op.extend(rep.to_op)
                case _:
                    self.to_op.append(rep)

        # Ensure hashable
        self.to_op = tuple(self.to_op)

    @abstractmethod
    def __call__(self, G: Optional[Group]) -> Rep:
        """
        This is where you implement the conversion from DeferredOpRep -> OpRep
        """
        pass

    @property
    def size(self) -> int:
        return sum(rep.size for rep in self.to_op)

    @property
    def T(self) -> Rep:
        return self.__class__([rep.T for rep in self.to_op])


class DeferredSumRep(DeferredOpRep):
    """
    SumRep of an unspecified group.
    """

    def rho(self, M: GroupElem) -> ReprElem:
        rhos = [lazify(rep.rho(M)) for rep in self.to_op]
        mults = len(self.to_op) * [1]
        return LazyDirectSum(rhos, mults)  # faster way?

    def drho(self, A: LieAlgebraElem) -> ReprElem:
        drhos = [lazify(rep.drho(A)) for rep in self.to_op]
        mults = len(self.to_op) * [1]
        return LazyDirectSum(drhos, mults)  # faster way?

    def __call__(self, G: Optional[Group]) -> Rep:
        if G is None:
            # return self.__class__(self.to_op)
            return self
        return SumRep([rep(G) for rep in self.to_op])

    def __repr__(self):
        return "(" + "+".join(f"{rep}" for rep in self.to_op) + ")"


class DeferredProductRep(DeferredOpRep):
    """
    ProductRep of an unspecified group.
    """

    def rho(self, g: GroupElem) -> ReprElem:
        return LazyKron([lazify(rep.rho(g)) for rep in self.to_op])

    def drho(self, A: LieAlgebraElem) -> ReprElem:
        return LazyKronsum([lazify(rep.drho(A)) for rep in self.to_op])

    def __call__(self, G: Optional[Group]) -> Rep:
        if G is None:
            # return self.__class__(self.to_op)
            return self
        return reduce(lambda a, b: a * b, [rep(G) for rep in self.to_op])

    def __repr__(self):
        return "⊗".join(f"{rep}" for rep in self.to_op)


class ScalarRep(Rep):
    def __init__(self, G: Optional[Group] = None):
        self.G = G
        self.is_permutation = True

    def __call__(self, G: Optional[Group]) -> Rep:
        self.G = G
        return self
        # TODO CHECK THAT ITS OK TO RETURN A NEW REPRESENTATION, NOT A COPY
        # return ScalarRep(G)

    @property
    def size(self) -> int:
        return 1

    def __repr__(self) -> str:
        return "V⁰"

    def rho(self, g: GroupElem) -> ReprElem:
        return torch.eye(1)

    def drho(self, A: LieAlgebraElem) -> ReprElem:
        return 0 * torch.eye(1)

    def constraint_matrix(self) -> LinearOperator:
        raise NotImplementedError

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

    def __rmul__(self, other: Rep | int) -> Rep:
        match other:
            case Rep():
                return other
            case int():
                return super().__rmul__(other)

    @property
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
    def size(self) -> int:
        if self.G is None:
            return 0
        # return self.rho(self.G.sample()).shape[-1]
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


def T(p: int, q: int = 0, G: Optional[Group] = None) -> Rep:
    """
    Convenience function for creating rank (p, q) tensors
    """
    match G:
        case Group():
            return (V**p * V.T**q)(G)
    return V**p * V.T**q
