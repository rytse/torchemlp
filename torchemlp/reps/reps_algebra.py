from abc import ABC, abstractmethod
from functools import reduce, lru_cache
import itertools
from typing import Optional, Tuple, Generator
from collections import defaultdict

import torch

from reps_base import Rep

from torchemlp.groups import Group
from torchemlp.utils import GroupElem, LieAlgebraElem, ReprElem

from torchemlp.ops import (
    LinearOperator,
    LazyPerm,
    LazyDirectSum,
    LazyKron,
    LazyKronsum,
    lazy_direct_matmat,
    lazify,
    product,
)


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
                    [reps], perm
                )

            case list():
                # Canonicalize each rep to be multiplied
                canreps, perms = zip(*[rep.canonicalize() for rep in reps])
                if not isinstance(perms, torch.Tensor):  # comfort the type checker
                    raise ValueError("Permutation must be a torch.Tensor")

                # Dict containing the set of unique Reps in the multiplication
                # and the number of occurances of each unique Rep
                in_counters = []
                for rep in canreps:
                    match rep:
                        case ProductRep():
                            in_counters.append(rep.counters)
                        case Rep():
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

        self.canonical = torch.all(self.perm == self.invperm)
        self.is_permutation = all(rep.is_permutation for rep in self.counters.keys())

    @abstractmethod
    @staticmethod
    def compute_canonical(
        reps: list[dict[Rep, int]], perm: torch.Tensor
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
            case OpRep():
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
        counters: list[dict[Rep, int]], perms: torch.Tensor
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
                permlist.append(shifted_perms[i][ids[i] : ids[i] + c * rep.size])
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
            rep: rep.equivariant_projector() for rep in self.counters.keys()
        }
        mults = self.counters.values()

        dtype = float
        shape = (self.size, mults)

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

    def __mul__(self, other: Rep) -> Rep:
        """
        Expand product of sums into sum of products for efficient tensor
        product of direct sum.
        """
        return distribute_product([self, other])

    def __rmul__(self, other: Rep) -> Rep:
        """
        Expand product of sums into sum of products for efficient tensor
        product of direct sum.
        """
        return distribute_product([other, self])

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

    # Compute axis-wise permutation to canonical rep
    axis_sizes = []
    for perm in perms:
        match perm:
            case torch.Tensor():
                axis_sizes.append(len(perm))
            case _:
                raise ValueError("Permutation must be a torch.Tensor")
    order = torch.arange(product(axis_sizes)).reshape(tuple(axis_sizes))
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
    for prod in itertools.product(*[rep.reps.items() for rep in sum_reps]):
        rs, cs = zip(*prod)
        prod_rep, canonicalizing_perm = (product(cs) * product(rs)).canonicalize()
        ordered_reps.append(prod_rep)
        shape = []
        for r, c in prod:
            shape.extend([c * r.size])

        num = list(range(0, 2 * len(prod)))
        evens = num[::2]
        odds = num[1::2]
        axis_perm = evens + odds
        mul_perm = torch.permute(
            torch.arange(len(canonicalizing_perm)).reshape(shape), axis_perm
        ).reshape(-1)

        each_perm.append(mul_perm[canonicalizing_perm] + i)
        i += len(canonicalizing_perm)

    each_perm = torch.tensor(each_perm)
    total_perm = order[block_perm[each_perm]]

    return SumRep(ordered_reps, total_perm)


@lru_cache(maxsize=None)
def rep_permutation(repsizes_all) -> torch.Tensor:
    """
    Permutation from block ordering to flattened ordering.
    """
    size_cumsums = []
    for repsizes in repsizes_all:
        padded_sizes = torch.tensor([0] + [size for size in repsizes])
        size_cumsums.append(torch.cumsum(torch.flatten(padded_sizes), 0))
    permutation = torch.zeros([cumsum[-1] for cumsum in size_cumsums], dtype=torch.int)
    arange = torch.arange(len(repsizes_all))
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
        slice_lengths = [sl.stop - sl.start for sl in slices]
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

        Gs = tuple(set(rep.G for rep in self.counters.keys()))
        assert len(Gs) == 1, "Multiple different groups in product rep"
        self.G = Gs[0]

    @staticmethod
    def compute_canonical(
        counters: list[dict[Rep, int]], perms: torch.Tensor
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
                    permlist.append(ids[i][rep])
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
        self.to_op: list[Rep] = []

        for rep in reps:
            match rep:
                case DeferredOpRep():
                    self.to_op.extend(rep.to_op)
                case _:
                    self.to_op.append(rep)

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
            return self.__class__(self.to_op)
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
            return self.__class__(self.to_op)
        return reduce(lambda a, b: a * b, [rep(G) for rep in self.to_op])

    def __repr__(self):
        return "⊗".join(f"{rep}" for rep in self.to_op)
