from functools import reduce
from functools import lru_cache as cache
from typing import Union, Optional, List, Dict

import torch

from reps_base import Rep, ScalarRep, Scalar

from torchemlp.utils import is_scalar
from torchemlp.ops import (
    LinearOperator,
    LazyPerm,
    LazyDirectSum,
    LazyKron,
    LazyKronsum,
    I,
    lazy_direct_matmat,
    lazify,
    product,
)


class SumRep(Rep):
    """
    Direct sum of a list of reps.
    """

    def __init__(self, *reps: list[Rep | int], extra_perm: Optional[dict] = None):
        """
        Compute the direct sum of a list of reps and apply any extra
        permutations specified.

        Args:
        ----------
            reps: list of reps, either of Rep type or of int to be converted
            extra_perm: extra permutation to be applied to the reps
        """

        # Fill in int -> Rep conversions
        sumreps = []
        for rep in reps:
            match rep:
                case Rep():
                    sumreps.append(rep)
                case int():
                    sumreps.append(SumRepFromCollection({Scalar: rep}))

        # Canonicalize each rep to be summed
        cansumreps, perms = zip(*[rep.canonicalize() for rep in sumreps])

        rep_counters = []
        for rep in cansumreps:
            match rep:
                case SumRep():
                    rep_counters.append(rep.reps)
                case Rep():
                    rep_counters.append({rep: 1})

        self.reps, perm = self.compute_canonical(rep_counters, perms)
        self.perm = extra_perm[perm] if extra_perm is not None else perm
        self.invperm = torch.argsort(self.perm)

        self.canonical = torch.all(self.perm == torch.arange(len(self.perm)))
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())


class SumRepFromCollection(Rep):
    def __init__(self, counter: dict[Rep, int], perm: Optional[torch.Tensor] = None):
        """
        Construct a tensor type from the sum of a list of tensors of given ranks.
        """
        self.reps = counter
        self.perm = perm if perm is not None else torch.arange(len(self.reps))
        self.reps, self.perm = self.compute_canonical([counter], [self.perm])
