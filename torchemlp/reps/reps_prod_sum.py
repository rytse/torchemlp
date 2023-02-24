from functools import reduce
from functools import lru_cache as cache

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
    def __init__(self, *reps, extra_perm=None):
        """
        Construct a tensor type from the sum of a list of tensors of given ranks.
        """

        reps = [
            SumRepFromCollection({Scalar: rep}) if is_scalar(rep) else rep
            for rep in reps
        ]
        reps, perms = zip(*[rep.canonicalize() for rep in reps])
        reps_counters = [
            rep.reps if isinstance(rep, SumRep) else {rep: 1} for rep in reps
        ]
