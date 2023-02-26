from typing import Union, List

import torch

from torchemlp.ops import LinearOperator

# Elements of a group embedded as a tensor or linear operator.
# Has the same python type as LieAlgebraElem(s) and ReprElem(s), but we define
# them differently so that our methods' type signatures are more helpful.
GroupElem = Union[torch.Tensor, LinearOperator]
GroupElems = Union[torch.Tensor, List[LinearOperator]]

# Elements of a group's Lie algebra embedded as a tensor or linear operator.
# Has the same python type as GroupElem(s) and ReprElem(s), but we define them
# differently so that our methods' type signatures are more helpful.
LieAlgebraElem = Union[torch.Tensor, LinearOperator]
LieAlgebraElems = Union[torch.Tensor, List[LinearOperator]]

# Representation of a group element as a tensor or linear operator.
# Has the same python type as GroupElem(s) and LieAlgebraElem(s), but we define
# them differently so that our methods' type signatures are more helpful.
ReprElem = Union[torch.Tensor, LinearOperator]
ReprElems = Union[torch.Tensor, List[LinearOperator]]


def is_scalar(x):
    return (
        (isinstance(x, torch.Tensor) and x.dim() == 0)
        or isinstance(x, float)
        or isinstance(x, int)
    )


def is_vector(x):
    return isinstance(x, torch.Tensor) and len(x.shape) == 1


def is_matrix(x):
    return isinstance(x, torch.Tensor) and len(x.shape) == 2
