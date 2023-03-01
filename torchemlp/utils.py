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

# Representation of a group element as a tensor or linear operator.
# Has the same python type as GroupElem(s) and LieAlgebraElem(s), but we define
# them differently so that our methods' type signatures are more helpful.
ReprElem = Union[torch.Tensor, LinearOperator]


def merge_torch_types(dtype1, dtype2):
    return (torch.ones(1, dtype=dtype1) * torch.ones(1, dtype=dtype2)).dtype
