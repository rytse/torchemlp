import sys

sys.path.append("../")

import torch

from torchemlp.groups import SO, O, S, Z
from torchemlp.nn.equivariant import EMLP
from torchemlp.datasets import Inertia

trainset = Inertia(1000)  # Initialize dataset with 1000 examples
testset = Inertia(2000)

G = SO(3)

f"Input type: {trainset.repin(G)}, output type: {trainset.repout(G)}"

model = EMLP(trainset.repin, trainset.repout, G, 384, 3)
