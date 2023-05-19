import sys

sys.path.append("../")

import torch
import torch.utils as utils

import pytorch_lightning as pl

from torchemlp.groups import SO, O, S, Z
from torchemlp.nn.equivariant import EMLP
from torchemlp.nn.runners import RegressionLightning
from torchemlp.nn.utils import Standardize
from torchemlp.datasets import O5Synthetic
