import sys

sys.path.append("../")

import torch
import torch.utils as utils

import pytorch_lightning as pl

from torchemlp.utils import DEFAULT_DEVICE
from torchemlp.groups import SO, O, S, Z
from torchemlp.nn.equivariant import EMLP
from torchemlp.nn.runners import FuncMSERegressionLightning
from torchemlp.nn.utils import Standardize
from torchemlp.datasets import DoublePendulum

TRAINING_SET_SIZE = 5_000
VALIDATION_SET_SIZE = 1_000
TEST_SET_SIZE = 1_000
# TRAINING_SET_SIZE = 100
# VALIDATION_SET_SIZE = 1
# TEST_SET_SIZE = 1

CHUNK_LEN = 30
DT = 0.1
T = 30

BATCH_SIZE = 500

N_EPOCHS = 5  # min(int(900_000 / TRAINING_SET_SIZE), 1000)
# N_EPOCHS = 100  # min(int(900_000 / TRAINING_SET_SIZE), 1000)

N_CHANNELS = 384
N_LAYERS = 3

DL_WORKERS = 0

dataset = DoublePendulum(TRAINING_SET_SIZE + VALIDATION_SET_SIZE + TEST_SET_SIZE, DT, T)
print(
    f"Input type: {dataset.repin(dataset.G)}, output type: {dataset.repout(dataset.G)}"
)

split_data = utils.data.random_split(
    dataset, [TRAINING_SET_SIZE, VALIDATION_SET_SIZE, TEST_SET_SIZE]
)

train_loader = utils.data.DataLoader(
    split_data[0], batch_size=BATCH_SIZE, num_workers=DL_WORKERS, shuffle=True
)
val_loader = utils.data.DataLoader(
    split_data[1], batch_size=BATCH_SIZE, num_workers=DL_WORKERS
)
test_loader = utils.data.DataLoader(
    split_data[2], batch_size=BATCH_SIZE, num_workers=DL_WORKERS
)
