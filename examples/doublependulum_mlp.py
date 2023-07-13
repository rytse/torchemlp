import sys

sys.path.append("../")

import torch
import torch.utils as utils

import pytorch_lightning as pl

from torchemlp.utils import DEFAULT_DEVICE, DEFAULT_DEVICE_STR
from torchemlp.groups import SO, O, S, Z
from torchemlp.nn.utils import MLP, AutonomousWrapper
from torchemlp.nn.runners import (
    FuncMSERegressionLightning,
    DynamicsL2RegressionLightning,
)
from torchemlp.nn.utils import Standardize
from torchemlp.datasets import DoublePendulum

# torch.set_default_dtype(torch.float32)

TRAINING_SET_SIZE = 5_000
VALIDATION_SET_SIZE = 1_000
TEST_SET_SIZE = 1_000

DT = 0.1
T = 30.0

BATCH_SIZE = 500

N_EPOCHS = 5  # min(int(900_000 / TRAINING_SET_SIZE), 1000)
# N_EPOCHS = 100  # min(int(900_000 / TRAINING_SET_SIZE), 1000)

N_CHANNELS = 384
N_LAYERS = 3

DL_WORKERS = 0

dataset = DoublePendulum(
    TRAINING_SET_SIZE + VALIDATION_SET_SIZE + TEST_SET_SIZE,
    DT,
    T,
)

print(f"Loaded dataset.")
print(f"Repin: {dataset.repin}")
print(f"Repout: {dataset.repout}")

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

model = AutonomousWrapper(
    Standardize(MLP(12, 12, 8, 4), dataset.stats).to(DEFAULT_DEVICE)
)
plmodel = DynamicsL2RegressionLightning(model)

trainer = pl.Trainer(
    limit_train_batches=BATCH_SIZE,
    max_epochs=N_EPOCHS,
    accelerator=DEFAULT_DEVICE_STR,
)
trainer.fit(plmodel, train_loader, val_loader)
trainer.test(plmodel, test_loader)
