import time

import torch
import torch.utils as utils

import pytorch_lightning as pl

from torchemlp.groups import SO, O, S, Z
from torchemlp.nn.equivariant import EMLP
from torchemlp.nn.runners import FuncMSERegressionLightning
from torchemlp.datasets import Inertia

torch.set_float32_matmul_precision("medium")

if torch.cuda.is_available():
    dev = "cuda:0"
    accelerator = "gpu"
else:
    dev = "cpu"
    accelerator = None

device = torch.device(dev)


# TRAINING_SET_SIZE = 1000
TRAINING_SET_SIZE = 10_000
BATCH_SIZE = 512

# N_EPOCHS = int(900000 / TRAINING_SET_SIZE)
# N_EPOCHS = 200
N_EPOCHS = 64

DL_WORKERS = 0
# DL_WORKERS = 8

N_CHANNELS = 384
N_LAYERS = 3


dataset = Inertia(TRAINING_SET_SIZE, device=device)

G = SO(3)

f"Input type: {dataset.repin(G)}, output type: {dataset.repout(G)}"

val_size = 250
test_size = 250
train_size = len(dataset) - val_size - test_size

split_data = utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = utils.data.DataLoader(
    split_data[0], batch_size=BATCH_SIZE, num_workers=DL_WORKERS, shuffle=True
)
val_loader = utils.data.DataLoader(
    split_data[1], batch_size=BATCH_SIZE, num_workers=DL_WORKERS
)
test_loader = utils.data.DataLoader(
    split_data[2], batch_size=BATCH_SIZE, num_workers=DL_WORKERS
)

model = EMLP(dataset.repin, dataset.repout, G, N_CHANNELS, N_LAYERS).to(device)
plmodel = FuncMSERegressionLightning(model)

trainer = pl.Trainer(
    limit_train_batches=BATCH_SIZE,
    max_epochs=N_EPOCHS,
    accelerator=accelerator,
)

st = time.time()
trainer.fit(plmodel, train_loader, val_loader)
print(f"\n\nTrain time: {time.time() - st} s\n\n")
trainer.test(plmodel, test_loader)
