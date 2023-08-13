from typing import Callable
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint

from torchemlp.nn.contdepth import hamiltonian_dynamics

import pytorch_lightning as pl


class RegressionLightning(ABC, pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 3e-3, weight_decay: float = 0.0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    @abstractmethod
    def batch2loss(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        pass

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        loss = self.batch2loss(batch)
        self.log("train_loss", loss)

        return loss

    def test_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        loss = self.batch2loss(batch)
        self.log("test_loss", loss.item())

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        loss = self.batch2loss(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


class FuncMSERegressionLightning(RegressionLightning):
    def batch2loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        y_pred = self.model(x)
        return F.mse_loss(y_pred, y)


class DynamicsL2RegressionLightning(RegressionLightning):
    def __init__(
        self,
        model: nn.Module,
        odeint_fn: Callable = odeint,
        lr: float = 3e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__(model, lr, weight_decay)
        self.odeint_fn = odeint_fn

    def batch2loss(
        self, batch: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        (z0, ts), zs = batch

        # Force them to have grad so that we can evaluate dynamics
        z0 = z0.requires_grad_(True)
        zs = zs.requires_grad_(True)
        ts = ts.requires_grad_(True)

        zs_pred = self.odeint_fn(
            self.model,
            z0,
            ts[0, ...],
            options={"dtype": torch.float32},
        )
        zs_pred = torch.swapaxes(zs_pred, 0, 1)

        return F.mse_loss(zs_pred, zs)
