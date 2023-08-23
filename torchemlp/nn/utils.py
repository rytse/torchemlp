from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl


class MLP(nn.Module):
    """
    Vanilla multi-layer perceptron
    """

    def __init__(
        self, dim_in: int, dim_out: int, width: int, depth: int, act: Callable = nn.SiLU
    ):
        """
        Args:
            dim_in: input dimension
            dim_out: output dimension
            width: width of each hidden layer
            depth: number of hidden layers
            act: activation function
        """
        super().__init__()
        self.act = act

        self.layers = []
        self.layers.append(nn.Linear(dim_in, width, bias=False))
        self.layers.append(self.act())
        for _ in range(depth):
            self.layers.append(nn.Linear(width, width, bias=False))
            self.layers.append(self.act())
        self.layers.append(nn.Linear(width, dim_out, bias=False))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Standardize(nn.Module):
    """
    Convenience module for normalizing a given module.

    Normalizes the module's input by its dataset's input mean and std.
    Un-normalizes the module's output by its dataset's output mean and std.
    """

    def __init__(self, model: nn.Module, ds_stats: list[float]):
        """
        Args:
            model: module to be normalized
            ds_stats: dataset's input and output mean and std
        """
        super(Standardize, self).__init__()

        self.model = model

        if len(ds_stats) == 2:
            self.muin, self.stdin = ds_stats
            self.muout, self.stdout = 0.0, 1.0
        elif len(ds_stats) == 4:
            self.muin, self.stdin, self.muout, self.stdout = ds_stats
        else:
            raise ValueError("ds_stats must be a list of length 2 or 4")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_in = (x - self.muin) / self.stdin
        y = self.model(normalized_in)
        unnormalized_out = self.stdout * y + self.muout
        return unnormalized_out


class RegressionLightning(pl.LightningModule):
    def __init__(self, model, lr=3e-3, weight_decay=0.0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.mse_loss(y_pred, y)
        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.mse_loss(y_pred, y)
        self.log("test_loss", loss.item())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.mse_loss(y_pred, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
