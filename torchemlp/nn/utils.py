from typing import Callable

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Vanilla multi-layer perceptron
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        width: int,
        depth: int,
        act: Callable = nn.PReLU,
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

        layers = []
        layers.append(nn.Linear(dim_in, width))
        layers.append(act())
        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(act)
        layers.append(nn.Linear(width, dim_out))

        self.network = nn.Sequential(*layers)

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
