from typing import List, Callable

import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        activation: Callable = torch.nn.Tanh(),
        bias: bool = True,
    ):
        super(MLP, self).__init__()

        self.layer_sizes = layer_sizes
        self.activation = activation

        self.layers = torch.nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias),
            )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)

        return x
