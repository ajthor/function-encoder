from typing import List, Callable, Union
import torch


class MLP(torch.nn.Module):
    """A simple multi-layer perceptron neural network.

    Args:
        layer_sizes (List[int]): List of layer sizes, including input and output dimensions
        activation (Callable, optional): Activation function to use between layers. Defaults to torch.nn.Tanh().
        bias (bool, optional): Whether to include bias in linear layers. Defaults to True.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: Union[torch.nn.Module, Callable] = torch.nn.Tanh(),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor [batch_size, ...]

        Returns:
            torch.Tensor: Output tensor [batch_size, ...]
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


class MultiHeadedMLP(torch.nn.Module):
    """A multi-headed MLP that outputs multiple vectors in parallel.

    The network shares parameters for all layers except the final one,
    which produces multiple outputs (heads) simultaneously.

    Args:
        layer_sizes (List[int]): List of layer sizes, including input and output dimensions
        num_heads (int): Number of output heads to produce
        activation (Callable, optional): Activation function to use between layers. Defaults to torch.nn.Tanh().
        bias (bool, optional): Whether to include bias in linear layers. Defaults to True.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        num_heads: int,
        activation: Union[torch.nn.Module, Callable] = torch.nn.Tanh(),
        bias: bool = True,
    ):
        super(MultiHeadedMLP, self).__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            self.layers.append(
                torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias),
            )
        self.layers.append(
            torch.nn.Linear(layer_sizes[-2], layer_sizes[-1] * num_heads, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the multi-headed MLP.

        Args:
            x (torch.Tensor): Input tensor [batch_size, ...]

        Returns:
            torch.Tensor: Output tensor [batch_size, ..., num_heads]
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = x.view(*x.shape[:-1], self.layer_sizes[-1], -1)
        return x
