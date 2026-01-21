from typing import List, Callable, Union
import math
import torch


class MLP(torch.nn.Module):
    """A simple multi-layer perceptron neural network.

    Args:
        layer_sizes (List[int]): List of layer sizes, including input and output dimensions
        activation (Callable, optional): Activation function to use between layers. Defaults to torch.nn.ReLU().
        bias (bool, optional): Whether to include bias in linear layers. Defaults to True.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: Union[torch.nn.Module, Callable] = torch.nn.ReLU(),
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
        activation (Callable, optional): Activation function to use between layers. Defaults to torch.nn.ReLU().
        bias (bool, optional): Whether to include bias in linear layers. Defaults to True.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        num_heads: int,
        activation: Union[torch.nn.Module, Callable] = torch.nn.ReLU(),
        bias: bool = True,
    ):
        super(MultiHeadedMLP, self).__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation
        trunk_layers = []
        for i in range(len(layer_sizes) - 2):
            trunk_layers.append(
                torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias),
            )
        self.trunk = torch.nn.Sequential(*trunk_layers)
        self.head = torch.nn.Linear(
            layer_sizes[-2], layer_sizes[-1] * num_heads, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the multi-headed MLP.

        Args:
            x (torch.Tensor): Input tensor [batch_size, ...]

        Returns:
            torch.Tensor: Output tensor [batch_size, ..., num_heads]
        """
        for layer in self.trunk:
            x = layer(x)
            x = self.activation(x)
        x = self.head(x)
        x = x.view(*x.shape[:-1], self.layer_sizes[-1], -1)
        return x


class StackedMLP(torch.nn.Module):
    """A stacked MLP with independent parameters per head.

    The network has separate parameters per head for every layer, but computes
    all heads in a single tensorized forward pass for efficiency.

    Args:
        layer_sizes (List[int]): List of layer sizes, including input and output dimensions
        num_heads (int): Number of output heads to produce
        activation (Callable, optional): Activation function to use between layers. Defaults to torch.nn.ReLU().
        bias (bool, optional): Whether to include bias in linear layers. Defaults to True.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        num_heads: int,
        activation: Union[torch.nn.Module, Callable] = torch.nn.ReLU(),
        bias: bool = True,
    ):
        super(StackedMLP, self).__init__()
        self.layer_sizes = layer_sizes
        self.num_heads = num_heads
        self.activation = activation
        self.bias = bias
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            weight = torch.nn.Parameter(
                torch.empty(num_heads, out_dim, in_dim),
            )
            self.weights.append(weight)
            if bias:
                self.biases.append(torch.nn.Parameter(torch.empty(num_heads, out_dim)))
            else:
                self.biases.append(torch.nn.Parameter(torch.empty(0)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i, weight in enumerate(self.weights):
            in_dim = weight.shape[-1]
            for h in range(self.num_heads):
                torch.nn.init.kaiming_uniform_(weight[h], a=math.sqrt(5))
            if self.bias:
                bound = 1 / math.sqrt(in_dim) if in_dim > 0 else 0.0
                torch.nn.init.uniform_(self.biases[i], -bound, bound)

    @staticmethod
    def _batched_linear(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        x = torch.einsum("...hi,hoi->...ho", x, weight)
        if bias.numel() > 0:
            x = x + bias
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the stacked MLP.

        Args:
            x (torch.Tensor): Input tensor [batch_size, ..., in_features]

        Returns:
            torch.Tensor: Output tensor [batch_size, ..., out_features, num_heads]
        """
        x = x.unsqueeze(-2)
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = self._batched_linear(x, weight, bias)
            x = self.activation(x)
        x = self._batched_linear(x, self.weights[-1], self.biases[-1])
        x = x.transpose(-1, -2)
        return x
