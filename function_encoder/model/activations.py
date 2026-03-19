import math
import torch

from function_encoder.model.tensor_layers import ParallelLinear, TensorLinear


class Sine(torch.nn.Module):
    """Sine activation with configurable frequency."""

    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)

    def extra_repr(self) -> str:
        return f"omega_0={self.omega_0}"


def init_siren(
    layer: torch.nn.Module,
    layer_idx: int,
    omega_0: float = 30.0,
) -> torch.nn.Module:
    if isinstance(layer, (torch.nn.Linear, TensorLinear, ParallelLinear)):
        in_dim = layer.weight.shape[-1]
    else:
        raise TypeError(
            "init_siren only supports torch.nn.Linear, TensorLinear, and "
            f"ParallelLinear; got {type(layer).__name__}"
        )

    if layer_idx == 0:
        bound = 1.0 / in_dim
    else:
        bound = math.sqrt(6.0 / in_dim) / omega_0

    torch.nn.init.uniform_(layer.weight, -bound, bound)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)
    return layer
