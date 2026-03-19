from typing import Sequence
import math
import torch


class TensorLinear(torch.nn.Module):
    """Affine map from a vector input to a tensor output.

    Args:
        in_features: Size of the input feature dimension.
        out_shape: Trailing output shape produced by the layer.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_shape: Sequence[int],
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_shape = tuple(out_shape)

        if len(self.out_shape) == 0:
            raise ValueError("out_shape must contain at least one dimension")
        if any(dim <= 0 for dim in self.out_shape):
            raise ValueError("out_shape dimensions must be positive")

        self.weight = torch.nn.Parameter(torch.empty(*self.out_shape, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(*self.out_shape))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        flat_weight = self.weight.view(-1, self.in_features)
        torch.nn.init.kaiming_uniform_(flat_weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                "Expected the last dimension of x to match in_features; "
                f"got {x.shape[-1]} and {self.in_features}"
            )
        y = torch.tensordot(x, self.weight, dims=([-1], [-1]))
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        bias = self.bias is not None
        return (
            f"in_features={self.in_features}, "
            f"out_shape={self.out_shape}, bias={bias}"
        )


class ParallelLinear(torch.nn.Module):
    """Parallel bank of affine maps applied independently along one tensor axis.

    Args:
        num_tensors: Number of independent affine maps.
        in_features: Size of the input feature dimension for each tensor slice.
        out_features: Size of the output feature dimension for each tensor slice.
        bias: Whether to include a bias term for each affine map.

    Shape:
        - Input: ``[..., in_features]`` or ``[..., in_features, num_tensors]``
        - Output: ``[..., out_features, num_tensors]``
    """

    def __init__(
        self,
        num_tensors: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        if num_tensors <= 0:
            raise ValueError("num_tensors must be positive")

        self.num_tensors = num_tensors
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(
            torch.empty(num_tensors, out_features, in_features)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(num_tensors, out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for idx in range(self.num_tensors):
            torch.nn.init.kaiming_uniform_(self.weight[idx], a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.in_features:
            x = x.unsqueeze(-1).expand(*x.shape, self.num_tensors)
        elif x.ndim >= 2 and x.shape[-2] == self.in_features and x.shape[-1] == self.num_tensors:
            pass
        else:
            raise ValueError(
                "Expected x to have shape [..., in_features] or "
                "[..., in_features, num_tensors]; "
                f"got {tuple(x.shape)} with in_features={self.in_features} "
                f"and num_tensors={self.num_tensors}"
            )
        y = torch.einsum("...it,toi->...ot", x, self.weight)
        if self.bias is not None:
            y = y + self.bias.transpose(0, 1)
        return y

    def extra_repr(self) -> str:
        bias = self.bias is not None
        return (
            f"num_tensors={self.num_tensors}, "
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={bias}"
        )
