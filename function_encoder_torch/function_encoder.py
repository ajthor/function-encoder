from typing import Callable, Optional

import torch

from function_encoder_torch.coefficients import least_squares
from function_encoder_torch.inner_products import L2


class FunctionEncoder(torch.nn.Module):
    def __init__(
        self,
        basis_functions: torch.nn.ModuleList,
        residual_function: Optional[torch.nn.Module] = None,
        coefficients_method: Callable = least_squares,
        inner_product: Callable = L2,
    ):
        super(FunctionEncoder, self).__init__()

        self.basis_functions = basis_functions
        self.residual_function = residual_function

        self.coefficients_method = coefficients_method
        self.inner_product = inner_product

    def compute_coefficients(self, x, y):
        G = torch.cat(
            [basis(x.unsqueeze(-1)) for basis in self.basis_functions], dim=-1
        )

        if self.residual_function is not None:
            y_residual = self.residual_function(x)
            coefficients = self.coefficients_method(
                y - y_residual, G, self.inner_product
            )
        else:
            coefficients = self.coefficients_method(y, G, self.inner_product)

        return coefficients

    def forward(self, x, coefficients):
        G = torch.cat(
            [basis(x.unsqueeze(-1)) for basis in self.basis_functions], dim=-1
        )
        y = torch.einsum("bdmk,bk->bdm", G, coefficients)

        if self.residual_function is not None:
            y += self.residual_function(x)

        return y
