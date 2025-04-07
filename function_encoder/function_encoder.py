from typing import Callable, Optional

import torch

from function_encoder.coefficients import least_squares
from function_encoder.inner_products import standard_inner_product


class BasisFunctions(torch.nn.Module):
    def __init__(self, basis_functions: torch.nn.ModuleList):
        super(BasisFunctions, self).__init__()

        self.basis_functions = basis_functions

    def forward(self, x):
        return torch.stack([basis(x) for basis in self.basis_functions], dim=-1)


class FunctionEncoder(torch.nn.Module):

    def __init__(
        self,
        basis_functions: torch.nn.Module,
        residual_function: Optional[torch.nn.Module] = None,
        coefficients_method: Callable = least_squares,
        inner_product: Callable = standard_inner_product,
    ):
        super(FunctionEncoder, self).__init__()

        self.basis_functions = basis_functions
        self.residual_function = residual_function

        self.coefficients_method = coefficients_method
        self.inner_product = inner_product

    def compute_coefficients(self, x, y, return_G=False):
        G = self.basis_functions(x)

        if self.residual_function is not None:
            y_residual = self.residual_function(x)
            coefficients = self.coefficients_method(
                y - y_residual, G, self.inner_product
            )
        else:
            coefficients = self.coefficients_method(y, G, self.inner_product)

        if return_G:
            return coefficients, G
        else:
            return coefficients

    def forward(self, x, coefficients):
        G = self.basis_functions(x)
        y = torch.einsum("bmdk,bk->bmd", G, coefficients)

        if self.residual_function is not None:
            y += self.residual_function(x)

        return y
