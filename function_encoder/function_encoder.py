from typing import Callable, Optional, Tuple, Union
import torch
from function_encoder.coefficients import least_squares
from function_encoder.inner_products import standard_inner_product


class BasisFunctions(torch.nn.Module):
    """A module representing a collection of basis functions.

    Args:
        basis_functions (torch.nn.Module): A collection of basis functions to be evaluated.
    """

    def __init__(self, *basis_functions: torch.nn.Module):
        super(BasisFunctions, self).__init__()
        self.basis_functions = torch.nn.ModuleList(basis_functions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all basis functions at the input points.

        Args:
            x (torch.Tensor): Input tensor [batch_size, n_points, n_features]

        Returns:
            torch.Tensor: Basis function evaluations [batch_size, n_points, n_features, n_basis]
        """
        return torch.stack([basis(x) for basis in self.basis_functions], dim=-1)


class FunctionEncoder(torch.nn.Module):
    """A module for encoding functions in terms of basis functions.

    The FunctionEncoder represents functions as linear combinations of basis
    functions. It can compute coefficients for these basis functions given
    input-output pairs, and can evaluate the functions at new inputs.

    Args:
        basis_functions (torch.nn.Module): Module providing basis function evaluations
        residual_function (Optional[torch.nn.Module], optional): Module for residual function. Defaults to None.
        coefficients_method (Callable, optional): Method to compute coefficients. Defaults to least_squares.
        inner_product (Callable, optional): Inner product function. Defaults to standard_inner_product.
    """

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

    def compute_coefficients(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the coefficients of the basis functions.

        Args:
            x (torch.Tensor): Input data [batch_size, n_points, n_features]
            y (torch.Tensor): Target data [batch_size, n_points, n_features]

        Returns:
            torch.Tensor: Basis coefficients [batch_size, n_basis]

        """
        f = y
        g = self.basis_functions(x)
        if self.residual_function is not None:
            f -= self.residual_function(x)

        coefficients, G = self.coefficients_method(f, g, self.inner_product)

        return coefficients, G

    def forward(self, x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """Evaluate the function corresponding to the coefficients at x.

        Args:
            x (torch.Tensor): Input data [batch_size, n_points, n_features]
            coefficients (torch.Tensor): Coefficients of the basis functions [batch_size, n_basis]

        Returns:
            torch.Tensor: Function values at x [batch_size, n_points, n_features]
        """
        G = self.basis_functions(x)
        y = torch.einsum("bmdk,bk->bmd", G, coefficients)
        if self.residual_function is not None:
            y += self.residual_function(x)
        return y
