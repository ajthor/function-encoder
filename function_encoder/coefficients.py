import torch


def monte_carlo_integration(
    f: torch.Tensor, g: torch.Tensor, inner_product: callable
) -> torch.Tensor:
    """Compute the coefficients using Monte Carlo integration.

    Args:
        f (torch.Tensor): Function evaluations [batch_size, n_points, n_features]
        g (torch.Tensor): Basis functions evaluations [batch_size, n_points, n_features, n_basis]
        inner_product (callable): Inner product function

    Returns:
        torch.Tensor: Coefficients of the basis functions [batch_size, n_basis]
    """
    F = inner_product(g, f.unsqueeze(-1)).squeeze(-1)
    coefficients = F
    return coefficients, None


def least_squares(
    f: torch.Tensor,
    g: torch.Tensor,
    inner_product: callable,
    regularization: float = 1e-6,
) -> torch.Tensor:
    """Compute the coefficients using least squares.

    Args:
        f (torch.Tensor): Function evaluations [batch_size, n_points, n_features]
        g (torch.Tensor): Basis functions evaluations [batch_size, n_points, n_features, n_basis]
        inner_product (callable): Inner product function
        regularization (float, optional): Regularization parameter. Defaults to 1e-6.

    Returns:
        torch.Tensor: Coefficients of the basis functions [batch_size, n_basis]
    """
    F = inner_product(g, f.unsqueeze(-1)).squeeze(-1)
    G = inner_product(g, g)
    G += regularization * torch.eye(G.size(-1), device=G.device)
    coefficients = torch.linalg.solve(G, F)
    return coefficients, G


def _soft_thresholding(x: torch.Tensor, regularization: float) -> torch.Tensor:
    """Apply soft thresholding to a tensor.

    Args:
        x (torch.Tensor): Input tensor
        regularization (float): Regularization parameter

    Returns:
        torch.Tensor: Soft thresholded tensor
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - regularization, min=0)


def lasso(
    f: torch.Tensor,
    g: torch.Tensor,
    inner_product: callable,
    n_iterations: int = 100,
    regularization: float = 1e-3,
    learning_rate: float = 1e-1,
) -> torch.Tensor:
    """Compute the coefficients using LASSO regression.

    Args:
        f (torch.Tensor): Function evaluations [batch_size, n_points, n_features]
        g (torch.Tensor): Basis functions evaluations [batch_size, n_points, n_features, n_basis]
        inner_product (callable): Inner product function
        n_iterations (int, optional): Number of iterations. Defaults to 100.
        regularization (float, optional): L1 regularization parameter. Defaults to 1e-3.
        learning_rate (float, optional): Learning rate for gradient descent. Defaults to 1e-1.

    Returns:
        torch.Tensor: Coefficients of the basis functions [batch_size, n_basis]
    """
    F = inner_product(g, f.unsqueeze(-1)).squeeze(-1)
    G = inner_product(g, g)
    coefficients = torch.zeros(g.shape[0], g.shape[-1], device=g.device)
    for _ in range(n_iterations):
        grad = torch.einsum("bkl,bl->bk", G, coefficients) - F
        coefficients = _soft_thresholding(
            coefficients - learning_rate * grad, regularization
        )
    return coefficients, G


def gradient_descent(
    f: torch.Tensor,
    g: torch.Tensor,
    inner_product: callable,
    n_iterations: int = 100,
    learning_rate: float = 1e-1,
) -> torch.Tensor:
    """Compute the coefficients using gradient descent.

    Args:
        f (torch.Tensor): Function evaluations [batch_size, n_points, n_features]
        g (torch.Tensor): Basis functions evaluations [batch_size, n_points, n_features, n_basis]
        inner_product (callable): Inner product function
        n_iterations (int, optional): Number of iterations. Defaults to 100.
        learning_rate (float, optional): Learning rate for gradient descent. Defaults to 1e-1.

    Returns:
        torch.Tensor: Coefficients of the basis functions [batch_size, n_basis]
    """
    F = inner_product(g, f.unsqueeze(-1)).squeeze(-1)
    G = inner_product(g, g)
    coefficients = torch.zeros(g.shape[0], g.shape[-1], device=g.device)
    for _ in range(n_iterations):
        grad = torch.einsum("bkl,bl->bk", G, coefficients) - F
        coefficients = coefficients - learning_rate * grad
    return coefficients, G
