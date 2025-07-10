import torch
import math


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
    regularization: float = 1e-3,
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
    coefficients = torch.linalg.solve(
        G + regularization * torch.eye(G.size(-1), device=G.device), F
    )
    return coefficients, G


def _soft_thresholding(x: torch.Tensor, regularization: float = 1e-6) -> torch.Tensor:
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


def recursive_least_squares_update(
    g: torch.Tensor,
    y: torch.Tensor,
    P: torch.Tensor,
    coefficients: torch.Tensor,
    forgetting_factor: float = 0.99,
    method: str = "woodbury",
):
    """Update coefficients using recursive least squares.

    Args:
        g (torch.Tensor): Basis functions evaluations [batch_size, n_points, n_features, n_basis]
        y (torch.Tensor): Function evaluations [batch_size, n_points, n_features]
        P (torch.Tensor): Covariance matrix [batch_size, n_basis, n_basis]
        coefficients (torch.Tensor): Current coefficients [batch_size, n_basis]
        forgetting_factor (float, optional): Forgetting factor for the update. Defaults to 0.99.
        method (str, optional): Method for updating coefficients. Options are "woodbury", "qr". Defaults to "woodbury".

    Returns:
        torch.Tensor: Updated coefficients [batch_size, n_basis]
    """

    if method == "woodbury":
        return _rls_woodbury(g, y, P, coefficients, forgetting_factor)
    elif method == "qr":
        return _rls_qr(g, y, P, coefficients, forgetting_factor)
    else:
        raise ValueError(f"Unknown method: {method}")


def _rls_woodbury(
    g: torch.Tensor,
    y: torch.Tensor,
    P: torch.Tensor,
    coefficients: torch.Tensor,
    forgetting_factor: float = 0.99,
):
    """Standard RLS update using the Woodbury identity (vectorized over batch).

    Args:
        g (torch.Tensor): Feature vector [batch_size, n_points, n_features, n_basis]
        y (torch.Tensor): Observation [batch_size, n_points, n_features]
        P (torch.Tensor): Covariance matrix [batch_size, n_basis, n_basis]
        coefficients (torch.Tensor): Current coefficient vector [batch_size, n_basis]
        forgetting_factor (float): Forgetting factor (lambda)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated coefficients and covariance P
    """

    g = g.squeeze(1)
    y = y.squeeze(1)

    # Compute residual
    y_pred = torch.einsum("bdk,bk->bd", g, coefficients)
    residual = y - y_pred  # [batch_size, n_features]

    # Compute the Kalman gain
    gTP = torch.einsum("bdk,bkl->bdl", g, P)
    S = torch.einsum("bcl,bdk->bcd", gTP, g)
    S += torch.eye(S.size(-1), device=S.device) * (forgetting_factor)

    K = torch.linalg.solve(S, gTP)  # Kalman gain [batch_size, n_features, n_basis]

    # Update the coefficients
    coefficients += torch.einsum("bdk,bd->bk", K, residual)

    # Update the covariance matrix using the Woodbury identity
    P = P - torch.einsum("bdk,bdl->bkl", K, gTP)

    P = P / forgetting_factor  # Scale the covariance matrix

    return coefficients, P


def _rls_qr(
    g: torch.Tensor,
    y: torch.Tensor,
    L: torch.Tensor,
    coefficients: torch.Tensor,
    forgetting_factor: float = 0.99,
):
    """Simple RLS update using QR decomposition (vectorized over batch).

    Args:
        g (torch.Tensor): Feature vector [batch_size, n_points, n_features, n_basis]
        y (torch.Tensor): Observation [batch_size, n_points, n_features]
        L (torch.Tensor): Cholesky factor of covariance matrix [batch_size, n_basis, n_basis]
        coefficients (torch.Tensor): Current coefficient vector [batch_size, n_basis]
        forgetting_factor (float): Forgetting factor (lambda)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated coefficients and covariance P
    """
    g = g.squeeze(1)
    y = y.squeeze(1)

    # Compute principal square root of forgetting_factor * I_{n_features}
    mu = math.sqrt(forgetting_factor) * torch.eye(g.size(-2), device=g.device)
    mu = torch.stack([mu] * g.size(0), dim=0)  # [batch_size, n_features, n_features]

    # Form the block system [batch, n_features + n_basis, n_features + n_basis]
    b = g.size(0)
    d = g.size(-2)  # n_features
    k = g.size(-1)  # n_basis
    LT = L.transpose(-1, -2)  # Cholesky factor transpose [batch_size, n_basis, n_basis]
    LTg = torch.einsum("bck,bdk->bcd", LT, g)  # [batch_size, n_basis, n_features]
    B = torch.zeros((b, d + k, d + k), device=g.device)
    B[:, :d, :d] = mu
    B[:, d:, :d] = LTg
    B[:, d:, d:] = LT

    # QR decomposition on block system
    _, R = torch.linalg.qr(
        B, mode="reduced"
    )  # reduced for square B is the complete decomposition, but more efficient
    # Extract blocks from the R matrix
    R1 = R[:, :d, :d]  # upper left block [batch_size, n_features, n_features]
    R1T = R[:, :d, :d].transpose(
        -1, -2
    )  # upper left block [batch_size, n_features, n_features]
    R2 = R[:, :d, d:]  # upper right block [batch_size, n_features, n_basis]
    R3 = R[:, d:, d:]  # lower right block [batch_size, n_basis, n_basis]
    R3T = R3.transpose(-1, -2)

    # Compute kalman gain R2T @ R1T^{-1}, use triangular solve for efficiency
    K = torch.linalg.solve_triangular(R1T, R2, upper=False, left=True).transpose(
        -1, -2
    )  # [batch_size, n_basis, n_features]

    # Compute residual
    y_pred = torch.einsum("bdk,bk->bd", g, coefficients)
    residual = y - y_pred  # [batch_size, n_features]

    # Update coefficients
    coefficients += torch.einsum("bkd,bd->bk", K, residual)

    # Update the covariance matrix
    P = (1 / forgetting_factor) * torch.einsum(
        "bck,bke->bce", R3T, R3
    )  # [batch_size, n_basis, n_basis]

    # Innovation covariance... for fun...
    S = torch.einsum("bdn,bnm->bdm", R1T, R1)

    return coefficients, P
