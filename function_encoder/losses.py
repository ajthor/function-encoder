import torch


def basis_normalization_loss(K: torch.Tensor) -> torch.Tensor:
    """Penalize the diagonal of the gram matrix being far from one.

    Args:
        K (torch.Tensor): Gram matrix [batch_size, n_basis, n_basis]

    Returns:
        torch.Tensor: Mean squared difference of diagonal elements from one
    """
    return ((torch.diagonal(K, dim1=-2, dim2=-1) - 1) ** 2).mean()


def basis_orthonormality_loss(K: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Penalize the gram matrix being far from the identity.

    Args:
        K (torch.Tensor): Gram matrix [batch_size, n_basis, n_basis]
        device (torch.device): Device on which to create the identity matrix

    Returns:
        torch.Tensor: Mean norm of the difference between K and the identity matrix
    """
    identity_matrix = torch.eye(K.shape[-1], device=device)
    gram_matrix_penalty = (K - identity_matrix).norm(dim=(1, 2)).mean()
    return gram_matrix_penalty


def residual_loss(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Compute the mean squared error loss between the residual prediction and targets.

    Args:
        model (torch.nn.Module): Model with a residual_function
        inputs (torch.Tensor): Input tensor [batch_size, n_points, n_features]
        targets (torch.Tensor): Target tensor [batch_size, n_points, n_features]

    Returns:
        torch.Tensor: Mean squared error loss
    """
    residual_pred = model.residual_function(inputs)
    return torch.nn.functional.mse_loss(residual_pred, targets)


def matryoshka_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    coefficients: torch.Tensor,
    sizes: list[int],
) -> torch.Tensor:
    """Compute a Matryoshka loss by progressively truncating coefficients.

    Args:
        model (torch.nn.Module): Model with basis_functions for evaluation
        inputs (torch.Tensor): Query points [batch_size, n_points, n_features]
        targets (torch.Tensor): Target values [batch_size, n_points, n_features]
        coefficients (torch.Tensor): Basis coefficients [batch_size, n_basis]
        sizes (list[int]): Prefix sizes for truncating the coefficient vector

    Returns:
        torch.Tensor: Average mean squared error across the requested sizes
    """
    g_query = model.basis_functions(inputs)
    total_loss = 0.0
    for k in sizes:
        coefficients_k = coefficients.clone()
        coefficients_k[..., k:] = 0.0
        y_pred = torch.einsum("bmdk,bk->bmd", g_query, coefficients_k)
        total_loss = total_loss + torch.nn.functional.mse_loss(y_pred, targets)
    return total_loss / len(sizes)
