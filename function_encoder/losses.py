import torch


def basis_normalization_loss(K):
    # Penalize the diagonal of the gram matrix being far from one.
    return ((torch.diagonal(K, dim1=-2, dim2=-1)) ** 2).mean()


def basis_orthonormality_loss(K, device):
    # Penalize the gram matrix being far from the identity.
    identity_matrix = torch.eye(K.shape[-1], device=device)
    gram_matrix_penalty = (K - identity_matrix).norm(dim=(1, 2)).mean()
    return gram_matrix_penalty


def residual_loss(model, inputs, targets):
    residual_pred = model.residual_function(inputs)
    return torch.nn.functional.mse_loss(residual_pred, targets)
