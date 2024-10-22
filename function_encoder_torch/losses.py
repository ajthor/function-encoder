import torch


def basis_normalization_loss(G):
    # Penalize the diagonal of the gram matrix being far from one.
    return ((torch.diagonal(G, dim1=-2, dim2=-1)) ** 2).mean()


def basis_orthonormality_loss(G, device):
    # Penalize the gram matrix being far from the identity.
    identity_matrix = torch.eye(G.shape[-1], device=device)
    gram_matrix_penalty = (G - identity_matrix).norm(dim=(1, 2)).mean()
    return gram_matrix_penalty


def residual_loss(model, inputs, targets):
    residual_pred = model.residual_function(inputs)
    return torch.nn.functional.mse_loss(residual_pred, targets)
