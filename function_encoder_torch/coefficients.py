import torch


def monte_carlo_integration(f, g, inner_product):
    F = torch.einsum("bmdk,bmdl->bmkl", g, f.unsqueeze(-1)).mean(dim=1)

    coefficients = F
    return coefficients


def least_squares(f, g, inner_product, regularization=1e-6):
    F = torch.einsum("bmdk,bmd->bmk", g, f).mean(dim=1)
    G = torch.einsum("bmdk,bmdl->bmkl", g, g).mean(dim=1)
    G[:: len(G) + 1] += regularization

    coefficients = torch.linalg.solve(G, F)
    return coefficients
