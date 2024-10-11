import torch


def monte_carlo_integration(f, g, inner_product):
    F = torch.einsum("bmdk,bmdl->bmkl", g, f.unsqueeze(-1)).mean(dim=1)

    coefficients = F
    return coefficients


def least_squares(f, g, inner_product, regularization=1e-6):
    F = torch.einsum("bmdk,bmd->bmk", g, f).mean(dim=1)
    G = torch.einsum("bmdk,bmdl->bmkl", g, g).mean(dim=1)
    # G[:: len(G) + 1] += regularization
    # Add regularization
    G += regularization * torch.eye(G.size(-1), device=G.device)

    coefficients = torch.linalg.solve(G, F)
    # coefficients = torch.einsum("bk,bkl->bl", F, torch.linalg.inv(G))

    return coefficients
