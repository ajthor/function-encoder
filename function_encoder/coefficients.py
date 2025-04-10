import torch


def monte_carlo_integration(f, g, inner_product):
    F = inner_product(g, f.unsqueeze(-1)).squeeze(-1)

    coefficients = F
    return coefficients


def least_squares(f, g, inner_product, regularization=1e-6):
    F = inner_product(g, f.unsqueeze(-1)).squeeze(-1)
    G = inner_product(g, g)
    G += regularization * torch.eye(G.size(-1), device=G.device)

    coefficients = torch.linalg.solve(G, F)
    return coefficients


def soft_thresholding(x, regularization):
    return torch.sign(x) * torch.clamp(torch.abs(x) - regularization, min=0)


def lasso(
    f, g, inner_product, n_iterations=100, regularization=1e-3, learning_rate=1e-1
):
    F = inner_product(g, f.unsqueeze(-1)).squeeze(-1)
    G = inner_product(g, g)

    coefficients = torch.zeros(g.shape[0], g.shape[-1], device=g.device)

    for _ in range(n_iterations):
        grad = torch.einsum("bkl,bl->bk", G, coefficients) - F
        coefficients = soft_thresholding(
            coefficients - learning_rate * grad, regularization
        )

    return coefficients


def gradient_descent(f, g, inner_product, n_iterations=100, learning_rate=1e-1):
    F = inner_product(g, f.unsqueeze(-1)).squeeze(-1)
    G = inner_product(g, g)

    coefficients = torch.zeros(g.shape[0], g.shape[-1], device=g.device)

    for _ in range(n_iterations):
        grad = torch.einsum("bkl,bl->bk", G, coefficients) - F
        coefficients = coefficients - learning_rate * grad

    return coefficients
