import torch


def euclidean_inner_product(x, y):
    return torch.sum(x * y, dim=-2)


def L2(x, y):
    return euclidean_inner_product(x, y).mean(dim=(1, 2))
