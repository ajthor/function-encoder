import torch


def euclidean_inner_product(f, g):
    return torch.einsum("bmdk,bmdl->bmkl", f, g).mean(dim=1)
