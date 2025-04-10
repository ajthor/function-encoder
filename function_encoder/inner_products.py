import torch


def standard_inner_product(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Computes the standard inner product between two tensors f and g.

    The input tensors are of shape (b, m, d, k) and (b, m, d, l), where:
      - b is the batch size
      - m is the number of data points
      - d is the data dimension
      - k is the number of basis functions or features for f
      - l is the number of basis functions or features for g

    For each (k, l) pair, we compute the inner product in R^d and then
    average over the m data points.

    Args:
        f (torch.Tensor): A tensor of shape (b, m, d, k)
        g (torch.Tensor): A tensor of shape (b, m, d, l)

    Returns:
        A tensor of shape (b, k, l)

    """

    return torch.einsum("bmdk,bmdl->bmkl", f, g).mean(dim=1)


def centered_inner_product(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Computes the centered inner product between two tensors f and g.

    The centered inner product is computed by first centering the tensors
    along the m dimension (i.e., subtracting the mean of each tensor
    across the m dimension) and then computing the inner product.

    The input tensors are of shape (b, m, d, k) and (b, m, d, l), where:
      - b is the batch size
      - m is the number of data points
      - d is the data dimension
      - k is the number of basis functions or features for f
      - l is the number of basis functions or features for g

    For each (k, l) pair, we compute the inner product in R^d and then
    average over the m data points.

    Args:
        f (torch.Tensor): A tensor of shape (b, m, d, k)
        g (torch.Tensor): A tensor of shape (b, m, d, l)

    Returns:
        A tensor of shape (b, k, l)

    """

    f_centered = f - f.mean(dim=1, keepdim=True)
    g_centered = g - g.mean(dim=1, keepdim=True)
    return torch.einsum("bmdk,bmdl->bmkl", f_centered, g_centered).mean(dim=1)
