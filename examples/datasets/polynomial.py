import torch
from torch.utils.data import IterableDataset


def polyval(coefficients, X):
    """Evaluate a polynomial at X using Horner's method."""
    y = coefficients[-1] * torch.ones_like(X)
    for c in reversed(coefficients[:-1]):
        y = y * X + c
    return y


class PolynomialDataset(IterableDataset):

    def __init__(
        self, coeff_range=(-1, 1), n_points=1000, n_example_points=100, degree=3
    ):
        super().__init__()
        self.n_points = n_points
        self.n_example_points = n_example_points

        self.coeff_range = coeff_range
        self.degree = degree

    def __iter__(self):
        while True:
            # Generate a single polynomial
            coefficients = torch.empty(self.degree + 1).uniform_(*self.coeff_range)

            # Sample random x values
            _X = torch.empty(self.n_example_points + self.n_points, 1).uniform_(-1, 1)
            _y = polyval(coefficients, _X)

            # Split the data
            X = _X[self.n_example_points :]
            y = _y[self.n_example_points :]
            example_X = _X[: self.n_example_points]
            example_y = _y[: self.n_example_points]

            yield X, y, example_X, example_y
