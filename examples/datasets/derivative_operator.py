import torch
from torch.utils.data import IterableDataset


def polyval(coefficients, X):
    """Evaluate a polynomial at X using Horner's method."""
    y = torch.zeros_like(X)
    for c in coefficients:
        y = y * X + c
    return y


def polyder(coefficients):
    """Compute the coefficients of the derivative of a polynomial."""
    degree = len(coefficients) - 1
    return torch.tensor([coefficients[i] * (degree - i) for i in range(degree)])


def polyint(coefficients):
    """Compute the coefficients of the antiderivative of a polynomial, constant of integration set to zero."""
    degree = len(coefficients) - 1
    new_coeffs = [coefficients[i] / (degree - i + 1) for i in range(degree + 1)]
    new_coeffs.append(torch.tensor(0.0, dtype=coefficients.dtype))
    return torch.stack(new_coeffs)


class DerivativeOperatorDataset(IterableDataset):
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
            coefficients = torch.empty(self.degree + 1).uniform_(*self.coeff_range)
            der_coefficients = polyder(coefficients)

            _X = torch.empty(self.n_example_points + self.n_points, 1).uniform_(-1, 1)
            _Y = torch.empty(self.n_example_points + self.n_points, 1).uniform_(-1, 1)
            _u = polyval(coefficients, _X)
            _s = polyval(der_coefficients, _Y)

            X = _X[self.n_example_points :]
            u = _u[self.n_example_points :]
            Y = _Y[self.n_example_points :]
            s = _s[self.n_example_points :]
            example_X = _X[: self.n_example_points]
            example_u = _u[: self.n_example_points]
            example_Y = _Y[: self.n_example_points]
            example_s = _s[: self.n_example_points]

            yield X, u, Y, s, example_X, example_u, example_Y, example_s


class AntiderivativeOperatorDataset(IterableDataset):
    def __init__(
        self, coeff_range=(-1, 1), n_points=1000, n_example_points=100, degree=2
    ):
        super().__init__()
        self.n_points = n_points
        self.n_example_points = n_example_points
        self.coeff_range = coeff_range
        self.degree = degree

    def __iter__(self):
        while True:
            coefficients = torch.empty(self.degree + 1).uniform_(*self.coeff_range)
            int_coefficients = polyint(coefficients)

            _X = torch.empty(self.n_example_points + self.n_points, 1).uniform_(-1, 1)
            _Y = torch.empty(self.n_example_points + self.n_points, 1).uniform_(-1, 1)
            _u = polyval(coefficients, _X)
            _s = polyval(int_coefficients, _Y)

            X = _X[self.n_example_points :]
            u = _u[self.n_example_points :]
            Y = _Y[self.n_example_points :]
            s = _s[self.n_example_points :]
            example_X = _X[: self.n_example_points]
            example_u = _u[: self.n_example_points]
            example_Y = _Y[: self.n_example_points]
            example_s = _s[: self.n_example_points]

            yield X, u, Y, s, example_X, example_u, example_Y, example_s
