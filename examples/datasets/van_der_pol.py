import torch
from torch.utils.data import IterableDataset

from function_encoder.model.neural_ode import rk4_step


def van_der_pol(t, x, mu=1.0):
    return torch.stack(
        [x[..., 1], mu * (1 - x[..., 0] ** 2) * x[..., 1] - x[..., 0]], dim=-1
    )


class VanDerPolDataset(IterableDataset):
    def __init__(
        self,
        n_points: int = 1000,
        n_example_points: int = 100,
        mu_range=(0.5, 2.5),
        y0_range=(-3.5, 3.5),
        dt_range=(0.01, 0.1),
    ):
        super().__init__()
        self.n_points = n_points
        self.n_example_points = n_example_points
        self.mu_range = mu_range
        self.y0_range = y0_range
        self.dt_range = dt_range

    def __iter__(self):
        while True:
            total_points = self.n_example_points + self.n_points
            # Generate a single mu
            mu = torch.empty(1).uniform_(*self.mu_range)
            # Generate random initial conditions
            _y0 = torch.empty(total_points, 2).uniform_(*self.y0_range)
            # Generate random time steps
            _dt = torch.empty(total_points).uniform_(*self.dt_range)
            # Integrate one step
            _y1 = rk4_step(van_der_pol, _y0, _dt, mu=mu)

            # Split the data
            y0_example = _y0[: self.n_example_points]
            dt_example = _dt[: self.n_example_points]
            y1_example = _y1[: self.n_example_points]

            y0 = _y0[self.n_example_points :]
            dt = _dt[self.n_example_points :]
            y1 = _y1[self.n_example_points :]

            yield mu, y0, dt, y1, y0_example, dt_example, y1_example
