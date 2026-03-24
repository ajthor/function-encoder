import math

import torch
from torch.utils.data import IterableDataset


def fourier_series(
    X: torch.Tensor,
    amplitudes: torch.Tensor,
    frequencies: torch.Tensor,
    phases: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a Fourier series with fixed frequencies at X."""
    angles = 2 * math.pi * X * frequencies.unsqueeze(0) + phases.unsqueeze(0)
    components = amplitudes.unsqueeze(0) * torch.sin(angles)
    return components.sum(dim=-1, keepdim=True)


class RandomFourierDataset(IterableDataset):
    def __init__(
        self,
        frequencies=(2.0, 4.0, 6.0, 8.0),
        amplitude_range=(-1, 1),
        n_points=1000,
        n_example_points=100,
    ):
        super().__init__()
        self.n_points = n_points
        self.n_example_points = n_example_points

        self.frequencies = torch.as_tensor(frequencies, dtype=torch.float32)
        self.amplitude_range = amplitude_range

    def __iter__(self):
        while True:
            amplitudes = torch.empty(len(self.frequencies)).uniform_(
                *self.amplitude_range
            )
            phases = torch.empty(len(self.frequencies)).uniform_(0, 2 * math.pi)

            _X = torch.empty(self.n_example_points + self.n_points, 1).uniform_(-1, 1)
            _y = fourier_series(_X, amplitudes, self.frequencies, phases)

            X = _X[self.n_example_points :]
            y = _y[self.n_example_points :]
            example_X = _X[: self.n_example_points]
            example_y = _y[: self.n_example_points]

            yield X, y, example_X, example_y
