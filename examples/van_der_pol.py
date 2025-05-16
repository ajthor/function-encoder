from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from function_encoder.coefficients import least_squares
from function_encoder.inner_products import standard_inner_product

from function_encoder.model.mlp import MLP, MultiHeadedMLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import fit

import matplotlib.pyplot as plt

import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)


def rk4_step(func, x, dt, **ode_kwargs):
    t = torch.zeros_like(dt, device=dt.device)
    k1 = func(t, x, **ode_kwargs)
    k2 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k1, **ode_kwargs)
    k3 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k2, **ode_kwargs)
    k4 = func(t + dt, x + dt.unsqueeze(-1) * k3, **ode_kwargs)
    return (dt / 6).unsqueeze(-1) * (k1 + 2 * k2 + 2 * k3 + k4)


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
            # Generate a single mu
            mu = torch.empty(1, device=device).uniform_(*self.mu_range)
            # Generate random initial conditions
            _y0 = torch.empty(
                self.n_example_points + self.n_points, 2, device=device
            ).uniform_(*self.y0_range)
            # Generate random time steps
            _dt = torch.empty(
                self.n_example_points + self.n_points, device=device
            ).uniform_(*self.dt_range)
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


# Create basis functions
def basis_factory():
    return NeuralODE(
        ode_func=ODEFunc(
            model=MLP(layer_sizes=[2, 64, 64, 2], activation=torch.nn.ReLU())
        ),
        integrator=rk4_step,
    )


n_basis = 10
basis_functions = BasisFunctions(*[basis_factory() for _ in range(n_basis)])

# Create model

model = FunctionEncoder(basis_functions).to(device)

# Train model

epochs = 1000
batch_size = 50
n_points = 1000
n_example_points = 100

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataloader = DataLoader(
    VanDerPolDataset(n_points=n_points, n_example_points=n_example_points),
    batch_size=batch_size,
)
dataloader_iter = iter(dataloader)

plot_progress_every_n = None  # e.g. 50
if plot_progress_every_n is not None:
    fig, ax = plt.subplots()
    plt.show(block=False)
    plt.draw()
    plt.pause(0.1)

with tqdm.trange(epochs) as tqdm_bar:
    for epoch in tqdm_bar:
        model.train()
        optimizer.zero_grad()

        batch = next(dataloader_iter)

        mu, y0, dt, y1, y0_example, dt_example, y1_example = batch

        coefficients, G = model.compute_coefficients(
            (y0_example, dt_example), y1_example
        )
        pred = model((y0, dt), coefficients=coefficients)

        pred_loss = torch.nn.functional.mse_loss(pred, y1)
        # norm_loss = basis_normalization_loss(G)
        loss = pred_loss  # + norm_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        tqdm_bar.set_postfix_str(f"loss: {loss.item():.2e}")

        if plot_progress_every_n is not None and epoch % plot_progress_every_n == 0:
            # Plot the evaluation trajectory.
            ax.clear()
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)

            with torch.no_grad():
                model.eval()

                # Plot a single trajectory using the data from the first batch
                _mu = mu[0]
                _y0 = torch.empty(1, 2, device=device).uniform_(
                    *dataloader.dataset.y0_range
                )
                _c = coefficients[0].unsqueeze(0)

                n = int(10 / 0.1)
                _dt = torch.tensor([0.1], device=device)

                # Integrate the true trajectory
                x = _y0.clone()
                y = [x]
                for i in range(n):
                    x = rk4_step(van_der_pol, x, _dt, mu=_mu) + x
                    y.append(x)
                y = torch.cat(y, dim=0)
                y = y.detach().cpu().numpy()

                # Integrate the predicted trajectory
                x = _y0.clone()
                x = x.unsqueeze(1)
                _dt = _dt.unsqueeze(0)
                pred = [x]
                for i in range(n):
                    x = model((x, _dt), coefficients=_c) + x
                    pred.append(x)
                pred = torch.cat(pred, dim=1)
                pred = pred.detach().cpu().numpy()

                ax.plot(y[:, 0], y[:, 1])
                ax.plot(pred[0, :, 0], pred[0, :, 1], label="Neural ODE")

                plt.draw()
                plt.pause(0.1)

if plot_progress_every_n is not None:
    plt.close()

# Save the model
torch.save(model.state_dict(), "examples/van_der_pol_model.pth")

# Plot a grid of evaluations
with torch.no_grad():
    model.eval()

    # Generate a single batch of functions for plotting
    dataloader = DataLoader(
        VanDerPolDataset(n_points=n_points, n_example_points=n_example_points),
        batch_size=9,
    )
    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)

    mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
    # Precompute the coefficients for the batch
    coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(3):
        for j in range(3):

            # Plot a single trajectory
            _mu = mu[i * 3 + j]
            _y0 = torch.empty(1, 2, device=device).uniform_(
                *dataloader.dataset.y0_range
            )
            # We use the coefficients that we computed before
            _c = coefficients[i * 3 + j].unsqueeze(0)
            n = int(10 / 0.1)
            _dt = torch.tensor([0.1], device=device)

            # Integrate the true trajectory
            x = _y0.clone()
            y = [x]
            for k in range(n):
                x = rk4_step(van_der_pol, x, _dt, mu=_mu) + x
                y.append(x)
            y = torch.cat(y, dim=0)
            y = y.detach().cpu().numpy()

            # Integrate the predicted trajectory
            x = _y0.clone()
            x = x.unsqueeze(1)
            _dt = _dt.unsqueeze(0)
            pred = [x]
            for k in range(n):
                x = model((x, _dt), coefficients=_c) + x
                pred.append(x)
            pred = torch.cat(pred, dim=1)
            pred = pred.detach().cpu().numpy()

            ax[i, j].set_xlim(-5, 5)
            ax[i, j].set_ylim(-5, 5)
            (_t,) = ax[i, j].plot(y[:, 0], y[:, 1], label="True")
            (_p,) = ax[i, j].plot(pred[0, :, 0], pred[0, :, 1], label="Predicted")

    fig.legend(
        handles=[_t, _p],
        loc="outside upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
    )

    plt.savefig("examples/van_der_pol.png")
    plt.close()
