import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, Union

from function_encoder.coefficients import least_squares
from function_encoder.inner_products import standard_inner_product

from function_encoder.model.mlp import MLP
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


# Define Van der Pol dynamics.
def van_der_pol(t, x, mu=1.0):
    return torch.stack([x[:, 1], mu * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]], dim=1)


def rk45_step(func, t, x, dt, **ode_kwargs):
    k1 = func(t, x, **ode_kwargs)
    k2 = func(t + dt / 2, x + dt / 2 * k1, **ode_kwargs)
    k3 = func(t + dt / 2, x + dt / 2 * k2, **ode_kwargs)
    k4 = func(t + dt, x + dt * k3, **ode_kwargs)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk45(func, y0, t, device=device, **ode_kwargs):
    """Runge-Kutta 4th order method for ODE integration."""

    y = torch.zeros(
        y0.shape[0], len(t), y0.shape[1], device=device
    )  # [batch, timesteps, features]
    y[:, 0, :] = y0.clone()
    for i in range(1, len(t)):
        y[:, i, :] = rk45_step(
            func, t[i - 1], y[:, i - 1, :].clone(), t[i] - t[i - 1], **ode_kwargs
        )

    return y


def generate_batch(batch_size=8, dt=0.01, T=10.0, device=device):

    # Generate random initial conditions
    y0 = torch.rand(batch_size, 2, device=device) * 3 - 1.5

    # Generate random mu parameters
    mu = torch.rand(batch_size, device=device) * 2 + 0.5

    t = torch.arange(0, T, dt, device=device)
    y = rk45(van_der_pol, y0, t, device=device, mu=mu)

    return y0, t, y


# Create basis functions
def basis_factory():
    return NeuralODE(
        ode_func=ODEFunc(
            model=MLP(layer_sizes=[2, 256, 256, 2], activation=torch.nn.ReLU())
        ),
        integrator=rk45,
    )


class NeuralODEBasisFunctions(torch.nn.Module):
    def __init__(self, *basis_functions):
        super(NeuralODEBasisFunctions, self).__init__()
        self.basis_functions = nn.ModuleList(basis_functions)

    def forward(self, x):
        return torch.stack([basis(*x) for basis in self.basis_functions], dim=-1)


n_basis = 8
basis_functions = NeuralODEBasisFunctions(*[basis_factory() for _ in range(n_basis)])

# Create model

model = FunctionEncoder(basis_functions)
model.to(device)


epochs = 1000
batch_size = 50

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

fig, ax = plt.subplots()
plt.show(block=False)

with tqdm.trange(epochs) as tqdm_bar:
    for epoch in tqdm_bar:

        optimizer.zero_grad()

        y0, t, y = generate_batch(batch_size=batch_size, dt=0.1, T=3.0, device=device)

        coefficients = model.compute_coefficients((y0, t[:10]), y[:, :10, :])
        pred = model((y0, t), coefficients=coefficients)

        loss = torch.nn.functional.mse_loss(pred, y)

        loss.backward()
        optimizer.step()

        tqdm_bar.set_postfix_str(f"loss: {loss.item():.2e}")

        # if epoch % 10 == 0:
        #     # Plot the evaluation trajectory.
        #     ax.clear()
        #     ax.set_xlim(-5, 5)
        #     ax.set_ylim(-5, 5)

        #     model.eval()

        #     y0, t, y = generate_batch(batch_size=1, dt=0.1, T=10.0, device=device)

        #     ax.plot(
        #         y[0, :, 0].detach().cpu().numpy(), y[0, :, 1].detach().cpu().numpy()
        #     )

        #     coefficients = model.compute_coefficients((y0, t[:10]), y[:, :10, :])
        #     pred = model((y0, t), coefficients=coefficients)
        #     pred = pred.detach().cpu().numpy()
        #     ax.plot(pred[0, :, 0], pred[0, :, 1], label="Neural ODE")

        #     plt.draw()
        #     plt.pause(0.1)

        #     model.train()

# Save the model
torch.save(model.state_dict(), "examples/van_der_pol_model.pth")

# Plot a grid of evaluations
with torch.no_grad():
    y0, t, y = generate_batch(batch_size=9, dt=0.1, T=10.0, device=device)

    coefficients = model.compute_coefficients((y0, t[:5]), y[:, :5, :])
    pred = model((y0, t), coefficients=coefficients)
    pred = pred.detach().cpu().numpy()

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(3):
        for j in range(3):
            ax[i, j].set_xlim(-5, 5)
            ax[i, j].set_ylim(-5, 5)
            ax[i, j].plot(
                y[i * 3 + j, :, 0].detach().cpu().numpy(),
                y[i * 3 + j, :, 1].detach().cpu().numpy(),
                label="True",
            )
            ax[i, j].plot(
                pred[i * 3 + j, :, 0], pred[i * 3 + j, :, 1], label="Neural ODE"
            )
            ax[i, j].legend()

    plt.show()
    plt.savefig("examples/van_der_pol.png")
    plt.close()
