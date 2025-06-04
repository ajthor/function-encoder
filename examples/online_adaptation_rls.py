from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.van_der_pol import VanDerPolDataset, van_der_pol

from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step

from function_encoder.coefficients import recursive_least_squares_update

import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)

# Load dataset

dataset = VanDerPolDataset(n_points=1000, n_example_points=100, dt_range=(0.1, 0.1))
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model


n_basis = 10
basis_functions = BasisFunctions(
    *[
        NeuralODE(
            ode_func=ODEFunc(model=MLP(layer_sizes=[3, 64, 64, 2])),
            integrator=rk4_step,
        )
        for _ in range(n_basis)
    ]
)

model = FunctionEncoder(basis_functions).to(device)
model.load_state_dict(torch.load("van_der_pol_model.pth", map_location=device))


# Evaluate model

import matplotlib.pyplot as plt


model.eval()
with torch.no_grad():

    # Initialize the coefficients
    coefficients = torch.zeros(1, n_basis, device=device)
    P = torch.eye(n_basis, device=device).unsqueeze(0)

    mu = torch.empty(1, device=device).uniform_(*dataset.mu_range)

    losses_baseline = []
    losses_rls = []
    coefficient_baseline_norms = []
    coefficient_rls_norms = []

    with tqdm.trange(5000) as tqdm_bar:
        for step in tqdm_bar:

            # Update the mu parameter every 500 steps
            if step % 1000 == 0 and step > 0:
                mu = torch.empty(1, device=device).uniform_(*dataset.mu_range)

            # Generate a new observation
            y0 = torch.empty(1, 1, 2, device=device).uniform_(*dataset.y0_range)
            dt = torch.empty(1, 1, device=device).uniform_(*dataset.dt_range)
            y1 = rk4_step(van_der_pol, y0, dt, mu=mu)

            # Compute the basis functions
            g = model.basis_functions((y0, dt))

            # Update the coefficients using recursive least squares
            coefficients, P = recursive_least_squares_update(
                g=g,
                y=y1,
                P=P,
                coefficients=coefficients,
                forgetting_factor=0.95,
                method="woodbury",
            )

            # Generate a new batch of data for evaluation
            n_points = 1000
            _y0 = torch.empty(1, n_points, 2, device=device).uniform_(*dataset.y0_range)
            _dt = torch.empty(1, n_points, device=device).uniform_(*dataset.dt_range)
            _y1 = rk4_step(van_der_pol, _y0, _dt, mu=mu)

            n_example_points = 100
            y0_example = _y0[:, :n_example_points, :]
            dt_example = _dt[:, :n_example_points]
            y1_example = _y1[:, :n_example_points, :]
            y0 = _y0[:, n_example_points:, :]
            dt = _dt[:, n_example_points:]
            y1 = _y1[:, n_example_points:, :]

            # Compute the baseline error
            coefficients_baseline, _ = model.compute_coefficients(
                (y0_example, dt_example), y1_example
            )
            pred_baseline = model((y0, dt), coefficients=coefficients_baseline)
            loss_baseline = torch.nn.functional.mse_loss(pred_baseline, y1)

            # Compute the recursive least squares prediction error
            pred = model((y0, dt), coefficients=coefficients)
            loss_rls = torch.nn.functional.mse_loss(pred, y1)

            losses_baseline.append(loss_baseline.item())
            losses_rls.append(loss_rls.item())

            coefficient_baseline_norms.append(
                coefficients_baseline.norm(dim=-1).mean().item()
            )
            coefficient_rls_norms.append(coefficients.norm(dim=-1).mean().item())

            tqdm_bar.set_postfix(
                {
                    "loss_baseline": f"{loss_baseline.item():.2e}",
                    "loss_rls": f"{loss_rls.item():.2e}",
                }
            )

    # Plot the losses
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(losses_baseline, label="Baseline")
    ax[0].plot(losses_rls, label="Recursive Least Squares")

    ax[1].plot(coefficient_baseline_norms, label="Baseline Coefficients Norm")
    ax[1].plot(coefficient_rls_norms, label="RLS Coefficients Norm")

    plt.show()
