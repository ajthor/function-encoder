from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.van_der_pol import VanDerPolDataset, van_der_pol

from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step

import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)

# Load dataset

dataset = VanDerPolDataset(n_points=1000, n_example_points=100)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model


def basis_factory():
    return NeuralODE(
        ode_func=ODEFunc(
            model=MLP(layer_sizes=[2, 64, 64, 2], activation=torch.nn.ReLU())
        ),
        integrator=rk4_step,
    )


n_basis = 10
basis_functions = BasisFunctions(*[basis_factory() for _ in range(n_basis)])

model = FunctionEncoder(basis_functions).to(device)

# Train model


def loss_function(model, batch):
    mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)
    pred = model((y0, dt), coefficients=coefficients)

    pred_loss = torch.nn.functional.mse_loss(pred, y1)

    return pred_loss


num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.trange(num_epochs) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        # model.train()
        # optimizer.zero_grad()

        # # mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
        # # y0 = y0.to(device)
        # # dt = dt.to(device)
        # # y1 = y1.to(device)
        # # y0_example = y0_example.to(device)
        # # dt_example = dt_example.to(device)
        # # y1_example = y1_example.to(device)

        # # coefficients, G = model.compute_coefficients(
        # #     (y0_example, dt_example), y1_example
        # # )
        # # pred = model((y0, dt), coefficients=coefficients)

        # # pred_loss = torch.nn.functional.mse_loss(pred, y1)
        # # # norm_loss = basis_normalization_loss(G)
        # # loss = pred_loss  # + norm_loss

        # loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # optimizer.step()

        tqdm_bar.set_postfix_str(f"loss: {loss:.2e}")


# Save the model
torch.save(model.state_dict(), "examples/van_der_pol_model.pth")

# Plot a grid of evaluations

import matplotlib.pyplot as plt


model.eval()
with torch.no_grad():
    # Generate a single batch of functions for plotting
    dataloader = DataLoader(dataset, batch_size=9)
    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)

    mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
    mu = mu.to(device)
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

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
