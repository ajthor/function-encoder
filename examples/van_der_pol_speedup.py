from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.van_der_pol import VanDerPolDataset, van_der_pol

from function_encoder.model.mlp import MLP, MultiHeadedMLP
from function_encoder.model.neural_ode import NeuralODEFast, ODEFunc, rk4_step, rk4_step_fast
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder, FunctionEncoderFast
from function_encoder.utils.training import train_step

import tqdm

import matplotlib.pyplot as plt

import argparse

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default='1000')
args = parser.parse_args()

torch.manual_seed(3)

# Load dataset

dataset = VanDerPolDataset(n_points=1000, n_example_points=100, dt_range=(0.1, 0.1))
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model


n_basis = 10
basis_functions = MultiHeadedMLP(
    layer_sizes=[3, 64, 64, 2], num_heads=n_basis)

model = NeuralODEFast(
    ode_func = FunctionEncoderFast(
        basis_functions=ODEFunc(model=basis_functions),
    ),
    integrator=rk4_step_fast
).to(device)

# Train model


def loss_function(model, batch):
    _, y0, dt, y1, y0_example, dt_example, y1_example = batch
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    pred = model((y0, dt, coefficients))

    pred_loss = torch.nn.functional.mse_loss(pred, y1)

    return pred_loss


num_epochs = args.num_epochs
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_history = []
with tqdm.trange(num_epochs) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        loss_history.append(loss)
        tqdm_bar.set_postfix_str(f"loss: {loss:.2e}")

# Plot loss curve
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Training Loss over Time")
plt.grid(True)
plt.savefig(f"speedup_loss_{num_epochs}.png")
plt.close()
# plt.show()


# Plot a grid of evaluations


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
            # _c = torch.randn((1, n_basis)).to(device)
            s = 0.1  # Time step for simulation
            n = int(10 / s)
            _dt = torch.tensor([s], device=device)

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
                x = model((x, _dt, _c)) + x
                # x = model((x, _dt)) + x
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

    # plt.show()
    plt.savefig(f"speedup_test_num_epochs={num_epochs}")

    # save the model

torch.save(model.state_dict(), f"van_der_pol_speedup_model_n_basis={n_basis}_epochs={num_epochs}.pth")
