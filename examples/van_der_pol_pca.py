from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.van_der_pol import VanDerPolDataset, van_der_pol

from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.losses import residual_loss
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

dataset = VanDerPolDataset(
    integrator=rk4_step, n_points=1000, n_example_points=100, dt_range=(0.1, 0.1)
)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model


def basis_function_factory():
    return NeuralODE(
        ode_func=ODEFunc(
            model=MLP(layer_sizes=[3, 64, 64, 2], activation=torch.nn.ReLU())
        ),
        integrator=rk4_step,
    )


num_basis = 5
# Only use one basis function initially for progressive training
basis_functions = BasisFunctions(basis_function_factory())
# residual_function = basis_function_factory()
model = FunctionEncoder(basis_functions).to(device)

# Train model

losses = []  # For plotting.
scores = []  # For plotting.


def loss_function(model, batch):
    _, y0, dt, y1, y0_example, dt_example, y1_example = batch
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    pred = model((y0, dt), coefficients=coefficients)

    pred_loss = torch.nn.functional.mse_loss(pred, y1)
    # res_loss = residual_loss(model, (y0_example, dt_example), y1_example)

    return pred_loss  # + res_loss


num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.trange(num_epochs) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        losses.append(loss)
        tqdm_bar.set_postfix_str(f"loss: {loss:.2e}")


def compute_explained_variance(model):
    dataloader_coeffs = DataLoader(dataset, batch_size=100)
    dataloader_coeffs_iter = iter(dataloader_coeffs)
    _, y0, dt, y1, y0_example, dt_example, y1_example = next(dataloader_coeffs_iter)
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)
    coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)

    coefficients_centered = coefficients - coefficients.mean(dim=0, keepdim=True)
    coefficients_cov = (
        torch.matmul(coefficients_centered.T, coefficients_centered)
        / coefficients.shape[0]
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(coefficients_cov)
    eigenvalues = eigenvalues.flip(0)  # Flip to descending order

    explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)

    # eigenvectors = eigenvectors.flip(1)  # Flip to descending order
    # fpc_scores = torch.matmul(coefficients_centered, eigenvectors)

    K = G.mean(dim=0)
    gram_eigenvalues, gram_eigenvectors = torch.linalg.eigh(K)
    gram_eigenvalues = gram_eigenvalues.flip(0)  # Flip to descending order

    return explained_variance_ratio, eigenvalues, gram_eigenvalues


model.eval()
with torch.no_grad():
    explained_variance_ratio, eigenvalues, _ = compute_explained_variance(model)
    scores.append(explained_variance_ratio)


# Train the remaining basis functions progressively
for k in range(num_basis - 1):

    # Freeze all existing parameters except the new basis function
    for param in model.parameters():
        param.requires_grad = False

    # Create a new basis function and add it to the model
    new_basis_function = basis_function_factory()
    for param in new_basis_function.parameters():
        param.requires_grad = True

    new_basis_function = new_basis_function.to(device)
    model.basis_functions.basis_functions.append(new_basis_function)

    # Select only the trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    with tqdm.tqdm(range(num_epochs), desc=f"basis {k + 2}/{num_basis}") as tqdm_bar:
        for epoch in tqdm_bar:
            batch = next(dataloader_iter)
            loss = train_step(model, optimizer, batch, loss_function)
            losses.append(loss)
            tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})

    model.eval()
    with torch.no_grad():
        explained_variance_ratio, *_ = compute_explained_variance(model)
        scores.append(explained_variance_ratio)


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

    plt.show()

    # Plot loss and explained variance
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot loss
    ax1.plot(losses)
    ax1.set_ylabel("MSE")
    ax1.grid(True)
    ax1.set_yscale("log")

    # Plot explained variance ratio
    for i in range(len(scores)):
        scores[i] = scores[i].cpu().numpy()
        ax2.plot(
            range(1, len(scores[i]) + 1),
            scores[i],
            marker="o",
            label=f"k = {i + 1}",
        )
    ax2.set_xlabel("Eigenvalue Index")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True)

    # Plot the eigenvalues of the coefficients
    _, eigenvalues, gram_eigenvalues = compute_explained_variance(model)
    eigenvalues = eigenvalues.cpu().numpy()
    gram_eigenvalues = gram_eigenvalues.cpu().numpy()

    ax3.plot(
        range(1, len(eigenvalues) + 1),
        eigenvalues,
        marker="o",
        label="Covariance Matrix",
    )
    ax3.plot(
        range(1, len(gram_eigenvalues) + 1),
        gram_eigenvalues,
        marker="o",
        label="Gram Matrix",
    )
    ax3.set_xlabel("Eigenvalue Index")
    ax3.set_ylabel("Eigenvalue")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()
