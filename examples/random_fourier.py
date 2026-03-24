import math
import torch

from torch.utils.data import DataLoader
from data.random_fourier import RandomFourierDataset

from function_encoder.model.activations import Sine, init_siren
from function_encoder.model.tensor_layers import ParallelLinear
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import (
    basis_normalization_loss,
    matryoshka_loss,
    basis_orthonormality_loss,
)
from function_encoder.utils.training import train_step

import tqdm

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


torch.manual_seed(42)

# Load dataset

dataset = RandomFourierDataset(
    n_points=1000,
    n_example_points=100,
    frequencies=(2.0, 4.0, 6.0, 8.0, 20.0),
)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model

n_basis = 8
basis_functions = torch.nn.Sequential(
    init_siren(ParallelLinear(n_basis, 1, 64), layer_idx=0),
    Sine(omega_0=30.0),
    init_siren(ParallelLinear(n_basis, 64, 64), layer_idx=1),
    Sine(omega_0=30.0),
    init_siren(ParallelLinear(n_basis, 64, 1), layer_idx=2),
)

model = FunctionEncoder(basis_functions).to(device)

# Train model

matryoshka_sizes = list(range(1, n_basis + 1))


def loss_function(model, batch):
    X, y, example_X, example_y = batch
    X = X.to(device)
    y = y.to(device)
    example_X = example_X.to(device)
    example_y = example_y.to(device)

    coefficients, G = model.compute_coefficients(example_X, example_y)
    norm_loss = basis_orthonormality_loss(G, device=device)
    pred_loss = matryoshka_loss(model, X, y, coefficients, matryoshka_sizes)
    # pred_loss = torch.nn.functional.mse_loss(model(X, coefficients), y)

    return pred_loss + norm_loss


num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.tqdm(range(num_epochs)) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})


# Plot an evaluation of the model


model.eval()
with torch.no_grad():
    dataloader = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader))

    X, y, example_X, example_y = batch
    X = X.to(device)
    y = y.to(device)
    example_X = example_X.to(device)
    example_y = example_y.to(device)

    idx = torch.argsort(X, dim=1, descending=False)
    X = torch.gather(X, dim=1, index=idx)
    y = torch.gather(y, dim=1, index=idx)

    coefficients, G = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coefficients)

    X = X.squeeze(0).cpu().numpy()
    y_pred = y_pred.squeeze(0).cpu().numpy()
    y = y.squeeze(0).cpu().numpy()

    example_X = example_X.squeeze(0).cpu().numpy()
    example_y = example_y.squeeze(0).cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(X, y, label="True")
    ax.plot(X, y_pred, label="Predicted")
    ax.scatter(example_X, example_y, label="Data", color="red")
    ax.legend()
    plt.savefig("random_fourier.png")

    basis_eval = model.basis_functions(torch.from_numpy(X).to(device).unsqueeze(0))
    basis_eval = basis_eval.squeeze(0).squeeze(1).detach().cpu().numpy()

    n_cols = 2
    n_rows = math.ceil(basis_eval.shape[-1] / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    axes = axes.flatten()
    for i in range(basis_eval.shape[-1]):
        axes[i].plot(X, basis_eval[:, i])
        axes[i].set_title(f"Basis {i}")
    for i in range(basis_eval.shape[-1], len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("random_fourier_basis.png")
