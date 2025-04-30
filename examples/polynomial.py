import torch

from datasets import load_dataset

from torch.utils.data import DataLoader

from function_encoder.model.mlp import MLP, MultiHeadedMLP
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import fit

import tqdm

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load dataset

ds = load_dataset("ajthor/polynomial")
ds = ds.with_format("torch", device=device)

dataloader = DataLoader(ds["train"], batch_size=50)


# Create basis functions

# basis_functions = BasisFunctions([MLP(layer_sizes=[1, 32, 1]) for _ in range(8)])
basis_functions = MultiHeadedMLP(layer_sizes=[1, 32, 1], num_heads=8)

# Create model

model = FunctionEncoder(basis_functions)
model.to(device)


# Train model


def loss_function(model, batch):
    X, y = batch["X"].to(device), batch["y"].to(device)
    X = X.unsqueeze(-1)  # Fix for 1D input
    y = y.unsqueeze(-1)  # Fix for 1D input

    coefficients, G = model.compute_coefficients(X, y)
    y_pred = model(X, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, y)
    norm_loss = basis_normalization_loss(G)

    return pred_loss + norm_loss


model = fit(model=model, ds=dataloader, loss_function=loss_function)


# Plot results

with torch.no_grad():

    point = ds["train"][0]
    example_X = point["X"].unsqueeze(-1)  # Fix for 1D input
    example_y = point["y"].unsqueeze(-1)  # Fix for 1D input

    # Add leading batch dimension
    example_X = example_X.unsqueeze(0)
    example_y = example_y.unsqueeze(0)

    X = torch.linspace(-1, 1, 100)
    X = X.unsqueeze(1)  # Fix for 1D input
    X = X.unsqueeze(0)  # Add batch dimension
    coefficients, _ = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coefficients)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot true function
    ax.plot(example_X[0], example_y[0], label="True")
    ax.scatter(example_X[0], example_y[0], label="Data", color="red")

    ax.plot(X[0], y_pred[0], label="Prediction")

    plt.legend()

    plt.show()
