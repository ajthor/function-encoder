import torch

from datasets import load_dataset

from torch.utils.data import DataLoader

from function_encoder.model.mlp import MLP, MultiHeadedMLP
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
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


def basis_function_factory():
    return MLP(layer_sizes=[1, 32, 1]).to(device)


n_basis = 8
basis_functions = BasisFunctions(torch.nn.ModuleList([basis_function_factory()]))


# Create model

model = FunctionEncoder(basis_functions)
model.to(device)


# Train model


def loss_function(model, batch):
    X, y = batch["X"].to(device), batch["y"].to(device)
    X = X.unsqueeze(-1)  # Fix for 1D input
    y = y.unsqueeze(-1)  # Fix for 1D input

    coefficients = model.compute_coefficients(X, y)
    y_pred = model(X, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, y)

    return pred_loss


# Train the first basis function
model = fit(model=model, ds=dataloader, loss_function=loss_function)

# Train the remaining basis functions progressively
for i in tqdm.tqdm(range(n_basis - 1)):

    # Freeze all parameters except the new basis function
    for param in model.parameters():
        param.requires_grad = False

    # Create new basis function
    new_basis_function = basis_function_factory()

    for param in new_basis_function.parameters():
        param.requires_grad = True

    model.basis_functions.basis_functions.append(new_basis_function)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    model.train()

    with tqdm.tqdm(range(1000), leave=True) as tqdm_bar:
        for epoch in tqdm_bar:
            for batch in dataloader:
                optimizer.zero_grad()
                loss = loss_function(model, batch)
                loss.backward()
                optimizer.step()
                break

            if epoch % 10 == 0:
                tqdm_bar.set_postfix_str(f"loss {loss.item():.2e}")


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
    coefficients = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coefficients)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot true function
    ax.plot(example_X[0], example_y[0], label="True")
    ax.scatter(example_X[0], example_y[0], label="Data", color="red")

    ax.plot(X[0], y_pred[0], label="Prediction")

    plt.legend()

    plt.show()

    # Visualize individual basis functions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, basis_fn in enumerate(model.basis_functions.basis_functions):
        if i >= n_basis:
            break

        # Generate the basis function output
        basis_output = basis_fn(X)

        axes[i].plot(X[0], basis_output[0])
        axes[i].set_title(f"Basis Function {i+1}")

    plt.tight_layout()
    plt.show()
