from datasets import load_dataset

import torch

from torch.utils.data import DataLoader

from function_encoder_torch.model.mlp import MLP
from function_encoder_torch.function_encoder import FunctionEncoder

import tqdm

import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load dataset

ds = load_dataset("ajthor/polynomial")
ds = ds.with_format("torch")

dataloader = DataLoader(ds["train"], batch_size=12)


# Create basis functions

basis_functions = torch.nn.ModuleList([MLP([1, 32, 1]) for _ in range(8)])

# Create model

model = FunctionEncoder(basis_functions)
model.to(device)


# Train model

epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

with tqdm.tqdm(range(epochs)) as tqdm_bar:
    for epoch in tqdm_bar:

        for batch in dataloader:

            X, y = batch["X"].to(device), batch["y"].to(device)
            X = X.unsqueeze(1)  # Fix for 1D input
            y = y.unsqueeze(1)  # Fix for 1D input

            coefficients = model.compute_coefficients(X, y)
            y_hat = model(X, coefficients)

            loss = torch.nn.functional.mse_loss(y_hat, y)

            loss.backward()

            # Backpropagation with gradient accumulation
            if epoch % 50 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            break

        tqdm_bar.set_postfix_str(f"Loss {loss.item()}")


# Plot results

with torch.no_grad():

    point = ds["train"][0]
    example_X = point["X"].unsqueeze(1)  # Fix for 1D input
    example_y = point["y"].unsqueeze(1)  # Fix for 1D input

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
