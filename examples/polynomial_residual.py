import torch

from torch.utils.data import DataLoader
from datasets.polynomial import PolynomialDataset

from function_encoder.model.mlp import MultiHeadedMLP, MLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import basis_normalization_loss, residual_loss
from function_encoder.utils.training import train_step

import tqdm
from tqdm import trange


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


torch.manual_seed(42)

# Load dataset

dataset = PolynomialDataset(n_points=100, n_example_points=10)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model

basis_functions = MultiHeadedMLP(layer_sizes=[1, 32, 1], num_heads=8)
residual_function = MLP(layer_sizes=[1, 32, 1])

model = FunctionEncoder(basis_functions, residual_function=residual_function).to(device)

# Train model


def loss_function(model, batch):
    X, y, example_X, example_y = batch
    X = X.to(device)
    y = y.to(device)
    example_X = example_X.to(device)
    example_y = example_y.to(device)

    coefficients, G = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coefficients)

    return (
        torch.nn.functional.mse_loss(y_pred, y)
        + basis_normalization_loss(G)
        + residual_loss(model, X, y)
    )


num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.tqdm(range(num_epochs)) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})


# Plot an evaluation of the model

import matplotlib.pyplot as plt

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
    plt.show()
