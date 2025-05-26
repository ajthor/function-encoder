import torch

from torch.utils.data import DataLoader
from datasets.polynomial import PolynomialDataset

from function_encoder.model.mlp import MLP, MultiHeadedMLP
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import fit, train_step

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

# basis_functions = BasisFunctions(*[MLP(layer_sizes=[1, 32, 1]) for _ in range(8)])
basis_functions = MultiHeadedMLP(
    layer_sizes=[1, 32, 1], num_heads=8, activation=torch.nn.Tanh()
)

model = FunctionEncoder(basis_functions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train model


def loss_function(model, batch):
    X, y, example_X, example_y = batch
    X = X.to(device)
    y = y.to(device)
    example_X = example_X.to(device)
    example_y = example_y.to(device)

    coefficients, G = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, y)
    norm_loss = basis_normalization_loss(G)

    return pred_loss + norm_loss


num_epochs = 1000
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

    coefficients, G = model.compute_coefficients(example_X, example_y)
    y_pred = model(X, coefficients)

    X = X.squeeze(0).cpu().numpy()
    y_pred = y_pred.squeeze(0).cpu().numpy()
    y = y.squeeze(0).cpu().numpy()

    example_X = example_X.squeeze(0).cpu().numpy()
    example_y = example_y.squeeze(0).cpu().numpy()

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="True")
    ax.scatter(X, y_pred, label="Predicted")
    ax.scatter(example_X, example_y, label="Data", color="red")
    ax.legend()
    plt.show()
