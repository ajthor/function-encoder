import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from datasets.polynomial import PolynomialDataset

from function_encoder.model.mlp import StackedMLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import basis_normalization_loss, matryoshka_loss
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

num_basis = 4

basis_functions = StackedMLP(layer_sizes=[1, 64, 64, 1], num_heads=num_basis)

model = FunctionEncoder(basis_functions).to(device)

# Train model

matryoshka_sizes = list(range(1, num_basis + 1))


def loss_function(model, batch):
    X, y, example_X, example_y = batch
    X = X.to(device)
    y = y.to(device)
    example_X = example_X.to(device)
    example_y = example_y.to(device)

    coefficients, G = model.compute_coefficients(example_X, example_y)

    return matryoshka_loss(model, X, y, coefficients, matryoshka_sizes)


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
    plt.show()

    basis_eval = model.basis_functions(torch.from_numpy(X).to(device).unsqueeze(0))
    basis_eval = basis_eval.squeeze(0).squeeze(1).detach().cpu().numpy()
    fig, ax = plt.subplots()
    for i in range(basis_eval.shape[-1]):
        ax.plot(X, basis_eval[:, i], label=f"Basis {i}")
    ax.legend()
    plt.show()
