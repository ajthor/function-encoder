import torch

from torch.utils.data import DataLoader
from datasets.polynomial import PolynomialDataset

from function_encoder.model.mlp import MLP
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.losses import basis_normalization_loss
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

dataset = PolynomialDataset(n_points=100, n_example_points=10)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model


def basis_function_factory():
    return MLP(layer_sizes=[1, 32, 1])


num_basis = 8
# Only use one basis function initially for progressive training
basis_functions = BasisFunctions(basis_function_factory())

model = FunctionEncoder(basis_functions).to(device)

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
    # norm_loss = basis_normalization_loss(G)

    return pred_loss  # + norm_loss


# Train the first basis function
num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.tqdm(range(num_epochs), desc=f"basis 1/{num_basis}") as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})

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
            tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})

# Plot results

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    dataloader_eval = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader_eval))

    X, y, example_X, example_y = batch
    X = X.to(device)
    y = y.to(device)
    example_X = example_X.to(device)
    example_y = example_y.to(device)

    idx = torch.argsort(X, dim=1, descending=False)
    X = torch.gather(X, dim=1, index=idx)
    y = torch.gather(y, dim=1, index=idx)

    coefficients, _ = model.compute_coefficients(example_X, example_y)
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

    # Visualize individual basis functions
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    X_plot = torch.linspace(-1, 1, 100).unsqueeze(1).unsqueeze(0).to(device)
    for i, basis_fn in enumerate(model.basis_functions.basis_functions):
        if i >= num_basis:
            break
        basis_output = basis_fn(X_plot)
        axes[i].plot(X_plot[0].cpu().numpy(), basis_output[0].detach().cpu().numpy())
        axes[i].set_title(f"Basis Function {i+1}")
    plt.tight_layout()
    plt.show()
