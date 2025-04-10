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

ds = load_dataset("ajthor/derivative_polynomial")
ds = ds.with_format("torch", device=device)

dataloader = DataLoader(ds["train"], batch_size=50)


# Create models

input_basis_functions = MultiHeadedMLP(layer_sizes=[1, 32, 1], num_heads=8)
input_function_encoder = FunctionEncoder(input_basis_functions)


output_basis_functions = MultiHeadedMLP(layer_sizes=[1, 32, 1], num_heads=8)
output_function_encoder = FunctionEncoder(output_basis_functions)


# Train model


# Train the input function encoder
def input_loss_function(model, batch):
    X, y = batch["X"].to(device), batch["f"].to(device)
    X = X.unsqueeze(-1)  # Fix for 1D input
    y = y.unsqueeze(-1)  # Fix for 1D input

    coefficients = model.compute_coefficients(X, y)
    y_pred = model(X, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, y)
    norm_loss = basis_normalization_loss(model.basis_functions(X))

    return pred_loss + norm_loss


input_function_encoder = fit(
    model=input_function_encoder,
    ds=dataloader,
    loss_function=input_loss_function,
    epochs=1000,
)


# Train the output function encoder
def output_loss_function(model, batch):
    X, y = batch["Y"].to(device), batch["Tf"].to(device)
    X = X.unsqueeze(-1)  # Fix for 1D input
    y = y.unsqueeze(-1)  # Fix for 1D input

    coefficients = model.compute_coefficients(X, y)
    y_pred = model(X, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, y)
    norm_loss = basis_normalization_loss(model.basis_functions(X))

    return pred_loss + norm_loss


output_function_encoder = fit(
    model=output_function_encoder,
    ds=dataloader,
    loss_function=output_loss_function,
    epochs=1000,
)

# Train the oeprator
ds_subset = ds["train"].take(1000)

source_coefficients = input_function_encoder.compute_coefficients(
    ds_subset["X"].unsqueeze(-1).to(device), ds_subset["f"].unsqueeze(-1).to(device)
)

target_coefficients = output_function_encoder.compute_coefficients(
    ds_subset["Y"].unsqueeze(-1).to(device), ds_subset["Tf"].unsqueeze(-1).to(device)
)

operator = torch.linalg.lstsq(source_coefficients, target_coefficients).solution


# Plot

input_function_encoder.eval()
output_function_encoder.eval()

point = ds["train"].take(1)[0]

X = point["X"]
f = point["f"]
Y = point["Y"]
Tf = point["Tf"]

idx = torch.argsort(X, dim=0).squeeze()
X = X[idx]
f = f[idx]

idx = torch.argsort(Y, dim=0).squeeze()
Y = Y[idx]
Tf = Tf[idx]

X = X.unsqueeze(-1).unsqueeze(0)
f = f.unsqueeze(-1).unsqueeze(0)
Y = Y.unsqueeze(-1).unsqueeze(0)
Tf = Tf.unsqueeze(-1).unsqueeze(0)

input_coefficients = input_function_encoder.compute_coefficients(X, f)
output_coefficients = torch.einsum("bk,kl->bl", input_coefficients, operator)

Tf_pred = output_function_encoder(Y, output_coefficients)

# Detach from device and squeeze
X = X.squeeze().cpu().detach().numpy()
f = f.squeeze().cpu().detach().numpy()
Y = Y.squeeze().cpu().detach().numpy()
Tf = Tf.squeeze().cpu().detach().numpy()
Tf_pred = Tf_pred.squeeze().cpu().detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, f, label="Original")
ax.scatter(X, f, label="Data", color="red")

ax.plot(Y, Tf, label="True")
ax.plot(Y, Tf_pred, label="Prediction")

plt.legend()
plt.show()
