import torch

from torch.utils.data import DataLoader
from datasets.derivative_operator import DerivativeOperatorDataset

from function_encoder.model.mlp import MultiHeadedMLP
from function_encoder.function_encoder import FunctionEncoder
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import train_step

import tqdm


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# Load dataset

dataset = DerivativeOperatorDataset(n_points=100, n_example_points=10)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)


# Create models

input_basis_functions = MultiHeadedMLP(layer_sizes=[1, 32, 1], num_heads=8)
input_function_encoder = FunctionEncoder(input_basis_functions).to(device)


output_basis_functions = MultiHeadedMLP(layer_sizes=[1, 32, 1], num_heads=8)
output_function_encoder = FunctionEncoder(output_basis_functions).to(device)


# Train model


# Train the input function encoder
def input_loss_function(model, batch):
    X, u, _, _, example_X, example_u, _, _ = batch
    X = X.to(device)
    u = u.to(device)
    example_X = example_X.to(device)
    example_u = example_u.to(device)

    coefficients, G = model.compute_coefficients(example_X, example_u)
    u_pred = model(X, coefficients)

    pred_loss = torch.nn.functional.mse_loss(u_pred, u)
    norm_loss = basis_normalization_loss(G)

    return pred_loss + norm_loss


input_optimizer = torch.optim.Adam(input_function_encoder.parameters(), lr=1e-3)
num_epochs = 1000
with tqdm.tqdm(range(num_epochs)) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(iter(dataloader))
        loss = train_step(
            input_function_encoder, input_optimizer, batch, input_loss_function
        )
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})


# Train the output function encoder
def output_loss_function(model, batch):
    _, _, Y, s, _, _, example_Y, example_s = batch
    Y = Y.to(device)
    s = s.to(device)
    example_Y = example_Y.to(device)
    example_s = example_s.to(device)

    coefficients, G = model.compute_coefficients(example_Y, example_s)
    s_pred = model(Y, coefficients)

    pred_loss = torch.nn.functional.mse_loss(s_pred, s)
    norm_loss = basis_normalization_loss(G)

    return pred_loss + norm_loss


output_optimizer = torch.optim.Adam(output_function_encoder.parameters(), lr=1e-3)
num_epochs = 1000
with tqdm.tqdm(range(num_epochs)) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(iter(dataloader))
        loss = train_step(
            output_function_encoder, output_optimizer, batch, output_loss_function
        )
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})


# Train the oeprator
XTX = torch.zeros((8, 8), device=device)
XTY = torch.zeros((8, 8), device=device)

num_epochs = 100
with torch.no_grad():
    with tqdm.tqdm(range(num_epochs)) as tqdm_bar:
        for epoch in tqdm_bar:
            # Get a batch from the dataloader
            batch = next(iter(dataloader))

            # Extract the input and output function encoders
            X, u, Y, s, example_X, example_u, example_Y, example_s = batch
            X = X.to(device)
            u = u.to(device)
            Y = Y.to(device)
            s = s.to(device)
            example_X = example_X.to(device)
            example_u = example_u.to(device)
            example_Y = example_Y.to(device)
            example_s = example_s.to(device)

            # Compute coefficients for the input and output functions
            input_coefficients, _ = input_function_encoder.compute_coefficients(
                example_X, example_u
            )
            output_coefficients, _ = output_function_encoder.compute_coefficients(
                example_Y, example_s
            )

            # Compute the normal equations in chunks
            XTX += torch.einsum("bk,bl->kl", input_coefficients, input_coefficients)
            XTY += torch.einsum("bk,bl->kl", input_coefficients, output_coefficients)

        XTX += 1e-6 * torch.eye(8, device=device)  # Regularization term

    # Compute the operator using the normal equations
    operator = torch.linalg.lstsq(XTX, XTY).solution


# Plot

import matplotlib.pyplot as plt

input_function_encoder.eval()
output_function_encoder.eval()

with torch.no_grad():
    dataloader = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader))

    X, u, Y, s, example_X, example_u, example_Y, example_s = batch
    X = X.to(device)
    u = u.to(device)
    Y = Y.to(device)
    s = s.to(device)
    example_X = example_X.to(device)
    example_u = example_u.to(device)
    example_Y = example_Y.to(device)
    example_s = example_s.to(device)

    idx = torch.argsort(X, dim=1, descending=False)
    X = torch.gather(X, dim=1, index=idx)
    u = torch.gather(u, dim=1, index=idx)

    idx = torch.argsort(Y, dim=1, descending=False)
    Y = torch.gather(Y, dim=1, index=idx)
    s = torch.gather(s, dim=1, index=idx)

    input_coefficients, _ = input_function_encoder.compute_coefficients(X, u)
    output_coefficients = torch.einsum("bk,kl->bl", input_coefficients, operator)

    s_pred = output_function_encoder(Y, output_coefficients)

    # Detach from device and squeeze
    X = X.squeeze().cpu().detach().numpy()
    u = u.squeeze().cpu().detach().numpy()
    Y = Y.squeeze().cpu().detach().numpy()
    s = s.squeeze().cpu().detach().numpy()
    s_pred = s_pred.squeeze().cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(X, u, label="Original")
    ax.scatter(X, u, label="Data", color="red")

    ax.plot(Y, s, label="True")
    ax.plot(Y, s_pred, label="Prediction")

    plt.legend()
    plt.show()
