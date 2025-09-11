import torch
from torch.utils.data import DataLoader

from datasets.kepler import KeplerDataset, kepler

from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
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

dataset = KeplerDataset(
    integrator=rk4_step,
    n_points=1000,
    n_example_points=100,
    dt_range=(0.1, 0.1),
    device=torch.device(device),
)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model

n_basis = 10
basis_functions = BasisFunctions(
    *[
        NeuralODE(
            ode_func=ODEFunc(
                model=MLP(layer_sizes=[5, 64, 64, 4])
            ),  # Reduced from 9->5 and 8->4
            integrator=rk4_step,
        )
        for _ in range(n_basis)
    ]
)

model = FunctionEncoder(basis_functions).to(device)

# Train model


def loss_function(model, batch):
    M_central, y0, dt, y1, y0_example, dt_example, y1_example = batch
    # Data is already on the correct device from the dataset

    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    pred = model((y0, dt), coefficients=coefficients)

    pred_loss = torch.nn.functional.mse_loss(pred, y1)

    return pred_loss


num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.trange(num_epochs) as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        tqdm_bar.set_postfix_str(f"loss: {loss:.2e}")

# Plot a grid of evaluations

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    # Generate a single batch of functions for plotting
    dataloader = DataLoader(dataset, batch_size=9)
    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)

    M_central, y0, dt, y1, y0_example, dt_example, y1_example = batch
    # Data is already on the correct device from the dataset

    # Precompute the coefficients for the batch
    coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)

    fig, ax = plt.subplots(3, 3, figsize=(12, 12))

    for i in range(3):
        for j in range(3):
            # Plot a single trajectory
            _M_central = M_central[i * 3 + j]

            # Generate initial conditions using orbital parameters
            from datasets.kepler import generate_kepler_states_batch

            _y0 = generate_kepler_states_batch(
                _M_central.item(),
                dataset.a_range,
                dataset.e_range,
                1,
                device=torch.device(device),
            )

            # We use the coefficients that we computed before
            _c = coefficients[i * 3 + j].unsqueeze(0)
            s = 0.01  # Time step for simulation
            n = int(2.0 / s)  # Simulate for 2 time units (longer for orbits)
            _dt = torch.tensor([s], device=device)

            # Integrate the true trajectory
            x = _y0.clone()
            y = [x]
            for k in range(n):
                x = rk4_step(kepler, x, _dt, M_central=_M_central) + x
                y.append(x)
            y = torch.cat(y, dim=0)
            y = y.detach().cpu().numpy()

            # Integrate the predicted trajectory
            x = _y0.clone()
            x = x.unsqueeze(1)
            _dt = _dt.unsqueeze(0)
            pred = [x]
            for k in range(n):
                x = model((x, _dt), coefficients=_c) + x
                pred.append(x)
            pred = torch.cat(pred, dim=1)
            pred = pred.detach().cpu().numpy()

            # Plot trajectory
            ax[i, j].set_xlim(-4, 4)
            ax[i, j].set_ylim(-4, 4)
            ax[i, j].set_aspect("equal")

            # True trajectory
            (_t,) = ax[i, j].plot(
                y[:, 0], y[:, 1], "b-", alpha=0.7, linewidth=1, label="True"
            )

            # Predicted trajectory
            (_p,) = ax[i, j].plot(
                pred[0, :, 0],
                pred[0, :, 1],
                "r--",
                alpha=0.9,
                linewidth=2,
                label="Predicted",
            )

            # Mark central body at origin
            ax[i, j].plot(0, 0, "ko", markersize=8, label="Central Body")

            # Mark initial position
            ax[i, j].plot(y[0, 0], y[0, 1], "go", markersize=6, label="Start")

            ax[i, j].grid(True, alpha=0.3)
            ax[i, j].set_title(f"M={_M_central.item():.2f}", fontsize=10)

    # Add legend to the entire figure
    fig.legend(
        handles=[_t, _p],
        labels=["True", "Predicted"],
        loc="outside upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
    )

    plt.suptitle("Kepler Problem: Test Particle Orbiting Central Mass", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), "kepler_model.pth")
