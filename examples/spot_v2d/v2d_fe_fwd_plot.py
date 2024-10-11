import torch

from datasets import load_from_disk, Dataset

from function_encoder_torch.model.mlp import MLP
from function_encoder_torch.function_encoder import FunctionEncoder

from safetensors.torch import load_model

import matplotlib.pyplot as plt

import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load dataset
ds = load_from_disk("examples/spot_v2d/data").with_format("torch")

# Create model
basis_functions = torch.nn.ModuleList([MLP([20, 128, 128, 14]) for _ in range(100)])
model = FunctionEncoder(basis_functions)
load_model(model, "examples/spot_v2d/v2d_fe_fwd.safetensors")

model.to(device)

# Plot results

# Get one function from the dataset
for bag_id in tqdm.tqdm(range(ds.num_rows)):
    bag = ds[bag_id]

    with torch.no_grad():

        sample_size = 100
        sample_idx = torch.randperm(bag["x"].size(0) - 1)[:sample_size]

        u_cols = ["u_vx", "u_vy", "u_vz", "u_wx", "u_wy", "u_wz"]
        X = torch.stack([bag[key][sample_idx] for key in ds.column_names])
        y = torch.stack(
            [bag[key][sample_idx + 1] for key in ds.column_names if key not in u_cols]
        )

        X = X.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        X = X.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        coefficients = model.compute_coefficients(X, y)

        # Predict the trajectory
        ground_truth = Dataset.from_dict(bag)
        ground_truth = ground_truth.with_format("torch")

        x0 = ground_truth[0]
        x0 = torch.tensor(
            [x0[key] for key in ground_truth.column_names if key not in u_cols]
        )
        trajectory = [x0]

        for t in range(ground_truth.num_rows - 1):
            x = trajectory[t].unsqueeze(0).unsqueeze(0).to(device)
            # Get the corresponding control inputs
            u = torch.tensor(
                [
                    ground_truth[t][key]
                    for key in ground_truth.column_names
                    if key in u_cols
                ]
            )
            u = u.unsqueeze(0).unsqueeze(0).to(device)

            state = torch.cat((x, u), dim=2)

            y = model(state, coefficients)

            trajectory.append(y.squeeze())

    trajectory = torch.stack(trajectory)

    # Plot the results
    fig, axs = plt.subplots(14, 1, figsize=(8, 20))

    for i, key in enumerate(ground_truth.column_names):
        if key in u_cols:
            continue

        axs[i].plot(ground_truth[key], label="Ground truth")
        axs[i].plot(trajectory[:, i], label="Prediction")

        axs[i].set_ylabel(key)

        axs[i].set_xticks([])
        axs[i].set_xticklabels([])
        axs[i].set_xlabel("")

        # axs[i].set_xlim(0, 100)

    plt.savefig(f"examples/spot_v2d/results/trajectory_{bag_id}.png")

    plt.close()

    # Plot the error
    fig, axs = plt.subplots(14, 1, figsize=(8, 20))

    for i, key in enumerate(ground_truth.column_names):
        if key in u_cols:
            continue

        error = torch.abs(ground_truth[key] - trajectory[:, i])

        axs[i].plot(error, label="Error")

        axs[i].set_ylabel(key)

        axs[i].set_xticks([])
        axs[i].set_xticklabels([])
        axs[i].set_xlabel("")

        # axs[i].set_xlim(0, 100)

    plt.savefig(f"examples/spot_v2d/results/error_{bag_id}.png")

    plt.close()
