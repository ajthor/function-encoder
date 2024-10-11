from datasets import load_dataset, Dataset, concatenate_datasets

import numpy as np

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from function_encoder_torch.model.mlp import MLP
from function_encoder_torch.function_encoder import FunctionEncoder

from safetensors.torch import save_model

import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load dataset

odom_ds = load_dataset("sarahnator/V2D", "odom_aligned", split="train")
odom_ds = odom_ds.with_format("numpy")


def generate_derived_odom_ds():
    for bag_id in odom_ds.unique("id"):
        bag_ds = odom_ds.filter(lambda x: x == bag_id, input_columns="id")

        qx = bag_ds["qx"]
        qy = bag_ds["qy"]
        qz = bag_ds["qz"]
        qw = bag_ds["qw"]

        # Convert the quaternions to Euler angles
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = np.arcsin(2 * (qw * qy - qz * qx))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        roll = np.unwrap(roll)
        pitch = np.unwrap(pitch)
        yaw = np.unwrap(yaw)

        # Convert this back to quaternions
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.cos(yaw / 2)

        theta = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        # Unwrap the quaternion
        qw = np.unwrap(qw)
        qx = np.unwrap(qx)
        qy = np.unwrap(qy)
        qz = np.unwrap(qz)

        theta = np.unwrap(theta)

        # # Process the quaternions.
        # qx = np.unwrap(bag_ds["qx"])
        # qy = np.unwrap(bag_ds["qy"])
        # qz = np.unwrap(bag_ds["qz"])
        # qw = np.unwrap(bag_ds["qw"])

        # theta = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        yield {
            "id": bag_id,
            "time": bag_ds["time"],
            # "terrain": bag_ds["terrain"],
            "x": bag_ds["x"],
            "y": bag_ds["y"],
            "z": bag_ds["z"],
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "theta": theta,
            "vx": bag_ds["vx"],
            "vy": bag_ds["vy"],
            "vz": bag_ds["vz"],
            "wx": bag_ds["wx"],
            "wy": bag_ds["wy"],
            "wz": bag_ds["wz"],
        }


derived_odom_ds = Dataset.from_generator(generate_derived_odom_ds)
derived_odom_ds = derived_odom_ds.with_format("torch")


cmd_vel_ds = load_dataset("sarahnator/V2D", "cmd_vel_aligned", split="train")
cmd_vel_ds = cmd_vel_ds.with_format("numpy")


def generate_derived_cmd_vel_ds():
    for bag_id in cmd_vel_ds.unique("id"):
        bag_ds = cmd_vel_ds.filter(lambda x: x == bag_id, input_columns="id")

        yield {
            "id": bag_id,
            "time": bag_ds["time"],
            # "terrain": bag_ds["terrain"],
            "u_vx": bag_ds["vx"],
            "u_vy": bag_ds["vy"],
            "u_vz": bag_ds["vz"],
            "u_wx": bag_ds["wx"],
            "u_wy": bag_ds["wy"],
            "u_wz": bag_ds["wz"],
        }


derived_cmd_vel_ds = Dataset.from_generator(generate_derived_cmd_vel_ds)
derived_cmd_vel_ds = derived_cmd_vel_ds.with_format("torch")


derived_ds = concatenate_datasets(
    [
        derived_odom_ds.remove_columns(["id", "time"]),
        derived_cmd_vel_ds.remove_columns(["id", "time"]),
    ],
    axis=1,
)
derived_ds.set_format(type="torch")


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_size=100):
        self.dataset = dataset
        self.sample_size = sample_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample_idx = torch.randperm(self.dataset[idx]["x"].size(0) - 1)[
            : self.sample_size
        ]

        _x_t = torch.stack(
            [
                self.dataset[idx][key][sample_idx]
                for key in self.dataset.column_names
                if key not in ["u_vx", "u_vy", "u_vz", "u_wx", "u_wy", "u_wz"]
            ]
        )
        _u_t = torch.stack(
            [
                self.dataset[idx][key][sample_idx]
                for key in self.dataset.column_names
                if key in ["u_vx", "u_vy", "u_vz", "u_wx", "u_wy", "u_wz"]
            ]
        )
        _y_t = torch.stack(
            [
                self.dataset[idx][key][sample_idx + 1]
                for key in self.dataset.column_names
                if key not in ["u_vx", "u_vy", "u_vz", "u_wx", "u_wy", "u_wz"]
            ]
        )

        example_X = torch.cat((_x_t, _y_t), dim=0)
        example_y = _u_t

        sample_idx = torch.randperm(self.dataset[idx]["x"].size(0) - 1)[
            : self.sample_size
        ]

        _x_t = torch.stack(
            [
                self.dataset[idx][key][sample_idx]
                for key in self.dataset.column_names
                if key not in ["u_vx", "u_vy", "u_vz", "u_wx", "u_wy", "u_wz"]
            ]
        )
        _u_t = torch.stack(
            [
                self.dataset[idx][key][sample_idx]
                for key in self.dataset.column_names
                if key in ["u_vx", "u_vy", "u_vz", "u_wx", "u_wy", "u_wz"]
            ]
        )
        _y_t = torch.stack(
            [
                self.dataset[idx][key][sample_idx + 1]
                for key in self.dataset.column_names
                if key not in ["u_vx", "u_vy", "u_vz", "u_wx", "u_wy", "u_wz"]
            ]
        )

        X = torch.cat((_x_t, _y_t), dim=0)
        y = _u_t

        return X, y, example_X, example_y


time_series_dataset = TimeSeriesDataset(derived_ds, sample_size=100)

dataloader = DataLoader(time_series_dataset, batch_size=20)

# Create basis functions
basis_functions = torch.nn.ModuleList(
    [MLP([28, 128, 128, 6], activation=torch.nn.ReLU()) for _ in range(100)]
)

# Create model
model = FunctionEncoder(basis_functions)
model.to(device)

writer = SummaryWriter()


# Train model
epochs = 2000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

with tqdm.tqdm(range(epochs)) as tqdm_bar:
    for epoch in tqdm_bar:

        for batch in dataloader:

            X, y, example_X, example_y = batch
            X = X.to(device)
            y = y.to(device)
            example_X = example_X.to(device)
            example_y = example_y.to(device)

            # Permute the dimensions to match the model's expectations
            X = X.permute(0, 2, 1)
            y = y.permute(0, 2, 1)
            example_X = example_X.permute(0, 2, 1)
            example_y = example_y.permute(0, 2, 1)

            coefficients = model.compute_coefficients(example_X, example_y)
            y_hat = model(X, coefficients)

            loss = torch.nn.functional.mse_loss(y_hat, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar("Loss", loss.item(), epoch)

            # # Backpropagation with gradient accumulation
            # if epoch % 50 == 0 and epoch > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #     optimizer.step()
            #     optimizer.zero_grad()

            #     # writer.add_scalar("Loss", loss.item(), epoch)

            break

        tqdm_bar.set_postfix_str(f"Loss {loss.item()}")


# Save model
save_model(model, "examples/spot_v2d/v2d_fe_inv.safetensors")
