from typing import List
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.van_der_pol import VanDerPolDataset, van_der_pol
from function_encoder.model.mlp import MLP, MultiHeadedMLP
from function_encoder.model.neural_ode import NeuralODE, NeuralODEFast, ODEFunc, rk4_step, rk4_step_fast
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder, FunctionEncoderFast

# ============================================================
# SETUP
# ============================================================

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.manual_seed(11)

# Dataset
n = 100
dataset = VanDerPolDataset(n_points=n, n_example_points=100, dt_range=(0.1, 0.1))
dataloader = DataLoader(dataset, batch_size=100)
batch = next(iter(dataloader))
mu, y0, dt, y1, y0_example, dt_example, y1_example = [b.to(device) for b in batch]

# ============================================================
# MODEL LOADERS
# ============================================================

def load_original_model(device):
    """Load the original FunctionEncoder model."""
    n_basis = 10
    basis_functions = BasisFunctions(
        *[
            NeuralODE(
                ode_func=ODEFunc(model=MLP(layer_sizes=[3, 64, 64, 2])),
                integrator=rk4_step,
            )
            for _ in range(n_basis)
        ]
    )
    model = FunctionEncoder(basis_functions).to(device)
    model.load_state_dict(torch.load("van_der_pol_model.pth", map_location=device))
    return model


def load_speedup_model(model_path: str, device):
    """Load a fast model given its checkpoint path."""
    n_basis = 10
    basis_functions = MultiHeadedMLP(layer_sizes=[3, 64, 64, 2], num_heads=n_basis)
    model = NeuralODEFast(
        ode_func=FunctionEncoderFast(basis_functions=ODEFunc(model=basis_functions)),
        integrator=rk4_step_fast,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# ============================================================
# EVALUATION ROUTINES
# ============================================================

def rollout_true_trajectory(y0, dt, mu):
    """Integrate the true van der Pol trajectory."""
    n = dt.shape[1]
    y = y0.clone()
    x = y[:, 0, :].unsqueeze(1)
    for k in range(n - 1):
        x = rk4_step(van_der_pol, x, dt[:, k].unsqueeze(1), mu=mu) + x
        y[:, k + 1, :] = x.squeeze(1)
    return y


def rollout_model(model, y0, dt, coefficients, fast=False):
    """Integrate predicted trajectory for a model."""
    n = dt.shape[1]
    y_pred = y0.clone()
    x = y_pred[:, 0, :].unsqueeze(1)
    start = time.time()
    for k in range(n - 1):
        if fast:
            x = model((x, dt[:, k].unsqueeze(1), coefficients)) + x
        else:
            x = model((x, dt[:, k].unsqueeze(1)), coefficients=coefficients) + x
        y_pred[:, k + 1, :] = x.squeeze(1)
    duration = time.time() - start
    return y_pred, duration


def compute_error_metrics(y_pred, y_true):
    """Return median and 10th/90th percentile rollout MSE."""
    err = torch.mean((y_pred - y_true) ** 2, dim=2).detach().cpu().numpy()
    med = np.median(err, axis=0)
    minv = np.percentile(err, 10, axis=0)
    maxv = np.percentile(err, 90, axis=0)
    return med, minv, maxv


# ============================================================
# MAIN EVALUATION
# ============================================================

# True reference rollout
y_true = rollout_true_trajectory(y0, dt, mu)

# Load baseline
model_og = load_original_model(device)
model_og.eval()
with torch.no_grad():
    coeffs_og, _ = model_og.compute_coefficients((y0_example, dt_example), y1_example)

# Rollout baseline
pred_og, time_og = rollout_model(model_og, y0, dt, coeffs_og, fast=False)
loss_og = torch.nn.functional.mse_loss(pred_og, y_true)
med_og, min_og, max_og = compute_error_metrics(pred_og, y_true)

print(f"[Original] Time: {time_og:.3f}s | Loss: {loss_og:.3e}")

# ============================================================
# LOOP OVER MULTIPLE NEW MODELS
# ============================================================

model_paths = [
    "van_der_pol_speedup_model_n_basis=10_epochs=1000.pth",
    "van_der_pol_speedup_model_n_basis=10_epochs=2000.pth",
    "van_der_pol_speedup_model_n_basis=10_epochs=3000.pth",
    "van_der_pol_speedup_model_n_basis=10_epochs=4000.pth",
    "van_der_pol_speedup_model_n_basis=10_epochs=5000.pth",
    "van_der_pol_speedup_model_n_basis=10_epochs=6000.pth",
    "van_der_pol_speedup_model_n_basis=10_epochs=7000.pth",
    "van_der_pol_speedup_model_n_basis=10_epochs=8000.pth",
    "van_der_pol_speedup_model_n_basis=10_epochs=9000.pth",
]

# colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(model_paths)))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
timesteps = np.arange(0, n * 0.1, 0.1)

plt.figure(figsize=(10, 6))
plt.plot(timesteps, med_og, color="k", label="Original")
# plt.fill_between(timesteps, min_og, max_og, color="tab:blue", alpha=0.2)

for color, model_path in zip(colors, model_paths):
    model = load_speedup_model(model_path, device)
    model.eval()
    with torch.no_grad():
        coeffs_new, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
        pred_new, time_new = rollout_model(model, y0, dt, coeffs_new, fast=True)
        loss_new = torch.nn.functional.mse_loss(pred_new, y_true)
        med_new, min_new, max_new = compute_error_metrics(pred_new, y_true)
        label = os.path.basename(model_path).replace("van_der_pol_speedup_model_", "").replace(".pth", "")
        print(f"[{label}] Time: {time_new:.3f}s | Loss: {loss_new:.3e}")
        plt.plot(timesteps, med_new, color=color, label=label)
        # plt.fill_between(timesteps, min_new, max_new, color=color, alpha=0.2)

plt.grid(True)
plt.yscale("log")
plt.xlabel("Time (s)")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.title("Van der Pol: Rollout Error vs Model Variant")
# plt.savefig("compare_models.png")
plt.show()
