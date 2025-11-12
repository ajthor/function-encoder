from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.van_der_pol import VanDerPolDataset, van_der_pol

from function_encoder.model.mlp import MLP, MultiHeadedMLP
from function_encoder.model.neural_ode import NeuralODE, NeuralODEFast, ODEFunc, rk4_step, rk4_step_fast
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder, FunctionEncoderFast
from function_encoder.utils.training import train_step

import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(2)

# Load dataset

n = 100
dataset = VanDerPolDataset(n_points=n, n_example_points=100, dt_range=(0.1, 0.1))
dataloader = DataLoader(dataset, batch_size=100)
dataloader_iter = iter(dataloader)
batch = next(dataloader_iter)

mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
mu = mu.to(device)
y0 = y0.to(device)
dt = dt.to(device)
y1 = y1.to(device)
y0_example = y0_example.to(device)
dt_example = dt_example.to(device)
y1_example = y1_example.to(device)


# Load the original model

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
model_og = FunctionEncoder(basis_functions).to(device)
model_og.load_state_dict(torch.load("van_der_pol_model.pth", map_location=device))

# Load the faster model

n_basis = 10
basis_functions = MultiHeadedMLP(
    layer_sizes=[3, 64, 64, 2], num_heads=n_basis)
model_new = NeuralODEFast(
    ode_func = FunctionEncoderFast(
        basis_functions=ODEFunc(model=basis_functions),
    ),
    integrator=rk4_step_fast
).to(device)
model_new.load_state_dict(torch.load("van_der_pol_speedup_model_n_basis=10_epochs=9000.pth", map_location=device))



# Evaluate the model predictive performance

model_og.eval()
# model_new.eval()

with torch.no_grad():

    # Precompute the coefficients for the batch
    coefficients_og, _ = model_og.compute_coefficients((y0_example, dt_example), y1_example)
    coefficients_new, _ = model_new.compute_coefficients((y0_example, dt_example), y1_example)

    # Integrate the true trajectory
    y = y0.clone()
    x = y[:,0,:].unsqueeze(1)
    for k in range(n-1):
        x = rk4_step(van_der_pol, x, dt[:,k].unsqueeze(1), mu=mu) + x
        y[:,k+1,:] = x.squeeze(1)

    # Integrate the predicted trajectory
    pred_og = y0.clone()
    x = pred_og[:,0,:].unsqueeze(1)
    import time
    start_time = time.time()
    for k in range(n-1):
        x = model_og((x, dt[:,k].unsqueeze(1)), coefficients=coefficients_og) + x
        pred_og[:,k+1,:] = x.squeeze(1)
    print("Time to integrate original model: ", time.time() - start_time)

    # Integrate the predicted trajectory
    pred_new = y0.clone()
    x = pred_new[:,0,:].unsqueeze(1)
    start_time = time.time()
    for k in range(n-1):
        x = model_new((x, dt[:,k].unsqueeze(1), coefficients_new)) + x
        pred_new[:,k+1,:] = x.squeeze(1)
    print("Time to integrate faster model: ", time.time() - start_time)


    # Compare the model-predictions and true outputs 
    loss_og = torch.nn.functional.mse_loss(pred_og, y)
    loss_new = torch.nn.functional.mse_loss(pred_new, y)

    print("Loss of Original Model: ", loss_og)
    print("Loss of New Fast Model: ", loss_new)


    # Compute errors
    # err_og = torch.abs(pred_og - y).detach().cpu().numpy()
    # err_new = torch.abs(pred_new - y).detach().cpu().numpy()
    err_og = torch.mean((pred_og - y) ** 2, dim=2).detach().cpu().numpy()
    err_new = torch.mean((pred_new - y) ** 2, dim=2).detach().cpu().numpy()

    # Compute median and percentiles
    import numpy as np
    med_og = np.median(err_og, axis=0)
    med_new = np.median(err_new, axis=0)

    max_og = np.percentile(err_og, 90, axis=0)
    max_new = np.percentile(err_new, 90, axis=0)

    min_og = np.percentile(err_og, 10, axis=0)
    min_new = np.percentile(err_new, 10, axis=0)


    # Plot per-state rollout error
    import matplotlib.pyplot as plt

    timesteps = np.arange(0, 10, 0.1)

    # for i in range(2):
    plt.plot(timesteps, med_og, color='tab:blue', label='Old')
    plt.fill_between(timesteps,
                            min_og,
                            max_og,
                            color='tab:blue', alpha=0.2)
        

    plt.plot(timesteps, med_new, color='tab:red', label='New')
    plt.fill_between(timesteps,
                            min_new,
                            max_new,
                            color='tab:red', alpha=0.2)
        
    plt.grid(True)
    plt.yscale("log")

    plt.show()



