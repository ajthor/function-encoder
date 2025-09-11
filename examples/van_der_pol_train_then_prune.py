from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.van_der_pol import VanDerPolDataset, van_der_pol

from function_encoder.model.mlp import MLP
from function_encoder.model.neural_ode import NeuralODE, ODEFunc, rk4_step
from function_encoder.function_encoder import BasisFunctions, FunctionEncoder
from function_encoder.utils.training import train_step

import tqdm
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)

# Load dataset
dataset = VanDerPolDataset(
    integrator=rk4_step, n_points=1000, n_example_points=100, dt_range=(0.1, 0.1)
)
dataloader = DataLoader(dataset, batch_size=50)
dataloader_iter = iter(dataloader)

# Create model
num_basis = 10
basis_functions = BasisFunctions(
    *[
        NeuralODE(
            ode_func=ODEFunc(model=MLP(layer_sizes=[3, 64, 64, 2])),
            integrator=rk4_step,
        )
        for _ in range(num_basis)
    ]
)

model = FunctionEncoder(basis_functions).to(device)

# Training parameters
losses = []


def loss_function(model, batch):
    _, y0, dt, y1, y0_example, dt_example, y1_example = batch
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    pred = model((y0, dt), coefficients=coefficients)

    pred_loss = torch.nn.functional.mse_loss(pred, y1)
    return pred_loss


def compute_explained_variance(model):
    dataloader_coeffs = DataLoader(dataset, batch_size=100)
    dataloader_coeffs_iter = iter(dataloader_coeffs)
    _, y0, dt, y1, y0_example, dt_example, y1_example = next(dataloader_coeffs_iter)
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)
    coefficients, G = model.compute_coefficients((y0_example, dt_example), y1_example)

    coefficients_centered = coefficients - coefficients.mean(dim=0, keepdim=True)
    coefficients_cov = (
        torch.matmul(coefficients_centered.T, coefficients_centered)
        / coefficients.shape[0]
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(coefficients_cov)
    eigenvalues = eigenvalues.flip(0)  # Flip to descending order (largest first)
    eigenvectors = eigenvectors.flip(1)  # Flip columns to match eigenvalue order

    explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)

    return explained_variance_ratio, eigenvalues, eigenvectors, coefficients_centered, G


# # Train all basis functions together
# num_epochs = 5000
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# with tqdm.trange(num_epochs) as tqdm_bar:
#     for epoch in tqdm_bar:
#         batch = next(dataloader_iter)
#         loss = train_step(model, optimizer, batch, loss_function)
#         losses.append(loss)
#         tqdm_bar.set_postfix_str(f"loss: {loss:.2e}")

# print("Training completed!")

# # Save the model

# torch.save(model.state_dict(), "van_der_pol_model.pth")

# Load the model

model.load_state_dict(torch.load("van_der_pol_model.pth"))

# PCA Analysis and Pruning
model.eval()
with torch.no_grad():
    explained_variance_ratio, eigenvalues, eigenvectors, coefficients_centered, G = (
        compute_explained_variance(model)
    )

    # Determine n: number of eigenvectors needed to explain 99% of variance
    cumulative_variance = torch.cumsum(explained_variance_ratio, dim=0)
    n = int((cumulative_variance < 0.99).sum().item() + 1)
    n = min(n, num_basis)  # Can't select more than we have

    print(f"Original model has {num_basis} basis functions")
    print(f"Selecting {n} basis functions to explain 99% of variance")

    # Print all eigenvalues for debugging
    print(f"\nAll eigenvalues (descending order): {eigenvalues.cpu().numpy()}")
    print(f"Explained variance ratios: {explained_variance_ratio.cpu().numpy()}")
    print(f"Cumulative variance: {cumulative_variance.cpu().numpy()}")

    # Get the top n eigenvectors
    top_eigenvectors = eigenvectors[:, :n]  # [num_basis, n]

    # Compute proper cosine similarity using inner products
    dataloader_test = DataLoader(dataset, batch_size=100)
    _, y0, dt, y1, y0_example, dt_example, y1_example = next(iter(dataloader_test))
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    eigenvector_outputs = []
    for i in range(n):
        eigenvector_output = model(
            (y0, dt), coefficients=top_eigenvectors[:, i : i + 1].T
        )
        eigenvector_outputs.append(eigenvector_output)

    eigenvector_outputs = torch.stack(eigenvector_outputs, dim=1)

    # Now get basis function outputs
    basis_outputs = model.basis_functions(
        (y0, dt)
    )  # [batch_size, n_points, n_features, num_basis]

    # Compute cosine similarity (normalized inner products) between each basis function and each eigenvector function
    similarity_scores = torch.zeros(n, num_basis, device=device)

    for i in range(n):
        # eigenvector_outputs[:, i]: [batch_size, n_points, n_features]
        eigenvec_output = eigenvector_outputs[:, i].unsqueeze(
            -1
        )  # [batch_size, n_points, n_features, 1]

        # Compute inner product with all basis functions
        inner_prod_matrix = model.inner_product(
            basis_outputs, eigenvec_output
        )  # [batch_size, num_basis, 1]

        # Compute norms of basis functions and eigenvector function
        # For basis functions: compute self inner product for each
        basis_norms_squared = torch.zeros(100, num_basis, device=device)
        for j in range(num_basis):
            basis_j = basis_outputs[
                :, :, :, j : j + 1
            ]  # [batch_size, n_points, n_features, 1]
            basis_norm_sq = model.inner_product(basis_j, basis_j)  # [batch_size, 1, 1]
            basis_norms_squared[:, j] = basis_norm_sq.squeeze(-1).squeeze(
                -1
            )  # [batch_size]

        basis_norms = torch.sqrt(basis_norms_squared)  # [batch_size, num_basis]

        # For eigenvector function: compute self inner product
        eigenvec_norm_sq = model.inner_product(
            eigenvec_output, eigenvec_output
        )  # [batch_size, 1, 1]
        eigenvec_norm = torch.sqrt(
            eigenvec_norm_sq.squeeze(-1).squeeze(-1)
        )  # [batch_size]

        # Compute cosine similarity: inner_product / (norm1 * norm2)
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        cosine_sim = inner_prod_matrix.squeeze(-1) / (
            basis_norms * eigenvec_norm.unsqueeze(-1) + eps
        )  # [batch_size, num_basis]

        # Average over batch dimension
        similarity_scores[i, :] = cosine_sim.mean(dim=0)  # [num_basis]

    print(f"\nCosine similarity matrix:")
    print(f"Shape: {similarity_scores.shape} (eigenvectors x basis functions)")
    for i in range(n):
        print(
            f"Eigenvector {i+1} cosine similarities: {similarity_scores[i, :].cpu().numpy()}"
        )

    # Select basis functions based on cosine similarity to eigenvector functions
    print(f"\nTop {n} eigenvalues: {eigenvalues[:n].cpu().numpy()}")

    # Method 1: Select based on highest cosine similarity to any eigenvector function
    max_similarity_per_basis = torch.max(torch.abs(similarity_scores), dim=0)[
        0
    ]  # [num_basis]
    print(
        f"\nMax absolute cosine similarity per basis function: {max_similarity_per_basis.cpu().numpy()}"
    )

    # Method 2: Select based on weighted sum of cosine similarities
    eigenvalue_weights = (
        eigenvalues[:n] / eigenvalues[:n].sum()
    )  # Normalize eigenvalues
    weighted_similarity = torch.zeros(num_basis, device=device)
    for i in range(n):
        weighted_similarity += (
            torch.abs(similarity_scores[i, :]) * eigenvalue_weights[i]
        )

    print(f"Weighted cosine similarity scores: {weighted_similarity.cpu().numpy()}")

    # Try both selection methods
    print(f"\n=== METHOD 1: Select by Maximum Cosine Similarity ===")
    _, selected_indices_method1 = torch.topk(
        max_similarity_per_basis, k=n, largest=True
    )
    selected_indices_method1 = selected_indices_method1.sort().values
    print(f"Selected basis functions: {selected_indices_method1.tolist()}")
    for idx in selected_indices_method1.tolist():
        max_sim = max_similarity_per_basis[idx].item()
        best_eigenvec = torch.argmax(torch.abs(similarity_scores[:, idx])).item()
        print(
            f"  Basis {idx+1}: max_similarity={max_sim:.6f} (with eigenvector {best_eigenvec+1})"
        )

    print(f"\n=== METHOD 2: Select by Weighted Cosine Similarity ===")
    _, selected_indices_method2 = torch.topk(weighted_similarity, k=n, largest=True)
    selected_indices_method2 = selected_indices_method2.sort().values
    print(f"Selected basis functions: {selected_indices_method2.tolist()}")
    for idx in selected_indices_method2.tolist():
        weighted_sim = weighted_similarity[idx].item()
        print(f"  Basis {idx+1}: weighted_similarity={weighted_sim:.6f}")

    # Method 3: Reconstruction Error Minimization
    print(f"\n=== METHOD 3: Reconstruction Error Minimization ===")

    def compute_reconstruction_error(subset_indices, eigenvector_funcs, test_inputs):
        """Compute reconstruction error for a given subset of basis functions"""
        if len(subset_indices) == 0:
            return float("inf")

        # Create a temporary model with only the selected basis functions
        temp_basis_functions = BasisFunctions()
        for idx in subset_indices:
            temp_basis_functions.basis_functions.append(
                model.basis_functions.basis_functions[idx]
            )
        temp_model = FunctionEncoder(temp_basis_functions).to(device)

        total_error = 0.0
        total_norm = 0.0

        for i in range(len(eigenvector_funcs)):
            # Target eigenvector function: eigenvector_funcs[i] [batch_size, n_points, n_features]
            target_func = eigenvector_funcs[i]

            # Fit coefficients for this target using the subset of basis functions
            coefficients, _ = temp_model.compute_coefficients(test_inputs, target_func)

            # Reconstruct using fitted coefficients
            reconstructed = temp_model(test_inputs, coefficients=coefficients)

            # Compute L2 error
            error = torch.nn.functional.mse_loss(reconstructed, target_func)
            norm = torch.nn.functional.mse_loss(
                target_func, torch.zeros_like(target_func)
            )

            # Weight by eigenvalue importance
            total_error += error * eigenvalue_weights[i]
            total_norm += norm * eigenvalue_weights[i]

        # Return relative error
        return (total_error / (total_norm + 1e-8)).item()

    # For computational efficiency, use greedy forward selection for Method 3
    # Start with empty set, iteratively add the basis function that most reduces reconstruction error

    print("Running greedy forward selection based on reconstruction error...")
    selected_indices_method3 = []
    remaining_indices = list(range(num_basis))

    # Convert eigenvector_outputs to list format for reconstruction error computation
    eigenvector_funcs_list = [eigenvector_outputs[:, i] for i in range(n)]
    test_inputs = (y0, dt)

    for step in range(n):
        best_error = float("inf")
        best_idx = None

        print(f"  Step {step+1}/{n}: Evaluating {len(remaining_indices)} candidates...")

        for candidate_idx in remaining_indices:
            candidate_subset = selected_indices_method3 + [candidate_idx]
            error = compute_reconstruction_error(
                candidate_subset, eigenvector_funcs_list, test_inputs
            )

            if error < best_error:
                best_error = error
                best_idx = candidate_idx

        selected_indices_method3.append(best_idx)
        remaining_indices.remove(best_idx)

        print(f"    Selected basis {best_idx+1} (error: {best_error:.6f})")

    selected_indices_method3 = torch.tensor(
        sorted(selected_indices_method3), device=device
    )

    print(f"\nMethod 3 results:")
    print(f"Selected basis functions: {selected_indices_method3.tolist()}")

    # Compute final reconstruction error for comparison
    final_error = compute_reconstruction_error(
        selected_indices_method3.tolist(), eigenvector_funcs_list, test_inputs
    )
    print(f"Final reconstruction error: {final_error:.6f}")

    # Compare all three methods
    print(f"\n=== COMPARISON OF ALL METHODS ===")
    print(f"Method 1 (Max Cosine Sim):      {selected_indices_method1.tolist()}")
    print(f"Method 2 (Weighted Cosine Sim): {selected_indices_method2.tolist()}")
    print(f"Method 3 (Reconstruction Error): {selected_indices_method3.tolist()}")

    # Compute reconstruction errors for all methods for comparison
    error1 = compute_reconstruction_error(
        selected_indices_method1.tolist(), eigenvector_funcs_list, test_inputs
    )
    error2 = compute_reconstruction_error(
        selected_indices_method2.tolist(), eigenvector_funcs_list, test_inputs
    )
    error3 = compute_reconstruction_error(
        selected_indices_method3.tolist(), eigenvector_funcs_list, test_inputs
    )

    print(f"\nReconstruction errors:")
    print(f"Method 1: {error1:.6f}")
    print(f"Method 2: {error2:.6f}")
    print(f"Method 3: {error3:.6f}")

    # Method 4: True Reconstruction Error Minimization (exhaustive search)
    from itertools import combinations

    print(f"\n=== METHOD 4: True Reconstruction Error Minimization (Exhaustive) ===")

    total_combinations = len(list(combinations(range(num_basis), n)))
    print(
        f"Evaluating all {total_combinations} combinations of {n} basis functions from {num_basis} total..."
    )

    if total_combinations > 1000:
        print(
            f"Warning: {total_combinations} combinations is quite large. This may take a while..."
        )

    best_error_method4 = float("inf")
    best_subset_method4 = None

    for i, subset in enumerate(combinations(range(num_basis), n)):
        error = compute_reconstruction_error(
            list(subset), eigenvector_funcs_list, test_inputs
        )

        if error < best_error_method4:
            best_error_method4 = error
            best_subset_method4 = subset

        # Progress reporting for large searches
        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"  Evaluated {i + 1}/{total_combinations} combinations (best so far: {best_error_method4:.6f})"
            )

    selected_indices_method4 = torch.tensor(sorted(best_subset_method4), device=device)

    print(f"\nMethod 4 results:")
    print(f"Selected basis functions: {selected_indices_method4.tolist()}")
    print(f"Best reconstruction error: {best_error_method4:.6f}")
    print(f"Total combinations evaluated: {total_combinations}")

    # Compare all four methods
    print(f"\n=== COMPARISON OF ALL METHODS ===")
    print(f"Method 1 (Max Cosine Sim):         {selected_indices_method1.tolist()}")
    print(f"Method 2 (Weighted Cosine Sim):    {selected_indices_method2.tolist()}")
    print(f"Method 3 (Greedy Forward):         {selected_indices_method3.tolist()}")
    print(f"Method 4 (Exhaustive Search):      {selected_indices_method4.tolist()}")

    # Compute reconstruction errors for all methods for comparison
    error4 = best_error_method4  # Already computed

    print(f"\nReconstruction errors:")
    print(f"Method 1 (Max Cosine):        {error1:.6f}")
    print(f"Method 2 (Weighted Cosine):   {error2:.6f}")
    print(f"Method 3 (Greedy Forward):    {error3:.6f}")
    print(f"Method 4 (Exhaustive):        {error4:.6f}")

    # Use the method with lowest reconstruction error
    errors = [error1, error2, error3, error4]
    methods = [
        selected_indices_method1,
        selected_indices_method2,
        selected_indices_method3,
        selected_indices_method4,
    ]
    method_names = [
        "Max Cosine Similarity",
        "Weighted Cosine Similarity",
        "Greedy Forward Selection",
        "Exhaustive Search",
    ]

    best_method_idx = torch.argmin(torch.tensor(errors)).item()

    selected_indices = methods[best_method_idx]

    # selected_indices = selected_indices_method1

    print(
        f"\nUsing Method {best_method_idx + 1} ({method_names[best_method_idx]}) - lowest reconstruction error"
    )
    print(f"Selected basis function indices: {selected_indices.tolist()}")
    print(f"Final reconstruction error: {errors[best_method_idx]:.6f}")

    selected_indices = selected_indices_method1

# Create pruned model
pruned_basis_functions = BasisFunctions()
for idx in selected_indices:
    # Copy the selected basis function
    original_basis = model.basis_functions.basis_functions[idx]
    pruned_basis_functions.basis_functions.append(original_basis)

pruned_model = FunctionEncoder(pruned_basis_functions).to(device)

print(
    f"Created pruned model with {len(pruned_model.basis_functions.basis_functions)} basis functions"
)

# Evaluation comparison
model.eval()
pruned_model.eval()

with torch.no_grad():
    # Test on a batch
    dataloader_eval = DataLoader(dataset, batch_size=9)
    batch = next(iter(dataloader_eval))

    mu, y0, dt, y1, y0_example, dt_example, y1_example = batch
    mu = mu.to(device)
    y0 = y0.to(device)
    dt = dt.to(device)
    y1 = y1.to(device)
    y0_example = y0_example.to(device)
    dt_example = dt_example.to(device)
    y1_example = y1_example.to(device)

    # Original model predictions
    coefficients_orig, _ = model.compute_coefficients(
        (y0_example, dt_example), y1_example
    )
    pred_orig = model((y0, dt), coefficients=coefficients_orig)

    # Pruned model predictions
    coefficients_pruned, _ = pruned_model.compute_coefficients(
        (y0_example, dt_example), y1_example
    )
    pred_pruned = pruned_model((y0, dt), coefficients=coefficients_pruned)

    # Compute losses
    loss_orig = torch.nn.functional.mse_loss(pred_orig, y1).item()
    loss_pruned = torch.nn.functional.mse_loss(pred_pruned, y1).item()

    print(f"Original model MSE: {loss_orig:.6f}")
    print(f"Pruned model MSE: {loss_pruned:.6f}")
    print(f"Performance ratio: {loss_pruned / loss_orig:.3f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot trajectory comparisons (top row)
for i in range(3):
    ax = axes[0, i]

    # Get a single trajectory for plotting
    _mu = mu[i]
    _y0 = torch.empty(1, 2, device=device).uniform_(*dataset.y0_range)
    _c_orig = coefficients_orig[i].unsqueeze(0)
    _c_pruned = coefficients_pruned[i].unsqueeze(0)

    s = 0.1  # Time step for simulation
    n_steps = int(10 / s)
    _dt = torch.tensor([s], device=device)

    # Integrate true trajectory
    x = _y0.clone()
    y_true = [x]
    for k in range(n_steps):
        x = rk4_step(van_der_pol, x, _dt, mu=_mu) + x
        y_true.append(x)
    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()

    # Integrate original model trajectory
    x = _y0.clone().unsqueeze(1)
    _dt_expand = _dt.unsqueeze(0)
    pred_orig = [x]
    for k in range(n_steps):
        x = model((x, _dt_expand), coefficients=_c_orig) + x
        pred_orig.append(x)
    pred_orig = torch.cat(pred_orig, dim=1)[0].detach().cpu().numpy()

    # Integrate pruned model trajectory
    x = _y0.clone().unsqueeze(1)
    pred_pruned = [x]
    for k in range(n_steps):
        x = pruned_model((x, _dt_expand), coefficients=_c_pruned) + x
        pred_pruned.append(x)
    pred_pruned = torch.cat(pred_pruned, dim=1)[0].detach().cpu().numpy()

    ax.plot(y_true[:, 0], y_true[:, 1], "k-", linewidth=2, label="True", alpha=0.8)
    ax.plot(
        pred_orig[:, 0],
        pred_orig[:, 1],
        "b--",
        linewidth=2,
        label=f"Original ({num_basis} basis)",
        alpha=0.7,
    )
    ax.plot(
        pred_pruned[:, 0],
        pred_pruned[:, 1],
        "r:",
        linewidth=2,
        label=f"Pruned ({n} basis)",
        alpha=0.7,
    )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(f"Trajectory {i+1}")
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend()

# Plot training loss (bottom left)
axes[1, 0].plot(losses)
axes[1, 0].set_xlabel("Training Step")
axes[1, 0].set_ylabel("MSE Loss")
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("Training Loss")
axes[1, 0].grid(True, alpha=0.3)

# Plot explained variance (bottom middle)
explained_variance_np = explained_variance_ratio.cpu().numpy()
cumulative_variance_np = cumulative_variance.cpu().numpy()

axes[1, 1].bar(
    range(1, len(explained_variance_np) + 1),
    explained_variance_np,
    alpha=0.6,
    color="skyblue",
    label="Individual",
)
axes[1, 1].plot(
    range(1, len(cumulative_variance_np) + 1),
    cumulative_variance_np,
    "ro-",
    alpha=0.8,
    label="Cumulative",
)
axes[1, 1].axhline(y=0.99, color="g", linestyle="--", alpha=0.8, label="99% threshold")
axes[1, 1].set_xlabel("Eigenvalue Index")
axes[1, 1].set_ylabel("Explained Variance Ratio")
axes[1, 1].set_title("Explained Variance Analysis")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot basis function selection (bottom right)
eigenvalues_np = eigenvalues.cpu().numpy()

axes[1, 2].bar(
    range(1, len(eigenvalues_np) + 1), eigenvalues_np, alpha=0.6, label="Eigenvalues"
)
axes[1, 2].axhline(
    y=eigenvalues_np[n - 1] if n > 0 else 0,
    color="r",
    linestyle="--",
    alpha=0.8,
    label=f"Selection cutoff (n={n})",
)

# Mark selected basis functions
selected_indices_np = selected_indices.cpu().numpy()
for idx in selected_indices_np:
    axes[1, 2].axvline(x=idx + 1, color="g", linestyle=":", alpha=0.5)

axes[1, 2].set_xlabel("Basis Function Index")
axes[1, 2].set_ylabel("Eigenvalue")
axes[1, 2].set_yscale("log")
axes[1, 2].set_title(f"Basis Selection (Selected: {list(selected_indices_np + 1)})")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n" + "=" * 60)
print("TRAIN-THEN-PRUNE SUMMARY")
print("=" * 60)
print(f"Original model: {num_basis} basis functions")
print(f"Pruned model: {n} basis functions ({100*n/num_basis:.1f}% of original)")
print(f"Variance explained by {n} components: {cumulative_variance[n-1]:.4f}")
print(f"Selected basis function indices: {selected_indices.tolist()}")
print(f"Original model MSE: {loss_orig:.6f}")
print(f"Pruned model MSE: {loss_pruned:.6f}")
print(f"Performance degradation: {100*(loss_pruned - loss_orig)/loss_orig:+.2f}%")
print(f"Model compression ratio: {num_basis/n:.1f}x")
print("=" * 60)
