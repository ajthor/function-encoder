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


def basis_function_factory():
    return NeuralODE(
        ode_func=ODEFunc(model=MLP(layer_sizes=[5, 64, 64, 4])),
        integrator=rk4_step,
    )


num_basis = 10
# Start with one basis function for progressive training
basis_functions = BasisFunctions(basis_function_factory())

model = FunctionEncoder(basis_functions).to(device)

# Train model

losses = []  # For plotting
scores = []  # For plotting
dataloader_coeffs = DataLoader(dataset, batch_size=100)
dataloader_coeffs_iter = iter(dataloader_coeffs)


def compute_explained_variance(model):
    _, _, _, _, example_y0, example_dt, example_y1 = next(dataloader_coeffs_iter)
    # Data is already on the correct device from the dataset

    coefficients, G = model.compute_coefficients((example_y0, example_dt), example_y1)

    # Compute PCA on coefficients (following van der Pol example)
    coefficients_centered = coefficients - coefficients.mean(dim=0, keepdim=True)
    coefficients_cov = (
        torch.matmul(coefficients_centered.T, coefficients_centered)
        / coefficients.shape[0]
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(coefficients_cov)
    eigenvalues = eigenvalues.flip(0)  # Flip to descending order

    # Also compute Gram matrix eigenvalues (alternative approach from van der Pol)
    K = G.mean(dim=0)
    gram_eigenvalues, gram_eigenvectors = torch.linalg.eigh(K)
    gram_eigenvalues = gram_eigenvalues.flip(0)  # Flip to descending order

    explained_variance_ratio = gram_eigenvalues / torch.sum(gram_eigenvalues)

    return explained_variance_ratio, eigenvalues, gram_eigenvalues


def loss_function(model, batch):
    _, y0, dt, y1, y0_example, dt_example, y1_example = batch
    # Data is already on the correct device from the dataset

    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    pred = model((y0, dt), coefficients=coefficients)

    pred_loss = torch.nn.functional.mse_loss(pred, y1)

    return pred_loss


# Train the first basis function
num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
with tqdm.tqdm(range(num_epochs), desc=f"basis 1/{num_basis}") as tqdm_bar:
    for epoch in tqdm_bar:
        batch = next(dataloader_iter)
        loss = train_step(model, optimizer, batch, loss_function)
        losses.append(loss)
        tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})

model.eval()
with torch.no_grad():
    explained_variance_ratio, _, _ = compute_explained_variance(model)
    scores.append(explained_variance_ratio)

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
            losses.append(loss)
            tqdm_bar.set_postfix({"loss": f"{loss:.2e}"})

    model.eval()
    with torch.no_grad():
        explained_variance_ratio, _, _ = compute_explained_variance(model)
        scores.append(explained_variance_ratio)

# Plot results (following van der Pol PCA structure more closely)

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    # Test on a single trajectory (like van der Pol)
    dataloader_eval = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader_eval))

    M_central, y0, dt, y1, y0_example, dt_example, y1_example = batch

    # Create time series data for plotting (like van der Pol X,y)
    from datasets.kepler import generate_kepler_states_batch
    _y0_single = generate_kepler_states_batch(
        M_central.item(), dataset.a_range, dataset.e_range, 1, device=torch.device(device)
    )
    
    # Generate time series for orbital trajectory
    n_points = 200
    t_span = 3.0  # Total time
    dt_plot = t_span / n_points
    t_values = torch.linspace(0, t_span, n_points, device=device)
    
    # Simulate true trajectory
    states_true = []
    x = _y0_single.clone()
    states_true.append(x)
    dt_step = torch.tensor([dt_plot], device=device)
    
    for i in range(n_points - 1):
        x = rk4_step(kepler, x, dt_step, M_central=M_central) + x
        states_true.append(x)
    states_true = torch.cat(states_true, dim=0)
    
    # Generate predicted trajectory
    coefficients, _ = model.compute_coefficients((y0_example, dt_example), y1_example)
    
    states_pred = []
    x = _y0_single.clone()
    x = x.unsqueeze(1)
    dt_step = dt_step.unsqueeze(0)
    states_pred.append(x)
    
    for i in range(n_points - 1):
        x = model((x, dt_step), coefficients=coefficients) + x
        states_pred.append(x)
    states_pred = torch.cat(states_pred, dim=1)

    # Convert to numpy
    states_true = states_true.detach().cpu().numpy()
    states_pred = states_pred.detach().cpu().numpy()
    t_values = t_values.cpu().numpy()
    y0_example = y0_example.detach().cpu().numpy()
    y1_example = y1_example.detach().cpu().numpy()

    # Plot the results (exactly like van der Pol)
    fig, ax = plt.subplots()
    ax.plot(states_true[:, 0], states_true[:, 1], label="True")
    ax.plot(states_pred[0, :, 0], states_pred[0, :, 1], label="Predicted") 
    ax.scatter(y0_example[:, 0], y0_example[:, 1], label="Data", color="red")
    ax.plot(0, 0, 'ko', markersize=6, label="Central Body")
    ax.set_aspect('equal')
    ax.legend()
    plt.show()

    # Visualize individual basis functions (exactly like van der Pol)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    # Create test states for basis function visualization
    test_states = torch.zeros(100, 1, 4, device=device)
    # Sample states along a circle for visualization
    angles = torch.linspace(0, 2*torch.pi, 100, device=device)
    radius = 2.0
    test_states[:, 0, 0] = radius * torch.cos(angles)  # x
    test_states[:, 0, 1] = radius * torch.sin(angles)  # y
    test_states[:, 0, 2] = -torch.sin(angles)  # vx (tangential)
    test_states[:, 0, 3] = torch.cos(angles)   # vy (tangential)
    
    dt_test = torch.ones(100, 1, device=device) * 0.1
    
    for i, basis_fn in enumerate(model.basis_functions.basis_functions):
        if i >= num_basis or i >= len(axes):
            break
        basis_output = basis_fn((test_states, dt_test))
        # Plot x-component of basis function output
        axes[i].plot(angles.cpu().numpy(), basis_output[:, 0, 0].detach().cpu().numpy())
        axes[i].set_title(f"Basis Function {i+1}")
        axes[i].set_xlabel("Angle")
        axes[i].set_ylabel("Output")
    plt.tight_layout()
    plt.show()

    # Plot loss and explained variance (exactly like van der Pol)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot loss
    ax1.plot(losses)
    ax1.set_ylabel("MSE")
    ax1.grid(True)
    ax1.set_yscale("log")

    # Plot explained variance ratio
    for i in range(len(scores)):
        scores_np = scores[i].cpu().numpy()
        ax2.plot(
            range(1, len(scores_np) + 1),
            scores_np,
            marker="o",
            label=f"k = {i + 1}",
        )
    ax2.set_xlabel("Eigenvalue Index")
    ax2.set_ylabel("Explained Variance Ratio")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True)

    # Plot the eigenvalues of the coefficients
    _, eigenvalues, gram_eigenvalues = compute_explained_variance(model)
    eigenvalues = eigenvalues.cpu().numpy()
    gram_eigenvalues = gram_eigenvalues.cpu().numpy()

    ax3.plot(
        range(1, len(eigenvalues) + 1),
        eigenvalues,
        marker="o",
        label="Covariance Matrix",
    )
    ax3.plot(
        range(1, len(gram_eigenvalues) + 1),
        gram_eigenvalues,
        marker="s",
        label="Gram Matrix",
    )
    ax3.set_xlabel("Eigenvalue Index")
    ax3.set_ylabel("Eigenvalue")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), "kepler_pca_model.pth")

print(
    f"Training completed with {len(model.basis_functions.basis_functions)} basis functions"
)
print(
    f"Final explained variance ratios: {scores[-1][:5].cpu().numpy()}"
)  # Show first 5
