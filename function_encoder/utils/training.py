from typing import Callable, Any, Iterator, Union
import torch
from function_encoder.function_encoder import BasisFunctions
import tqdm


def fit(
    model: torch.nn.Module,
    ds: Union[torch.utils.data.Dataset, Iterator],
    loss_function: Callable,
    epochs: int = 1000,
    learning_rate: float = 1e-3,
) -> torch.nn.Module:
    """Train the model using the provided dataset and loss function.

    Args:
        model (torch.nn.Module): Function encoder model
        ds (Union[torch.utils.data.Dataset, Iterator]): Dataset or iterator providing batches
        loss_function (Callable): Loss function to optimize
        epochs (int, optional): Number of epochs to train. Defaults to 1000.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.

    Returns:
        torch.nn.Module: The trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    with tqdm.tqdm(range(epochs)) as tqdm_bar:
        for epoch in tqdm_bar:
            for batch in ds:
                optimizer.zero_grad()
                loss = loss_function(model, batch)
                loss.backward()
                optimizer.step()
                break
            if epoch % 10 == 0:
                tqdm_bar.set_postfix_str(f"loss {loss.item():.2e}")
    return model


def fit_progressive(
    model: torch.nn.Module,
    ds: Union[torch.utils.data.Dataset, Iterator],
    loss_function: Callable,
    n_basis: int,
    basis_function_factory: Callable,
    epochs_per_function: int = 1000,
    learning_rate: float = 1e-3,
    freeze_existing: bool = True,
) -> torch.nn.Module:
    """Train the model progressively by adding basis functions one by one.

    This function trains the model in a progressive manner by adding basis
    functions one at a time and training the model after each addition.

    Args:
        model (torch.nn.Module): Function encoder model
        ds (Union[torch.utils.data.Dataset, Iterator]): Dataset or iterator providing batches
        loss_function (Callable): Loss function to optimize
        n_basis (int): Total number of basis functions
        basis_function_factory (Callable): Factory function to create new basis functions
        epochs_per_function (int, optional): Number of epochs to train each basis function. Defaults to 1000.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        freeze_existing (bool, optional): Whether to freeze existing basis functions during training. Defaults to True.

    Returns:
        torch.nn.Module: The trained model

    Raises:
        ValueError: If model.basis_functions is not an instance of BasisFunctions
    """
    # Ensure the model is using BasisFunctions
    if not isinstance(model.basis_functions, BasisFunctions):
        raise ValueError("model.basis_functions must be an instance of BasisFunctions")

    # Train the first basis function
    model = fit(
        model=model,
        ds=ds,
        loss_function=loss_function,
        epochs=epochs_per_function,
        learning_rate=learning_rate,
    )

    # Train the remaining basis functions progressively
    for _ in tqdm.tqdm(range(n_basis - 1)):
        if freeze_existing is True:
            # Freeze all parameters except the new basis function
            for param in model.parameters():
                param.requires_grad = False

        # Create new basis function
        new_basis_function = basis_function_factory()
        for param in new_basis_function.parameters():
            param.requires_grad = True
        model.basis_functions.basis_functions.append(new_basis_function)

        # Select only the trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
        model.train()

        with tqdm.tqdm(range(epochs_per_function), leave=True) as tqdm_bar:
            for epoch in tqdm_bar:
                for batch in ds:
                    optimizer.zero_grad()
                    loss = loss_function(model, batch)
                    loss.backward()
                    optimizer.step()
                    break
                if epoch % 10 == 0:
                    tqdm_bar.set_postfix_str(f"loss {loss.item():.2e}")

    return model


def train_step(model, optimizer, batch, loss_function):
    """Performs a single training step and returns the loss value."""
    model.train()
    optimizer.zero_grad()
    loss = loss_function(model, batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def test_eval(model, dataloader, loss_function):
    """Evaluates the model on the dataset and returns the average loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            loss = loss_function(model, batch)
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else float("nan")
