from typing import Callable

import torch

from function_encoder.function_encoder import BasisFunctions

import tqdm


def fit(
    model,
    ds,
    loss_function: Callable,
    epochs: int = 1000,
    learning_rate: float = 1e-3,
):
    """Train the model using the provided dataset and loss function.

    Args:
        model: function encoder model
        ds: dataset
        loss_function: loss function
        epochs: number of epochs to train
        learning_rate: learning rate for the optimizer

    Returns:
        The trained model
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
    model,
    ds,
    loss_function: Callable,
    n_basis: int,
    basis_function_factory: Callable,
    epochs_per_function: int = 1000,
    learning_rate: float = 1e-3,
    freeze_existing: bool = True,
):
    """Train the model progressively by adding basis functions one by one.

    Args:
        model: function encoder model
        ds: dataset
        loss_function: loss function
        n_basis: total number of basis functions
        basis_function_factory: factory function to create new basis functions
        epochs_per_function: number of epochs to train each basis function
        learning_rate: learning rate for the optimizer
        freeze_existing: whether to freeze existing basis functions during training

    Returns:
        The trained model
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
