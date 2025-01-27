from typing import Callable

import torch

import tqdm


def fit(
    model,
    ds,
    loss_function: Callable,
    epochs=1000,
    learning_rate=1e-3,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    with tqdm.tqdm(range(epochs)) as tqdm_bar:
        for i, epoch in enumerate(tqdm_bar):

            for batch in ds:

                optimizer.zero_grad()

                loss = loss_function(model, batch)
                loss.backward()

                optimizer.step()

                break

            if i % 10 == 0:
                tqdm_bar.set_postfix_str(f"Loss {loss.item()}")

    return model
