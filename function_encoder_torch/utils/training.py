import torch

import tqdm


def fit(
    model,
    ds,
    loss_function,
    epochs=100,
    learning_rate=1e-3,
    gradient_accumulation_steps=50,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in trange(epochs):

        for i, batch in enumerate(ds):

            x, y, _ = batch

            optimizer.zero_grad()
            loss = loss_function(model, batch)
            loss.backward()
            if i % gradient_accumulation_steps == 0:
                optimizer.step()
    return model
