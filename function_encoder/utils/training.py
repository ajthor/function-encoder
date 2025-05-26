from typing import Callable, Any, Iterator, Union
import torch
from function_encoder.function_encoder import BasisFunctions
import tqdm


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
