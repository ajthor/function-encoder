from typing import Callable, Optional, Tuple, Dict
import torch


class ODEFunc(torch.nn.Module):
    """A wrapper for a PyTorch model to make it compatible with ODE solvers.

    Args:
        model (torch.nn.Module): The neural network model.
    """

    def __init__(self, model: torch.nn.Module):
        super(ODEFunc, self).__init__()
        self.model = model

    def forward(self, t, x):
        """Compute the time derivative at the current state.

        Args:
            t (torch.Tensor): Current time
            x (torch.Tensor): Current state

        Returns:
            torch.Tensor: The time derivative dx/dt at the current state
        """
        return self.model(x)


class NeuralODE(torch.nn.Module):
    """Neural Ordinary Differential Equation model.

    Args:
        ode_func (torch.nn.Module): The vector field
        integrator (Callable): The ODE solver (e.g., `rk4_step`, `odeint`).
    """

    def __init__(
        self,
        ode_func: Callable,
        integrator: Callable,
    ):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
        self.integrator = integrator

    def forward(
        self,
        inputs,
        ode_kwargs: Optional[Dict] = {},
    ):
        """Solve the initial value problem.

        Args:
            inputs (tuple): A tuple containing (y0, t), where:
                y0 (torch.Tensor): Initial condition
                dt (torch.Tensor): Time step
            ode_kwargs (dict, optional): Additional integrator arguments. Defaults to {}.

        Returns:
            torch.Tensor: Solution of the ODE at the next time step.
        """
        return self.integrator(self.ode_func, *inputs, **ode_kwargs)
