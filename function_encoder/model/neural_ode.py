from typing import Callable, Optional, Tuple, Dict, Union
import torch
from function_encoder.coefficients import least_squares
from function_encoder.inner_products import standard_inner_product


def rk4_step(func, x, dt, **ode_kwargs):
    """Runge-Kutta 4th order ODE integrator for a single step."""
    t = torch.zeros_like(dt, device=dt.device)
    k1 = func(t, x, **ode_kwargs)
    k2 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k1, **ode_kwargs)
    k3 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k2, **ode_kwargs)
    k4 = func(t + dt, x + dt.unsqueeze(-1) * k3, **ode_kwargs)
    return (dt / 6).unsqueeze(-1) * (k1 + 2 * k2 + 2 * k3 + k4)

def rk4_step_fast(func, x, dt, c, **ode_kwargs):
    """Runge-Kutta 4th order ODE integrator for a single step."""
    t = torch.zeros_like(dt, device=dt.device)
    k1 = func(t, x, c, **ode_kwargs)
    k2 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k1, c, **ode_kwargs)
    k3 = func(t + dt / 2, x + (dt / 2).unsqueeze(-1) * k2, c,  **ode_kwargs)
    k4 = func(t + dt, x + dt.unsqueeze(-1) * k3, c, **ode_kwargs)
    return (dt / 6).unsqueeze(-1) * (k1 + 2 * k2 + 2 * k3 + k4)


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
        tx = torch.cat([t.unsqueeze(-1), x], dim=-1)  # Concatenate time and state
        return self.model(tx)


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



class NeuralODEFast(torch.nn.Module):
    """Neural Ordinary Differential Equation model.

    Args:
        ode_func (torch.nn.Module): The vector field
        integrator (Callable): The ODE solver (e.g., `rk4_step`, `odeint`).
    """

    def __init__(
        self,
        ode_func: Callable,
        integrator: Callable,
        coefficients_method: Callable = least_squares,
        inner_product: Callable = standard_inner_product,
    ):
        super(NeuralODEFast, self).__init__()
        self.ode_func = ode_func
        self.integrator = integrator
        self.coefficients_method = coefficients_method
        self.inner_product = inner_product

    def compute_coefficients(
        self, x: torch.Tensor, 
        y: torch.Tensor, 
        ode_kwargs: Optional[Dict] = {},
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the coefficients of the basis functions.

        Args:
            x (torch.Tensor): Input data [batch_size, n_points, n_features]
            y (torch.Tensor): Target data [batch_size, n_points, n_features]

        Returns:
            torch.Tensor: Basis coefficients [batch_size, n_basis]

        """
        f = y
        g = torch.zeros((y.shape[0], y.shape[1], y.shape[2], 10)).to("cuda")
        for ii in range(10):
            c = torch.zeros((1,10)).to("cuda")
            c[:,ii] = 1
            inputs = x + (c,)
            g[:,:,:,ii] = self.integrator(self.ode_func, *inputs, **ode_kwargs)

        coefficients, G = self.coefficients_method(f, g, self.inner_product)

        return coefficients, G

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