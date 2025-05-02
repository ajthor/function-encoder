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
        integrator (callable): ODE solver
    """

    def __init__(self, ode_func, integrator):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
        self.integrator = integrator

    def forward(self, inputs, ode_kwargs={}):
        """Solve the initial value problem.

        The output of the integrator has to be [batch_size, n_points, n_features].
        This means the inputs (y0, t) have to be [batch_size, 1, n_features] and [batch_size, n_points].

        Args:
            inputs (tuple): A tuple containing (y0, t), where:
                y0 (torch.Tensor): Initial condition
                t (torch.Tensor): Time points at which to return the solution
            ode_kwargs (dict, optional): Additional integrator arguments. Defaults to {}.

        Returns:
            torch.Tensor: Solution of the ODE at the requested time points
        """
        y0, t = inputs
        return self.integrator(self.ode_func, y0, t, **ode_kwargs)
