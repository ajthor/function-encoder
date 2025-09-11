from typing import Callable
import math

import torch
from torch.utils.data import IterableDataset


def orbital_to_cartesian_kepler(a, e, i, omega, Omega, nu, mu):
    """Convert orbital elements to Cartesian coordinates for Kepler problem.

    Args:
        a: Semi-major axis
        e: Eccentricity
        i: Inclination (rad)
        omega: Argument of periapsis (rad)
        Omega: Longitude of ascending node (rad)
        nu: True anomaly (rad)
        mu: Gravitational parameter (G * M_central)

    Returns:
        r, v: Position and velocity vectors [3D] of test particle
    """
    # Distance from central body
    r_mag = a * (1 - e**2) / (1 + e * torch.cos(nu))

    # Position in orbital frame
    r_orb = torch.stack(
        [r_mag * torch.cos(nu), r_mag * torch.sin(nu), torch.zeros_like(r_mag)], dim=-1
    )

    # Velocity in orbital frame
    h = torch.sqrt(mu * a * (1 - e**2))  # Angular momentum magnitude
    v_orb = torch.stack(
        [
            -(mu / h) * torch.sin(nu),
            (mu / h) * (e + torch.cos(nu)),
            torch.zeros_like(r_mag),
        ],
        dim=-1,
    )

    # Rotation matrices for 3D orientation
    cos_omega = torch.cos(omega)
    sin_omega = torch.sin(omega)
    cos_Omega = torch.cos(Omega)
    sin_Omega = torch.sin(Omega)
    cos_i = torch.cos(i)
    sin_i = torch.sin(i)

    # Transformation matrix from orbital to inertial frame
    R11 = cos_omega * cos_Omega - sin_omega * cos_i * sin_Omega
    R12 = -sin_omega * cos_Omega - cos_omega * cos_i * sin_Omega
    R13 = sin_i * sin_Omega

    R21 = cos_omega * sin_Omega + sin_omega * cos_i * cos_Omega
    R22 = -sin_omega * sin_Omega + cos_omega * cos_i * cos_Omega
    R23 = -sin_i * cos_Omega

    R31 = sin_omega * sin_i
    R32 = cos_omega * sin_i
    R33 = cos_i

    # Transform to inertial frame
    r = torch.stack(
        [
            R11 * r_orb[..., 0] + R12 * r_orb[..., 1] + R13 * r_orb[..., 2],
            R21 * r_orb[..., 0] + R22 * r_orb[..., 1] + R23 * r_orb[..., 2],
            R31 * r_orb[..., 0] + R32 * r_orb[..., 1] + R33 * r_orb[..., 2],
        ],
        dim=-1,
    )

    v = torch.stack(
        [
            R11 * v_orb[..., 0] + R12 * v_orb[..., 1] + R13 * v_orb[..., 2],
            R21 * v_orb[..., 0] + R22 * v_orb[..., 1] + R23 * v_orb[..., 2],
            R31 * v_orb[..., 0] + R32 * v_orb[..., 1] + R33 * v_orb[..., 2],
        ],
        dim=-1,
    )

    return r, v


def generate_kepler_states_batch(M_central, a_range, e_range, n_samples, device=None):
    """Generate batch of initial conditions for Kepler problem using orbital parameters.

    Args:
        M_central: Mass of central body (scalar)
        a_range: Range for semi-major axis
        e_range: Range for eccentricity
        n_samples: Number of states to generate
        device: Device to create tensors on

    Returns:
        Initial state vectors [n_samples, 4] [x, y, vx, vy] of test particle
    """
    if device is None:
        device = torch.device("cpu")

    G = 1.0  # Gravitational constant
    mu = G * M_central  # Gravitational parameter

    # Generate orbital elements in batch
    a = torch.empty(n_samples, device=device).uniform_(*a_range)
    e = torch.empty(n_samples, device=device).uniform_(*e_range)

    # Random orientation (simplified to 2D for visualization)
    i = torch.zeros(n_samples, device=device)  # Keep in x-y plane
    omega = torch.empty(n_samples, device=device).uniform_(0, 2 * math.pi)
    Omega = torch.zeros(n_samples, device=device)  # No need for ascending node in 2D
    nu = torch.empty(n_samples, device=device).uniform_(0, 2 * math.pi)

    # Get position and velocity vectors
    r, v = orbital_to_cartesian_kepler(a, e, i, omega, Omega, nu, mu)

    # Extract 2D components (x, y, vx, vy)
    state = torch.cat([r[..., :2], v[..., :2]], dim=-1)

    return state


def generate_kepler_state(M_central, a_range, e_range):
    """Generate single initial condition (backward compatibility)."""
    return generate_kepler_states_batch(M_central, a_range, e_range, 1)[0:1]


def kepler(t, x, M_central=1.0):
    """Kepler problem dynamics - test particle orbiting massive central body.

    Args:
        t: Time (unused but required for ODE interface)
        x: State vector [x, y, vx, vy] of test particle
        M_central: Mass of central body (scalar)

    Returns:
        State derivatives [vx, vy, ax, ay]
    """
    G = 1.0  # Gravitational constant (normalized units)

    # Extract position and velocity of test particle
    px, py = x[..., 0], x[..., 1]
    vx, vy = x[..., 2], x[..., 3]

    # Distance from origin (where central body is located)
    r = torch.sqrt(px**2 + py**2 + 1e-8)  # Small epsilon to avoid singularities

    # Ensure M_central is a scalar for proper broadcasting
    if isinstance(M_central, torch.Tensor):
        M_central = M_central.item()

    # Gravitational acceleration towards origin
    acc_mag = G * M_central / (r**2)

    # Acceleration components (pointing toward origin)
    ax = -acc_mag * (px / r)
    ay = -acc_mag * (py / r)

    # Return state derivatives
    return torch.stack([vx, vy, ax, ay], dim=-1)


class KeplerDataset(IterableDataset):
    def __init__(
        self,
        integrator: Callable,
        n_points: int = 1000,
        n_example_points: int = 100,
        M_central_range=(0.8, 1.2),  # Central mass range
        a_range=(1.0, 3.0),  # Semi-major axis range
        e_range=(0.0, 0.7),  # Eccentricity range (avoid parabolic/hyperbolic)
        dt_range=(0.1, 0.1),
        device=None,
    ):
        super().__init__()
        self.integrator = integrator
        self.n_points = n_points
        self.n_example_points = n_example_points
        self.M_central_range = M_central_range
        self.a_range = a_range
        self.e_range = e_range
        self.dt_range = dt_range
        self.device = device or torch.device("cpu")

    def __iter__(self):
        while True:
            total_points = self.n_example_points + self.n_points

            # Generate randomized central mass
            M_central = torch.empty(1, device=self.device).uniform_(
                *self.M_central_range
            )

            # Generate initial conditions using orbital parameters (batch mode)
            _y0 = generate_kepler_states_batch(
                M_central.item(),
                self.a_range,
                self.e_range,
                total_points,
                device=self.device,
            )

            # Generate random time steps
            _dt = torch.empty(total_points, device=self.device).uniform_(*self.dt_range)

            # Integrate one step
            _y1 = self.integrator(kepler, _y0, _dt, M_central=M_central)

            # Split the data
            y0_example = _y0[: self.n_example_points]
            dt_example = _dt[: self.n_example_points]
            y1_example = _y1[: self.n_example_points]

            y0 = _y0[self.n_example_points :]
            dt = _dt[self.n_example_points :]
            y1 = _y1[self.n_example_points :]

            yield M_central, y0, dt, y1, y0_example, dt_example, y1_example
