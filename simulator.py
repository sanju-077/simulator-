"""
simulator.py

A single-degree-of-freedom spring-mass-damper system:

    m*x'' + c*x' + k*x = F(t)

This module provides:
  - The exact analytical solution for free vibration (F=0), covering all
    three damping regimes (under/critically/over-damped) with closed-form
    formulas from vibrations theory (see e.g. Rao, "Mechanical Vibrations").
  - The exact steady-state analytical solution for harmonic forcing
    (frequency-response / resonance formulas).
  - Three numerical integrators implemented from scratch (no scipy):
    Forward Euler (1st order), Heun's method / RK2 (2nd order), and
    classical RK4 (4th order) — used later to demonstrate their different
    convergence rates against the known analytical solution.

Units are SI throughout (kg, N/m, N*s/m, seconds) unless noted.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


@dataclass
class SystemParams:
    m: float  # mass, kg
    c: float  # damping coefficient, N*s/m
    k: float  # spring stiffness, N/m

    @property
    def omega_n(self) -> float:
        """Undamped natural frequency, rad/s."""
        return np.sqrt(self.k / self.m)

    @property
    def zeta(self) -> float:
        """Damping ratio (dimensionless). zeta<1 under, =1 critical, >1 over."""
        return self.c / (2 * np.sqrt(self.k * self.m))

    @property
    def regime(self) -> str:
        z = self.zeta
        if np.isclose(z, 1.0, atol=1e-9):
            return "critically damped"
        return "underdamped" if z < 1.0 else "overdamped"


# --------------------------------------------------------------------------
# Analytical solutions (free vibration, F(t) = 0)
# --------------------------------------------------------------------------

def analytical_free_response(
    params: SystemParams, t: np.ndarray, x0: float, v0: float
) -> np.ndarray:
    """
    Exact closed-form solution to m*x'' + c*x' + k*x = 0 for the given
    initial conditions x(0)=x0, x'(0)=v0. Dispatches on damping regime.
    """
    wn = params.omega_n
    z = params.zeta

    if params.regime == "underdamped":
        wd = wn * np.sqrt(1 - z ** 2)
        A = x0
        B = (v0 + z * wn * x0) / wd
        return np.exp(-z * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))

    elif params.regime == "critically damped":
        A = x0
        B = v0 + wn * x0
        return np.exp(-wn * t) * (A + B * t)

    else:  # overdamped
        disc = wn * np.sqrt(z ** 2 - 1)
        r1 = -z * wn + disc
        r2 = -z * wn - disc
        C1 = (v0 - r2 * x0) / (r1 - r2)
        C2 = x0 - C1
        return C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)


# --------------------------------------------------------------------------
# Analytical steady-state response to harmonic forcing F(t) = F0*cos(w t)
# --------------------------------------------------------------------------

def steady_state_amplitude_and_phase(
    params: SystemParams, F0: float, w: float
) -> tuple[float, float]:
    """
    Exact steady-state amplitude X and phase lag phi (radians) for
    m*x'' + c*x' + k*x = F0*cos(w*t), i.e. x_ss(t) = X*cos(w*t - phi).

    Standard frequency-response formulas in terms of the frequency ratio
    r = w/omega_n and damping ratio zeta:
        X   = (F0/k) / sqrt((1 - r^2)^2 + (2*zeta*r)^2)
        phi = atan2(2*zeta*r, 1 - r^2)
    """
    wn = params.omega_n
    z = params.zeta
    r = w / wn
    static_defl = F0 / params.k
    X = static_defl / np.sqrt((1 - r ** 2) ** 2 + (2 * z * r) ** 2)
    phi = np.arctan2(2 * z * r, 1 - r ** 2)
    return X, phi


# --------------------------------------------------------------------------
# Numerical integration, hand-rolled (state vector y = [x, v])
# --------------------------------------------------------------------------

def _derivative(
    t: float, y: np.ndarray, params: SystemParams, F: Callable[[float], float]
) -> np.ndarray:
    x, v = y
    a = (F(t) - params.c * v - params.k * x) / params.m
    return np.array([v, a])


def simulate(
    params: SystemParams,
    method: str,
    h: float,
    t_end: float,
    x0: float,
    v0: float,
    F: Optional[Callable[[float], float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the system from t=0 to t=t_end with fixed step size h using
    one of: 'euler' (forward Euler, O(h) global error),
            'heun'  (RK2 / improved Euler, O(h^2)),
            'rk4'   (classical 4th-order Runge-Kutta, O(h^4)).

    Returns (t_array, x_array, v_array).
    """
    if F is None:
        F = lambda t: 0.0

    n_steps = int(round(t_end / h))
    t = np.linspace(0, n_steps * h, n_steps + 1)
    y = np.zeros((n_steps + 1, 2))
    y[0] = [x0, v0]

    for i in range(n_steps):
        ti, yi = t[i], y[i]

        if method == "euler":
            y[i + 1] = yi + h * _derivative(ti, yi, params, F)

        elif method == "heun":
            k1 = _derivative(ti, yi, params, F)
            y_pred = yi + h * k1
            k2 = _derivative(ti + h, y_pred, params, F)
            y[i + 1] = yi + (h / 2) * (k1 + k2)

        elif method == "rk4":
            k1 = _derivative(ti, yi, params, F)
            k2 = _derivative(ti + h / 2, yi + h / 2 * k1, params, F)
            k3 = _derivative(ti + h / 2, yi + h / 2 * k2, params, F)
            k4 = _derivative(ti + h, yi + h * k3, params, F)
            y[i + 1] = yi + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        else:
            raise ValueError(f"Unknown method: {method!r}")

    return t, y[:, 0], y[:, 1]
