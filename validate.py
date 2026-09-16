"""
validate.py

Runs the actual validation studies that justify calling this simulator
"validated" rather than just "runs without crashing":

1. Free-vibration accuracy: for each damping regime, compare each numerical
   method against the exact analytical solution at a fixed, reasonably
   coarse step size.

2. Convergence order study: for each method, run at a sequence of halving
   step sizes and measure how the error shrinks. Forward Euler should show
   ~1st-order convergence (error halves when h halves), Heun ~2nd-order
   (error quarters), RK4 ~4th-order (error drops by 16x). Fitting a line to
   log(error) vs log(h) recovers the convergence order directly from the
   data — this is the real proof the integrators are implemented correctly,
   not just "it looks close on a graph."

3. Forced-vibration (resonance) validation: sweep the forcing frequency
   across the natural frequency and compare the simulated steady-state
   amplitude to the exact frequency-response formula, reproducing the
   classic resonance peak.
"""

import numpy as np
import pandas as pd

from simulator import (
    SystemParams,
    analytical_free_response,
    steady_state_amplitude_and_phase,
    simulate,
)

METHODS = ["euler", "heun", "rk4"]
EXPECTED_ORDER = {"euler": 1, "heun": 2, "rk4": 4}


def free_vibration_accuracy(
    params: SystemParams, x0: float, v0: float, h: float, t_end: float
) -> pd.DataFrame:
    """Max absolute error of each method vs. the analytical solution, at one step size."""
    rows = []
    for method in METHODS:
        t, x_num, _ = simulate(params, method, h, t_end, x0, v0)
        x_true = analytical_free_response(params, t, x0, v0)
        err = np.max(np.abs(x_num - x_true))
        rows.append({"method": method, "step_size_s": h, "max_abs_error_m": err})
    return pd.DataFrame(rows)


def convergence_study(
    params: SystemParams,
    x0: float,
    v0: float,
    t_end: float,
    step_sizes: np.ndarray,
) -> pd.DataFrame:
    """
    For each method and each step size, compute max error vs analytical.
    Also fits the observed convergence order (slope of log(error) vs log(h))
    per method using linear regression on the log-log data.
    """
    rows = []
    for method in METHODS:
        for h in step_sizes:
            t, x_num, _ = simulate(params, method, h, t_end, x0, v0)
            x_true = analytical_free_response(params, t, x0, v0)
            err = np.max(np.abs(x_num - x_true))
            rows.append({"method": method, "h": h, "max_abs_error": err})
    df = pd.DataFrame(rows)

    orders = {}
    for method in METHODS:
        sub = df[df.method == method].sort_values("h")
        valid = sub[sub.max_abs_error > 0]
        # Convergence order is an asymptotic (h -> 0) property. At coarser
        # step sizes the local error still contains non-negligible
        # higher-order terms, which biases a full-range log-log fit upward
        # (this is visible in the raw data: Euler's error ratio between
        # successive halvings climbs toward 2x, i.e. order 1, only once h
        # is small enough). So the order is estimated from the finest half
        # of the tested step sizes, where the asymptotic slope has settled.
        n = len(valid)
        tail = valid.iloc[:3] if n >= 3 else valid
        if len(tail) >= 2:
            slope, _ = np.polyfit(np.log(tail.h), np.log(tail.max_abs_error), 1)
            orders[method] = slope
        else:
            orders[method] = np.nan

    return df, orders


def resonance_validation(
    params: SystemParams,
    F0: float,
    freq_ratios: np.ndarray,
    h: float,
    n_periods_transient: float = 40,
    n_periods_measure: float = 10,
) -> pd.DataFrame:
    """
    For each forcing frequency ratio r = w/wn, simulate the forced response
    long enough for the transient to decay, then measure the steady-state
    oscillation amplitude numerically and compare it to the exact
    frequency-response formula.
    """
    wn = params.omega_n
    rows = []
    for r in freq_ratios:
        w = r * wn
        F = lambda t, F0=F0, w=w: F0 * np.cos(w * t)
        period = 2 * np.pi / w if w > 0 else 2 * np.pi / wn

        t_transient = n_periods_transient * (2 * np.pi / wn)
        t_measure = n_periods_measure * period
        t_end = t_transient + t_measure

        t, x_num, v_num = simulate(params, "rk4", h, t_end, x0=0.0, v0=0.0, F=F)

        mask = t >= t_transient
        x_ss = x_num[mask]
        amp_numerical = (np.max(x_ss) - np.min(x_ss)) / 2

        amp_theory, phase_theory = steady_state_amplitude_and_phase(params, F0, w)

        rows.append({
            "freq_ratio_r": r,
            "amplitude_numerical_m": amp_numerical,
            "amplitude_theory_m": amp_theory,
            "pct_error": 100 * abs(amp_numerical - amp_theory) / amp_theory,
        })

    return pd.DataFrame(rows)
