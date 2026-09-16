"""
visualize.py

All matplotlib plots for the vibration simulator validation study.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from simulator import SystemParams, analytical_free_response, simulate

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 10


def plot_regime_comparison(params_list, labels, x0, v0, h, t_end, save_path):
    """Time-response: numerical (RK4) vs analytical, one subplot per damping regime."""
    fig, axes = plt.subplots(1, len(params_list), figsize=(5 * len(params_list), 4), sharey=False)
    if len(params_list) == 1:
        axes = [axes]

    for ax, params, label in zip(axes, params_list, labels):
        t, x_num, _ = simulate(params, "rk4", h, t_end, x0, v0)
        x_true = analytical_free_response(params, t, x0, v0)

        ax.plot(t, x_true, color="#1B2735", linewidth=2.5, label="Analytical", alpha=0.9)
        ax.plot(t, x_num, color="#C4622D", linewidth=1.2, linestyle="--", label="RK4 (numerical)")
        ax.set_title(f"{label}\n(ζ={params.zeta:.2f})")
        ax.set_xlabel("Time (s)")
        ax.axhline(0, color="gray", linewidth=0.5)
        if ax is axes[0]:
            ax.set_ylabel("Displacement x(t) (m)")
        ax.legend(fontsize=8)

    fig.suptitle("Free Vibration: Numerical vs. Analytical Solution", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_convergence(conv_df, orders, save_path):
    """Log-log plot of max error vs step size for each method, annotated with fitted order."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"euler": "#C4622D", "heun": "#2F8F6E", "rk4": "#3D5A73"}
    markers = {"euler": "o", "heun": "s", "rk4": "^"}

    for method in ["euler", "heun", "rk4"]:
        sub = conv_df[conv_df.method == method].sort_values("h")
        ax.loglog(sub.h, sub.max_abs_error, marker=markers[method], color=colors[method],
                   label=f"{method} (fitted order {orders[method]:.2f})", linewidth=1.5)

    ax.set_xlabel("Step size h (s)")
    ax.set_ylabel("Max absolute error vs. analytical (m)")
    ax.set_title("Convergence Study: Error vs. Step Size (log-log)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_phase_portrait(params_list, labels, x0, v0, h, t_end, save_path):
    """Phase portrait (x vs v) for each damping regime — visualizes energy dissipation."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = ["#2F8F6E", "#3D5A73", "#C4622D"]
    for params, label, color in zip(params_list, labels, colors):
        t, x_num, v_num = simulate(params, "rk4", h, t_end, x0, v0)
        ax.plot(x_num, v_num, color=color, linewidth=1.5, label=f"{label} (ζ={params.zeta:.2f})")
        ax.plot(x0, v0, "o", color=color, markersize=6)

    ax.set_xlabel("Displacement x (m)")
    ax.set_ylabel("Velocity v (m/s)")
    ax.set_title("Phase Portrait — Spiral Inward Shows Energy Dissipation")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_resonance_curve(resonance_df, save_path):
    """Amplitude vs frequency ratio: numerical points over the exact theoretical curve."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(resonance_df.freq_ratio_r, resonance_df.amplitude_theory_m,
            color="#1B2735", linewidth=2, label="Theoretical frequency response")
    ax.plot(resonance_df.freq_ratio_r, resonance_df.amplitude_numerical_m,
            "o", color="#C4622D", markersize=7, label="Simulated (RK4) steady-state amplitude")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, label="Resonance (r=1)")
    ax.set_xlabel("Frequency ratio r = ω / ωₙ")
    ax.set_ylabel("Steady-state amplitude (m)")
    ax.set_title("Forced Vibration: Resonance Curve — Simulation vs. Theory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_energy_decay(params, x0, v0, h, t_end, save_path):
    """Total mechanical energy over time — should decay monotonically for a damped free system."""
    t, x_num, v_num = simulate(params, "rk4", h, t_end, x0, v0)
    KE = 0.5 * params.m * v_num ** 2
    PE = 0.5 * params.k * x_num ** 2
    E = KE + PE

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, KE, label="Kinetic energy", color="#3D5A73")
    ax.plot(t, PE, label="Potential energy", color="#2F8F6E")
    ax.plot(t, E, label="Total mechanical energy", color="#1B2735", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title(f"Energy Dissipation (ζ={params.zeta:.2f}) — Damping Removes Mechanical Energy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
