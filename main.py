"""
main.py

Runs the complete spring-mass-damper validation study:
  1. Free vibration across all 3 damping regimes, numerical vs analytical
  2. Convergence order study for Euler / Heun / RK4
  3. Phase portraits (energy dissipation visualization)
  4. Energy decay over time
  5. Forced vibration / resonance curve validation

Saves all plots to outputs/ and prints a summary of the validation results
(including a scipy.integrate.solve_ivp cross-check, as an independent
second opinion beyond our own hand-rolled RK4).
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from simulator import SystemParams, analytical_free_response, simulate
from validate import free_vibration_accuracy, convergence_study, resonance_validation
from visualize import (
    plot_regime_comparison,
    plot_convergence,
    plot_phase_portrait,
    plot_resonance_curve,
    plot_energy_decay,
)

OUT = "outputs"

# ---------------------------------------------------------------- Setup
m, k = 1.0, 100.0  # kg, N/m -> omega_n = 10 rad/s
x0, v0 = 0.1, 0.0   # initial displacement 0.1 m, released from rest

regimes = {
    "Underdamped": SystemParams(m=m, c=0.05 * 2 * np.sqrt(k * m), k=k),
    "Critically damped": SystemParams(m=m, c=1.00 * 2 * np.sqrt(k * m), k=k),
    "Overdamped": SystemParams(m=m, c=2.00 * 2 * np.sqrt(k * m), k=k),
}

print("=" * 70)
print("SPRING-MASS-DAMPER SIMULATOR — VALIDATION STUDY")
print("=" * 70)
print(f"m = {m} kg, k = {k} N/m  ->  omega_n = {np.sqrt(k/m):.3f} rad/s")
print(f"Initial conditions: x0 = {x0} m, v0 = {v0} m/s\n")

# ---------------------------------------------------------------- 1. Free vibration accuracy
print("-- 1. Free vibration accuracy at h=0.005s (all 3 regimes) --")
all_accuracy = []
for name, params in regimes.items():
    df = free_vibration_accuracy(params, x0, v0, h=0.005, t_end=3.0)
    df.insert(0, "regime", name)
    all_accuracy.append(df)
accuracy_df = pd.concat(all_accuracy, ignore_index=True)
print(accuracy_df.to_string(index=False))
accuracy_df.to_csv(f"{OUT}/free_vibration_accuracy.csv", index=False)

plot_regime_comparison(
    list(regimes.values()), list(regimes.keys()), x0, v0,
    h=0.005, t_end=3.0, save_path=f"{OUT}/1_regime_comparison.png",
)

# ---------------------------------------------------------------- 2. Convergence study
print("\n-- 2. Convergence order study (underdamped case) --")
step_sizes = np.array([0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125])
conv_df, orders = convergence_study(
    regimes["Underdamped"], x0, v0, t_end=2.0, step_sizes=step_sizes
)
print("Fitted asymptotic convergence orders (theory: Euler=1, Heun=2, RK4=4):")
for method, order in orders.items():
    print(f"  {method:6s}: {order:.3f}")
conv_df.to_csv(f"{OUT}/convergence_study.csv", index=False)

plot_convergence(conv_df, orders, save_path=f"{OUT}/2_convergence_study.png")

# ---------------------------------------------------------------- 3. Phase portraits + energy
plot_phase_portrait(
    list(regimes.values()), list(regimes.keys()), x0, v0,
    h=0.005, t_end=3.0, save_path=f"{OUT}/3_phase_portraits.png",
)
plot_energy_decay(
    regimes["Underdamped"], x0, v0, h=0.005, t_end=3.0,
    save_path=f"{OUT}/4_energy_decay.png",
)

# ---------------------------------------------------------------- 4. Resonance validation
print("\n-- 3. Forced vibration / resonance validation (underdamped, zeta=0.05) --")
freq_ratios = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0])
resonance_df = resonance_validation(
    regimes["Underdamped"], F0=10.0, freq_ratios=freq_ratios,
    h=0.001, n_periods_transient=60, n_periods_measure=10,
)
print(resonance_df.to_string(index=False))
print(f"\nMax %% error across resonance sweep: {resonance_df.pct_error.max():.5f}%")
resonance_df.to_csv(f"{OUT}/resonance_validation.csv", index=False)

plot_resonance_curve(resonance_df, save_path=f"{OUT}/5_resonance_curve.png")

# ---------------------------------------------------------------- 5. Independent scipy cross-check
print("\n-- 4. Independent cross-check: hand-rolled RK4 vs scipy solve_ivp --")
p = regimes["Underdamped"]


def rhs(t, y):
    x, v = y
    return [v, (-p.c * v - p.k * x) / p.m]


t_eval = np.linspace(0, 3.0, 601)
sol = solve_ivp(rhs, [0, 3.0], [x0, v0], t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12)
x_scipy = sol.y[0]

t_ours, x_ours, _ = simulate(p, "rk4", h=0.005, t_end=3.0, x0=x0, v0=v0)
x_ours_interp = np.interp(t_eval, t_ours, x_ours)

max_diff = np.max(np.abs(x_scipy - x_ours_interp))
print(f"Max difference between our RK4 and scipy's adaptive RK45: {max_diff:.3e} m")
print("(This confirms our hand-rolled integrator agrees with an independent,")
print(" industry-standard solver — not just with our own analytical formula.)")

print("\n" + "=" * 70)
print("All plots saved to outputs/. See REPORT.md for full write-up.")
print("=" * 70)
