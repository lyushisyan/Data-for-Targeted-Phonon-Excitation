
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# -----------------------------
# Global font size
# -----------------------------
plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral"
})

# -----------------------------
# 1. Frequency and Kn grids
# -----------------------------
n_omega = 1500
n_target = 260
n_kn = 240

Omega = np.linspace(0.000, 1, n_omega)
Omega_targ = np.linspace(0, 1, n_target)
Kn_values = np.logspace(-2, 2, n_kn)

# -----------------------------
# 2. Sinusoidal dispersion model
# Omega = sin(pi Q / 2)
# -----------------------------
Q = (2.0 / np.pi) * np.arcsin(Omega)

v = np.sqrt(1.0 - Omega**2)
v = np.maximum(v, 1e-4)

D = Q**2 / v

# -----------------------------
# 3. Simplified heat-capacity factor
# -----------------------------
C = np.exp(-1.2 * Omega)

# -----------------------------
# 4. Intrinsic mean free path model
# -----------------------------
Omega_c = 0.05
p_tau = 1.4

tau = (Omega + Omega_c)**(-p_tau)
ell_raw = v * tau

# -----------------------------
# 5. Equilibrium spectral conductivity weight
# -----------------------------
W_raw = D * C * v * ell_raw
W = W_raw / np.trapezoid(W_raw, Omega)

L = ell_raw / np.trapezoid(W * ell_raw, Omega)

# -----------------------------
# 6. Excitation and scattering models
# -----------------------------
Omega_grid = Omega[:, None]
Omega_targ_grid = Omega_targ[None, :]

alpha0 = 0.75
sigma_alpha = 0.055
alpha_amp = alpha0 * np.exp(-2.8 * Omega_targ_grid)
alpha = alpha_amp * np.exp(
    -((Omega_grid - Omega_targ_grid)**2) / (2.0 * sigma_alpha**2)
)

beta0 = 0.13
beta1 = 4.5
beta_power = 2.0
sigma_beta = 0.16
beta_amp = beta0 * (1.0 + beta1 * Omega_targ_grid**beta_power)
beta = beta_amp * np.exp(
    -((Omega_grid - Omega_targ_grid)**2) / (2.0 * sigma_beta**2)
)

# -----------------------------
# 7. Calculate eta(Omega_targ, Kn)
# -----------------------------
eta = np.zeros((n_kn, n_target))

for i, Kn in enumerate(Kn_values):
    s = Kn * L[:, None]
    R = (1.0 + alpha) * (1.0 + s) / (1.0 + beta + s) - 1.0
    eta[i, :] = np.trapezoid(W[:, None] * R, Omega, axis=0)

eta_percent = 100.0 * eta

# -----------------------------
# 8. Plot heatmap
# -----------------------------
fig, ax = plt.subplots(figsize=(8.0, 6.2))

vmax = np.nanmax(np.abs(eta_percent))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

mesh = ax.pcolormesh(
    Omega_targ,
    Kn_values,
    eta_percent,
    shading="auto",
    cmap="RdBu_r",
    norm=norm
)

ax.set_yscale("log")
ax.set_xlim(0, 1)
ax.set_ylim(Kn_values.min(), Kn_values.max())

# only symbols, no explanatory words
ax.set_xlabel(r"$\Omega_{\mathrm{targ}}$")
ax.set_ylabel(r"$\mathrm{Kn}$")

# no title
# no contour line
# no text annotations

ax.tick_params(axis='both', which='major', labelsize=20, length=6, width=1.2)
ax.tick_params(axis='both', which='minor', length=3, width=1.0)

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label(r"$\eta$", fontsize=24)
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig("Fig_Theory_eta_map_symbol_only.png", dpi=600, bbox_inches="tight")
plt.savefig("Fig_Theory_eta_map_symbol_only.pdf", bbox_inches="tight")
plt.show()