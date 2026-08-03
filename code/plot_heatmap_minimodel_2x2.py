import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ==========================================
# 0. Global style
# ==========================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 14,
    "axes.labelsize": 20,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 300,
})


# ==========================================
# 1. Material heatmap configuration
# ==========================================
materials = [
    {"name": "Ge", "dir": "Germanium", "f_debye": 7.5},
    {"name": "Si", "dir": "Silicon", "f_debye": 13.0},
    {"name": "3C-SiC", "dir": "3C-SiC", "f_debye": 20.0},
]

sizes = [
    10, 20, 50, 100, 200, 500, 1000,
    2000, 5000, 10000, 20000, 50000, 100000, 200000,
]
filename_pattern = "result_S1_Lx{}.txt"
e_folder_name = "E2"

cmap = "RdBu_r"
norm = TwoSlopeNorm(vmin=-20.0, vcenter=0.0, vmax=20.0)


def load_avg_mfp():
    mfp_csv = "MFP/mfp_summary.csv"
    avg_mfp_nm = {}

    if os.path.exists(mfp_csv):
        try:
            df_mfp = pd.read_csv(mfp_csv)
            for _, r in df_mfp.iterrows():
                key = str(r["Material"]).strip()
                avg_mfp_nm[key] = float(r["avg_MFP_nm"])
        except Exception:
            pass

    return avg_mfp_nm


def extract_pivot_300k(mat_root, mat_name):
    data_list = []

    for size in sizes:
        root = os.path.join(mat_root, f"Results_cross_Lx{int(size)}_target_300K")
        if not os.path.exists(root):
            continue

        filename = filename_pattern.format(int(size))
        kappa_0 = None

        for base_folder in ["Baseline", "Bseline"]:
            base_path = os.path.join(root, base_folder, filename)
            if os.path.exists(base_path):
                df_base = pd.read_csv(base_path)
                kappa_0 = float(df_base["Kappa_Sim"].iloc[0])
                break

        if not kappa_0:
            continue

        exc_root = os.path.join(root, e_folder_name)
        if not os.path.exists(exc_root):
            continue

        for item in os.listdir(exc_root):
            if not item.startswith("Freq_"):
                continue

            freq = float(item.split("_", 1)[1])
            target_file = os.path.join(exc_root, item, filename)

            if os.path.exists(target_file):
                kappa_exc = pd.read_csv(target_file)["Kappa_Sim"].iloc[0]
                data_list.append({
                    "Size": int(size),
                    "Frequency": freq,
                    "Ratio": kappa_exc / kappa_0,
                })

    if not data_list:
        return None

    df = pd.DataFrame(data_list)
    pivot_df = df.pivot_table(index="Size", columns="Frequency", values="Ratio")

    if "SiC" in mat_name:
        mask = (pivot_df.columns >= 18) & (pivot_df.columns <= 24)
        pivot_df.loc[:, mask] = 1.0
        pivot_df = pivot_df.iloc[:, ::2]
    else:
        pivot_df = pivot_df.iloc[:, :-1]

    return pivot_df


def plot_material_heatmap(ax, mat, panel_label, avg_mfp_nm):
    pivot_df = extract_pivot_300k(mat["dir"], mat["name"])
    if pivot_df is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return None

    x_freq_norm = pivot_df.columns.values / mat["f_debye"]
    y_size = pivot_df.index.values
    z_percent = 100.0 * (pivot_df.values - 1.0)

    h_nm = y_size / 10.0
    mfp_key = "Ge" if "Ge" in mat["name"] else ("SiC" if "SiC" in mat["name"] else "Si")
    mfp_val = avg_mfp_nm.get(mfp_key, 100.0)
    kn = mfp_val / h_nm

    mesh = ax.pcolormesh(
        x_freq_norm,
        kn,
        z_percent,
        cmap=cmap,
        norm=norm,
        shading="gouraud",
    )

    ax.set_yscale("log")
    ax.set_xlim(0.06, 0.98)
    ax.set_xlabel(r"$\omega_{\mathrm{t}} / \omega_{\mathrm{D}}$")
    ax.set_title(mat["name"], fontsize=18, pad=8)
    ax.text(0.04, 0.93, f"({panel_label})", transform=ax.transAxes, fontsize=17, va="top")

    return mesh


# ==========================================
# 2. Mini model
# ==========================================
def calculate_minimodel_eta():
    n_omega = 1500
    n_target = 260
    n_kn = 240

    omega = np.linspace(0.000, 1.0, n_omega)
    omega_targ = np.linspace(0.0, 1.0, n_target)
    kn_values = np.logspace(-2, 2, n_kn)

    q = (2.0 / np.pi) * np.arcsin(omega)
    v = np.sqrt(1.0 - omega**2)
    v = np.maximum(v, 1e-4)
    density = q**2 / v
    heat_capacity = np.exp(-1.2 * omega)

    omega_c = 0.05
    p_tau = 1.4
    tau = (omega + omega_c) ** (-p_tau)
    ell_raw = v * tau

    weight_raw = density * heat_capacity * v * ell_raw
    weight = weight_raw / np.trapezoid(weight_raw, omega)
    ell_norm = ell_raw / np.trapezoid(weight * ell_raw, omega)

    omega_grid = omega[:, None]
    omega_targ_grid = omega_targ[None, :]

    alpha0 = 0.75
    sigma_alpha = 0.055
    alpha_amp = alpha0 * np.exp(-2.8 * omega_targ_grid)
    alpha = alpha_amp * np.exp(
        -((omega_grid - omega_targ_grid) ** 2) / (2.0 * sigma_alpha**2)
    )

    beta0 = 0.13
    beta1 = 4.5
    beta_power = 2.0
    sigma_beta = 0.16
    beta_amp = beta0 * (1.0 + beta1 * omega_targ_grid**beta_power)
    beta = beta_amp * np.exp(
        -((omega_grid - omega_targ_grid) ** 2) / (2.0 * sigma_beta**2)
    )

    eta = np.zeros((n_kn, n_target))
    for i, kn in enumerate(kn_values):
        surface = kn * ell_norm[:, None]
        response = (1.0 + alpha) * (1.0 + surface) / (1.0 + beta + surface) - 1.0
        eta[i, :] = np.trapezoid(weight[:, None] * response, omega, axis=0)

    return omega_targ, kn_values, 100.0 * eta


def plot_minimodel(ax, panel_label):
    omega_targ, kn_values, eta_percent = calculate_minimodel_eta()

    mesh = ax.pcolormesh(
        omega_targ,
        kn_values,
        eta_percent,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(kn_values.min(), kn_values.max())
    ax.set_xlabel(r"$\omega_{\mathrm{t}} / \omega_{\mathrm{D}}$")
    ax.set_title("RSTM", fontsize=18, pad=8)
    ax.text(0.04, 0.93, f"({panel_label})", transform=ax.transAxes, fontsize=17, va="top")

    return mesh


# ==========================================
# 3. Combined 2x2 figure
# ==========================================
def main():
    avg_mfp_nm = load_avg_mfp()

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharey=False)
    axes_flat = axes.ravel()
    mappable = None

    for idx, mat in enumerate(materials):
        mesh = plot_material_heatmap(
            axes_flat[idx],
            mat,
            chr(97 + idx),
            avg_mfp_nm,
        )
        if mesh is not None:
            mappable = mesh

    mesh = plot_minimodel(axes_flat[3], "d")
    if mesh is not None:
        mappable = mesh

    for idx, ax in enumerate(axes_flat):
        if idx in (0, 1):
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        if idx in (0, 2):
            ax.set_ylabel(r"$\mathrm{Kn} = \langle \ell \rangle / H$")
        else:
            ax.tick_params(labelleft=False)

    fig.subplots_adjust(
        left=0.09,
        right=0.86,
        bottom=0.09,
        top=0.92,
        wspace=0.14,
        hspace=0.20,
    )

    if mappable is not None:
        cax = fig.add_axes([0.89, 0.16, 0.018, 0.68])
        cbar = fig.colorbar(mappable, cax=cax, extend="both")
        ticks = [-20, -10, 0, 10, 20]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])
        cbar.set_label(
            r"$(\kappa-\kappa_0)/\kappa_0$ (%)",
            rotation=270,
            labelpad=28,
            fontsize=18,
        )
        cbar.ax.tick_params(labelsize=13, direction="in")
        cbar.outline.set_linewidth(1.5)

    output_name = "Heatmap_Materials_MiniModel_2x2.png"
    pdf_output_name = "Heatmap_Materials_MiniModel_2x2.pdf"
    fig.savefig(output_name, bbox_inches="tight")
    fig.savefig(pdf_output_name, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_name}")
    print(f"Saved figure: {pdf_output_name}")


if __name__ == "__main__":
    main()
