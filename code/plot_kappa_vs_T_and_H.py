import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


# ============================================================
# 0. 全局绘图风格：两个图统一使用这一套
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',

    'font.size': 14,
    'axes.labelsize': 22,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,

    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.top': True,
    'ytick.right': True,

    'lines.linewidth': 2,
    'lines.markeredgewidth': 1.5,
    'legend.frameon': False,
    'figure.dpi': 300,
})


# ============================================================
# 1. 统一配置
# ============================================================
COLORS_MAP = {
    'low': '#D9534F',
    'high': '#104E8B',
}

MARKERS_MAP = {
    'low': 's',
    'high': '^',
}

ENERGY_TAG = "E3"

SIZES = [
    10, 20, 50, 100, 200, 500,
    1000, 2000, 5000, 10000,
    20000, 50000, 100000, 200000
]

TEMPS_K = [100, 200, 300, 400, 500, 600, 1000, 2000]
XTICKS_LOG = [100, 200, 500, 1000, 2000]

MATERIALS = [
    {
        "name": "Ge",
        "base_root": "Germanium",
        "bulk_dir": "Germanium/Ge_kappa_bulk",
        "bulk_csv_300K": "Germanium/Ge_kappa_bulk/Kappa_Excitation_300K.csv",
        "high_root": "Ge-1000K",
        "freqs": [0.8, 5.6],
        "sizes": SIZES,
    },
    {
        "name": "Si",
        "base_root": "Silicon",
        "bulk_dir": "Silicon/Si_kappa_bulk",
        "bulk_csv_300K": "Silicon/Si_kappa_bulk/Kappa_Excitation_300K.csv",
        "high_root": "Si-1000K",
        "freqs": [1.2, 10.0],
        "sizes": SIZES,
    },
    {
        "name": "SiC",
        "base_root": "3C-SiC",
        "bulk_dir": "3C-SiC/SiC_kappa_bulk",
        "bulk_csv_300K": "3C-SiC/SiC_kappa_bulk/Kappa_Excitation_300K.csv",
        "high_root": "3C-SiC-1000K",
        "freqs": [2.0, 15.0],
        "sizes": SIZES,
    },
]

OUT_FIG_H = "Kappa_vs_H_Scale.png"
OUT_FIG_T = "Kappa_vs_T_Bulk_Film100nm.png"


# ============================================================
# 2. 通用辅助函数
# ============================================================
def read_kappa_sim(file_path):
    """
    读取 result_S1_Lx*.txt 中的 Kappa_Sim。
    """
    if file_path is None or not os.path.exists(file_path):
        return None

    try:
        df = pd.read_csv(file_path)

        if "Kappa_Sim" not in df.columns or df.empty:
            return None

        return float(df["Kappa_Sim"].iloc[0])

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def find_freq_dir(e_root, target_freq, tol=1e-6):
    """
    在 E3 文件夹中寻找与目标频率匹配的 Freq_* 文件夹。

    可兼容：
        Freq_10
        Freq_10.0
        Freq_10.000000
    """
    if not os.path.isdir(e_root):
        return None

    best_name = None
    best_diff = None

    for name in os.listdir(e_root):
        if not name.startswith("Freq_"):
            continue

        try:
            freq_value = float(name.split("_", 1)[1])
        except Exception:
            continue

        diff = abs(freq_value - target_freq)

        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_name = name

    if best_diff is not None and best_diff < tol:
        return best_name

    return None


def baseline_file_100nm(folder):
    """
    读取 100 nm 薄膜对应的 baseline 文件。
    100 nm 对应 Lx1000。
    兼容 Baseline 和 Bseline 两种拼写。
    """
    file_name = "result_S1_Lx1000.txt"

    p1 = os.path.join(folder, "Baseline", file_name)
    if os.path.exists(p1):
        return p1

    p2 = os.path.join(folder, "Bseline", file_name)
    if os.path.exists(p2):
        return p2

    return None


def baseline_file_by_size(folder, size):
    """
    用于厚度依赖图的 baseline 文件。
    兼容 Baseline 和 Bseline 两种拼写。
    """
    file_name = f"result_S1_Lx{int(size)}.txt"

    p1 = os.path.join(folder, "Baseline", file_name)
    if os.path.exists(p1):
        return p1

    p2 = os.path.join(folder, "Bseline", file_name)
    if os.path.exists(p2):
        return p2

    return None


def bulk_csv_for_temperature(mat, temp_k):
    """
    根据温度返回 bulk CSV 文件路径。

    100-600 K 使用 base_root。
    1000 K 和 2000 K 使用 high_root。
    """
    if temp_k >= 1000:
        bulk_subdir = os.path.basename(mat["bulk_dir"])
        return os.path.join(
            mat["high_root"],
            bulk_subdir,
            f"Kappa_Excitation_{temp_k}K.csv"
        )

    return os.path.join(
        mat["bulk_dir"],
        f"Kappa_Excitation_{temp_k}K.csv"
    )


def root_for_temperature(mat, temp_k):
    """
    根据温度选择根目录。
    """
    if temp_k >= 1000:
        return mat["high_root"]

    return mat["base_root"]


# ============================================================
# 3. 图 1：kappa vs H
# ============================================================
def collect_kappa_vs_H():
    """
    收集 300 K 下，热导率随薄膜厚度 H 的变化数据。
    H = Lx / 10，单位 nm。
    """
    records = []

    for mat in MATERIALS:
        name = mat["name"]
        root = mat["base_root"]
        freqs = mat["freqs"]

        # ---------- bulk 参考值 ----------
        k_bulk = None
        bulk_csv = mat["bulk_csv_300K"]

        if os.path.exists(bulk_csv):
            try:
                df_bulk = pd.read_csv(bulk_csv)

                if "Original" in df_bulk.columns and not df_bulk.empty:
                    k_bulk = float(df_bulk["Original"].iloc[0])

            except Exception as e:
                print(f"Error reading bulk file for {name}: {e}")
        else:
            print(f"Warning: missing bulk CSV for {name}: {bulk_csv}")

        # ---------- film baseline 和 excited ----------
        for size in mat["sizes"]:
            folder_name = f"Results_cross_Lx{int(size)}_target_300K"
            folder = os.path.join(root, folder_name)
            file_name = f"result_S1_Lx{int(size)}.txt"

            H_nm = size / 10.0

            # baseline
            base_file = baseline_file_by_size(folder, size)
            k_base = read_kappa_sim(base_file)

            records.append({
                "Material": name,
                "H_nm": H_nm,
                "Size_Lx_A": size,
                "State": "Unexcited",
                "Frequency_THz": None,
                "Kappa_W_mK": k_base,
                "Bulk_Kappa_W_mK": k_bulk,
            })

            # excited
            e_root = os.path.join(folder, ENERGY_TAG)

            for freq in freqs:
                freq_dir = find_freq_dir(e_root, freq)
                k_exc = None

                if freq_dir is not None:
                    exc_file = os.path.join(e_root, freq_dir, file_name)
                    k_exc = read_kappa_sim(exc_file)

                records.append({
                    "Material": name,
                    "H_nm": H_nm,
                    "Size_Lx_A": size,
                    "State": "Excited",
                    "Frequency_THz": freq,
                    "Kappa_W_mK": k_exc,
                    "Bulk_Kappa_W_mK": k_bulk,
                })

    return pd.DataFrame(records)


def plot_kappa_vs_H(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for i, mat in enumerate(MATERIALS):
        ax = axes[i]
        name = mat["name"]
        freqs = mat["freqs"]

        sub_mat = df[df["Material"] == name].copy()

        # ---------- Bulk reference ----------
        k_bulk_values = sub_mat["Bulk_Kappa_W_mK"].dropna().unique()

        if len(k_bulk_values) > 0:
            k_bulk = k_bulk_values[0]

            ax.axhline(
                y=k_bulk,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"{name} bulk"
            )

        # ---------- Unexcited ----------
        sub_base = sub_mat[sub_mat["State"] == "Unexcited"].copy()
        sub_base = sub_base.dropna(subset=["Kappa_W_mK"])
        sub_base = sub_base.sort_values("H_nm")

        if not sub_base.empty:
            ax.plot(
                sub_base["H_nm"],
                sub_base["Kappa_W_mK"],
                label="Film unexcited",
                color="k",
                marker="o",
                markerfacecolor="none",
                markersize=7,
                linestyle="-"
            )

        # ---------- Excited ----------
        for idx, freq in enumerate(freqs):
            key = "low" if idx == 0 else "high"

            sub_exc = sub_mat[
                (sub_mat["State"] == "Excited") &
                (sub_mat["Frequency_THz"] == freq)
            ].copy()

            sub_exc = sub_exc.dropna(subset=["Kappa_W_mK"])
            sub_exc = sub_exc.sort_values("H_nm")

            if not sub_exc.empty:
                ax.plot(
                    sub_exc["H_nm"],
                    sub_exc["Kappa_W_mK"],
                    label=f"Film excited ({freq} THz)",
                    color=COLORS_MAP[key],
                    marker=MARKERS_MAP[key],
                    markerfacecolor="none",
                    markersize=7,
                    linestyle="--"
                )

        ax.set_xscale("log")
        ax.set_yscale("log")

        # 和另一个图保持一致
        ax.set_xlabel(r"$H$ (nm)", fontsize=22)

        if i == 0:
            ax.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)", fontsize=22)

        # 标题字号调大，与另一个图一致
        ax.set_title(
            f"({chr(97 + i)}) {name}",
            loc="left",
            fontweight="normal",
            fontsize=20
        )

        # 刻度字号显式设置
        ax.tick_params(axis='both', which='major', labelsize=14)

        ax.grid(True, which="both", ls=":", alpha=0.3)

        # 图例字号和另一个图统一
        ax.legend(loc="best", fontsize=14, frameon=False)

    plt.tight_layout()
    fig.savefig(OUT_FIG_H, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 4. 图 2：kappa vs T
# ============================================================
def bulk_values(mat, temp_k, freq_thz):
    """
    读取 bulk 的 unexcited 和 excited 热导率。
    """
    bulk_csv = bulk_csv_for_temperature(mat, temp_k)

    if not os.path.exists(bulk_csv):
        print(f"Warning: missing bulk CSV: {bulk_csv}")
        return None, None

    try:
        df = pd.read_csv(bulk_csv)

        if df.empty:
            return None, None

        if "Frequency_THz" not in df.columns:
            return None, None

        idx = (df["Frequency_THz"] - freq_thz).abs().idxmin()

        k_unexcited = None
        k_excited = None

        if "Original" in df.columns:
            k_unexcited = float(df.loc[idx, "Original"])

        if ENERGY_TAG in df.columns:
            k_excited = float(df.loc[idx, ENERGY_TAG])

        return k_unexcited, k_excited

    except Exception as e:
        print(f"Error reading bulk CSV {bulk_csv}: {e}")
        return None, None


def film_values_100nm(mat, temp_k, freq_thz):
    """
    读取 100 nm thin film 的 unexcited 和 excited 热导率。
    100 nm 对应 Lx1000。
    """
    root = root_for_temperature(mat, temp_k)
    folder = os.path.join(root, f"Results_cross_Lx1000_target_{temp_k}K")

    base_file = baseline_file_100nm(folder)
    k_unexcited = read_kappa_sim(base_file)

    e_root = os.path.join(folder, ENERGY_TAG)
    freq_dir = find_freq_dir(e_root, freq_thz)

    k_excited = None

    if freq_dir is not None:
        exc_file = os.path.join(e_root, freq_dir, "result_S1_Lx1000.txt")
        k_excited = read_kappa_sim(exc_file)

    return k_unexcited, k_excited


def collect_kappa_vs_T():
    records = []

    for mat in MATERIALS:
        for freq in mat["freqs"]:
            for temp in TEMPS_K:
                k_bulk_0, k_bulk_e = bulk_values(mat, temp, freq)
                k_film_0, k_film_e = film_values_100nm(mat, temp, freq)

                records.append({
                    "Material": mat["name"],
                    "Frequency_THz": freq,
                    "Temperature_K": temp,
                    "Bulk_Unexcited_W_mK": k_bulk_0,
                    "Bulk_Excited_W_mK": k_bulk_e,
                    "Film100nm_Unexcited_W_mK": k_film_0,
                    "Film100nm_Excited_W_mK": k_film_e,
                })

    return pd.DataFrame(records)


def plot_kappa_vs_T(df):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True)

    for col, mat in enumerate(MATERIALS):
        name = mat["name"]
        freqs = mat["freqs"]

        ax_bulk = axes[0, col]
        ax_film = axes[1, col]

        # 当前数据中 unexcited 与频率无关，所以只画一次
        base_freq = freqs[0]

        sub_base = df[
            (df["Material"] == name) &
            (df["Frequency_THz"] == base_freq)
        ].copy()

        sub_base = sub_base.sort_values("Temperature_K")

        # ---------- Bulk unexcited ----------
        ax_bulk.plot(
            sub_base["Temperature_K"],
            sub_base["Bulk_Unexcited_W_mK"],
            color="k",
            marker="o",
            markerfacecolor="none",
            linestyle="-",
            label="Unexcited",
        )

        # ---------- Film unexcited ----------
        ax_film.plot(
            sub_base["Temperature_K"],
            sub_base["Film100nm_Unexcited_W_mK"],
            color="k",
            marker="o",
            markerfacecolor="none",
            linestyle="-",
            label="Unexcited",
        )

        # ---------- Excited ----------
        for idx, freq in enumerate(freqs):
            key = "low" if idx == 0 else "high"

            sub = df[
                (df["Material"] == name) &
                (df["Frequency_THz"] == freq)
            ].copy()

            sub = sub.sort_values("Temperature_K")

            ax_bulk.plot(
                sub["Temperature_K"],
                sub["Bulk_Excited_W_mK"],
                color=COLORS_MAP[key],
                marker=MARKERS_MAP[key],
                markerfacecolor="none",
                linestyle="--",
                label=f"Excited ({freq} THz)",
            )

            ax_film.plot(
                sub["Temperature_K"],
                sub["Film100nm_Excited_W_mK"],
                color=COLORS_MAP[key],
                marker=MARKERS_MAP[key],
                markerfacecolor="none",
                linestyle="--",
                label=f"Excited ({freq} THz)",
            )

        # ---------- 标题和标注 ----------
        ax_bulk.set_title(name, fontsize=20, y=1.02)

        ax_bulk.text(
            0.96, 0.94, "Bulk",
            transform=ax_bulk.transAxes,
            ha="right",
            va="top",
            fontsize=16
        )

        ax_film.text(
            0.96, 0.94, "Thin film (100 nm)",
            transform=ax_film.transAxes,
            ha="right",
            va="top",
            fontsize=16
        )

        ax_bulk.text(
            0.04, 0.94, f"({chr(97 + col)})",
            transform=ax_bulk.transAxes,
            fontsize=18,
            va="top"
        )

        ax_film.text(
            0.04, 0.94, f"({chr(100 + col)})",
            transform=ax_film.transAxes,
            fontsize=18,
            va="top"
        )

        if col == 0:
            ax_bulk.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)")
            ax_film.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)")
        else:
            ax_bulk.tick_params(labelleft=False)
            ax_film.tick_params(labelleft=False)

        ax_film.set_xlabel("Temperature (K)")
        ax_bulk.tick_params(labelbottom=False)

        ax_bulk.set_xscale("log")
        ax_film.set_xscale("log")

        ax_film.xaxis.set_major_locator(FixedLocator(XTICKS_LOG))
        ax_film.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{int(x)}" if x in XTICKS_LOG else "")
        )

        for ax in (ax_bulk, ax_film):
            ax.grid(True, which="both", ls=":", alpha=0.3)

            y_min, y_max = ax.get_ylim()
            span = y_max - y_min

            if span <= 0:
                span = max(abs(y_min), 1.0)

            pad = 0.15 * span
            ax.set_ylim(y_min - pad, y_max + pad)

        # 每列底部放一个图例
        legend_handles = [
            Line2D(
                [0], [0],
                color="k",
                marker="o",
                markerfacecolor="none",
                linestyle="-",
                label="Unexcited"
            ),
            Line2D(
                [0], [0],
                color=COLORS_MAP["low"],
                marker=MARKERS_MAP["low"],
                markerfacecolor="none",
                linestyle="--",
                label=f"Excited ({freqs[0]} THz)"
            ),
            Line2D(
                [0], [0],
                color=COLORS_MAP["high"],
                marker=MARKERS_MAP["high"],
                markerfacecolor="none",
                linestyle="--",
                label=f"Excited ({freqs[1]} THz)"
            ),
        ]

        ax_film.legend(
            handles=legend_handles,
            loc="lower left",
            frameon=False,
            fontsize=14
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG_T, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. 主程序：一次性绘制两个图，不输出 CSV
# ============================================================
def main():
    # 图 1：kappa vs H
    df_H = collect_kappa_vs_H()
    plot_kappa_vs_H(df_H)

    # 图 2：kappa vs T
    df_T = collect_kappa_vs_T()
    plot_kappa_vs_T(df_T)

    print("All done.")
    print(f"Saved figure: {OUT_FIG_H}")
    print(f"Saved figure: {OUT_FIG_T}")


if __name__ == "__main__":
    main()