import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 0. 全局风格设置
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 
    'font.size': 14,
    'axes.labelsize': 22,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 11,
    'axes.linewidth': 2.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 2.2,
    'legend.frameon': False,
    'figure.dpi': 300
})

C_BASE = '#000000'
C_E1_2 = '#FF0000'
C_E10  = '#0070FF'

# ==========================================
# 1. 配置参数与数据路径
# ==========================================
material_configs = [
    {"name": "Ge", "base_dir": "Germanium", "freqs": ("0.8", "5.6")},
    {"name": "Si", "base_dir": "Silicon", "freqs": ("1.2", "10.0")},
    {"name": "3C-SiC", "base_dir": "3C-SiC", "freqs": ("2.0", "15.0")},
]

temp_configs = ["100K", "300K"]
lx = 1000

# 预扫描全局最大热导率以统一右轴范围
global_max_k = 0
for mat in material_configs:
    low_freq, high_freq = mat["freqs"]
    plot_configs = [
        ("", "Baseline", "Unexcited", C_BASE),
        ("E3", f"Freq_{low_freq}", "Low freq.", C_E1_2),
        ("E3", f"Freq_{high_freq}", "High freq.", C_E10),
    ]

    for temp in temp_configs:
        sub_dir = f"Results_cross_Lx{lx}_target_{temp}"
        file_name = f"spectral_ibz_S1_Lx{lx}.csv"
        for exp, freq_f, _, _ in plot_configs:
            p = os.path.join(mat["base_dir"], sub_dir, exp, freq_f, file_name)
            if os.path.exists(p):
                df_tmp = pd.read_csv(p)
                max_val = df_tmp['Kappa_Sim_W_mK'].sum() * 2.0
                if max_val > global_max_k: global_max_k = max_val

# ==========================================
# 2. 正式绘图 (2x3 布局)
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
sample_rate = 0.1 # 散点 10% 采样

for row_idx, temp in enumerate(temp_configs):
    for col_idx, mat in enumerate(material_configs):
        ax_l = axes[row_idx, col_idx]
        ax_r = ax_l.twinx() # 开启右轴

        low_freq, high_freq = mat["freqs"]
        plot_configs = [
            ("", "Baseline", "Unexcited", C_BASE),
            ("E3", f"Freq_{low_freq}", "Low freq.", C_E1_2),
            ("E3", f"Freq_{high_freq}", "High freq.", C_E10),
        ]

        sub_dir = f"Results_cross_Lx{lx}_target_{temp}"
        file_name = f"spectral_ibz_S1_Lx{lx}.csv"

        for exp, freq_f, label, color in plot_configs:
            path = os.path.join(mat["base_dir"], sub_dir, exp, freq_f, file_name)
            if os.path.exists(path):
                # 读取并排序
                df = pd.read_csv(path).sort_values('Frequency_THz')
                df = df[df['Tau_Sim_ps'] >= 1e-5] # 过滤过小值
                df['Kappa_Sim_W_mK'] *= 2.0      # 热导率乘2

                # A. 绘制左轴散点 (Relaxation Time)
                df_s = df.sample(frac=sample_rate, random_state=42)
                ax_l.scatter(df_s['Frequency_THz'], df_s['Tau_Sim_ps'],
                             s=5, color=color, alpha=0.5, edgecolors='none')

                # B. 绘制右轴累计热导率 (Cumulative Kappa)
                cum_k = df['Kappa_Sim_W_mK'].cumsum()
                ls = '--' if label == "Unexcited" else '-'
                ax_r.plot(df['Frequency_THz'], cum_k, color=color, linestyle=ls, label=label)

        # --- 坐标轴范围与刻度设置 ---
        ax_l.set_yscale('log')
        ax_l.set_ylim(0.5, 1e4) # 左轴统一
        ax_r.set_ylim(0, global_max_k * 1.2) # 右轴统一

        # 内部标注：面板编号 (a-f)
        panel_idx = row_idx * len(material_configs) + col_idx
        ax_l.text(0.04, 0.94, f"({chr(97 + panel_idx)})",
                  transform=ax_l.transAxes, fontsize=20, va='top')

        # 内部标注：【温度与厚度写在右上角】
        ax_l.text(0.96, 0.94, f"$H = 100$ nm\n$T = {temp[:-1]}$ K",
                  transform=ax_l.transAxes, fontsize=13, ha='right', va='top',
                  linespacing=1.35)

        if row_idx == 0:
            ax_l.set_title(mat["name"], fontsize=20, pad=10)
            ax_l.tick_params(labelbottom=False)
        else:
            ax_l.set_xlabel('Frequency $\omega/2\pi$ (THz)')

        if col_idx == 0:
            ax_l.set_ylabel(r'$\tau$ (ps)')
        else:
            ax_l.tick_params(labelleft=False)

        if col_idx == len(material_configs) - 1:
            ax_r.set_ylabel(r"$\kappa_{\mathrm{cum}}$ (W m$^{-1}$K$^{-1}$)")
        else:
            ax_r.tick_params(labelright=False)

        # 图例：放在第一个面板中
        if row_idx == 0 and col_idx == 0:
            ax_r.legend(loc='center right', bbox_to_anchor=(0.98, 0.62),
                        fontsize=13)

# ==========================================
# 3. 布局微调与保存
# ==========================================
plt.subplots_adjust(wspace=0.12, hspace=0.18, top=0.91, bottom=0.1, left=0.08, right=0.92)

output_name = 'Phonon_Analysis_Internal_Labels.png'
plt.savefig(output_name, bbox_inches='tight')
plt.close()

print(f"--- 绘图完成：{output_name} ---")
