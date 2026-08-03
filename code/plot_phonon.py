import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. 全局样式设置
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 
    'font.size': 14,                
    'axes.labelsize': 24,           
    'xtick.labelsize': 14,          
    'ytick.labelsize': 14,
    'legend.fontsize': 16,          
    'axes.linewidth': 1.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'lines.linewidth': 2,
    'legend.frameon': False,
    'figure.dpi': 300               
})

materials_config = [
    {'name': 'Ge', 'folder': 'Germanium/Ge_from_shengbte', 'color': '#104E8B'},
    {'name': 'Si', 'folder': 'Silicon/Si_from_shengbte', 'color': '#D9534F'},
    {'name': 'SiC', 'folder': '3C-SiC/SiC_from_shengbte', 'color': '#20B2AA'}
]

# 设定统一的 Y 轴范围
LIMITS = {
    'dos': (0, 0.6),        
    'vg': (0, 14),          
    'tau': (1e-1, 1e4)      # 弛豫时间 (ps) 的对数范围，通常在 0.1 到 1000+ ps 之间
}

fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex='col', sharey='row')

for col, mat in enumerate(materials_config):
    try:
        # 1. 读取频率并按列拉平 (order='F')
        omega_raw = np.loadtxt(os.path.join(mat['folder'], 'BTE.omega'))
        n_qpts, n_branches = omega_raw.shape
        omega = omega_raw.flatten(order='F') / (2 * np.pi) # 转换为 THz
        
        # 2. 读取 q 点权重并对齐
        qpts_data = np.loadtxt(os.path.join(mat['folder'], 'BTE.qpoints'))
        weights_raw = qpts_data[:, 3]
        weights = np.tile(weights_raw, (n_branches, 1)).flatten(order='C')

        # 3. 读取群速度并计算模长 (假设原始单位为 km/s 或已处理)
        v_raw = np.loadtxt(os.path.join(mat['folder'], 'BTE.v'))
        v_mag = np.sqrt(np.sum(v_raw**2, axis=1)).flatten()
        
        # 4. 读取散射率并计算弛豫时间 tau = 1/gamma
        gamma_raw = np.loadtxt(os.path.join(mat['folder'], 'BTE.w_T300K_original'))
        gamma = gamma_raw[:, 1].flatten()
        
        # 避免除以 0 的错误 (如果有极小的散射率)
        gamma = np.where(gamma == 0, np.nan, gamma)
        tau = 1.0 / gamma 

        # --- Row 0: Weighted DOS ---
        axes[0, col].hist(omega, bins=100, color=mat['color'], alpha=0.5, 
                          density=True, weights=weights)
        axes[0, col].set_ylim(LIMITS['dos'])
        axes[0, col].set_title(f"{mat['name']}", fontweight='bold', fontsize=22)
        if col == 0: axes[0, col].set_ylabel('DOS (a.u.)')

        # --- Row 1: Group Velocity (vg) ---
        axes[1, col].scatter(omega, v_mag, color=mat['color'], s=10, alpha=0.5, edgecolors='none')
        axes[1, col].set_ylim(LIMITS['vg'])
        if col == 0: axes[1, col].set_ylabel(r'$v_g$ (km/s)')

        # --- Row 2: Relaxation Time (tau) ---
        axes[2, col].scatter(omega, tau, color=mat['color'], s=10, alpha=0.5, edgecolors='none')
        axes[2, col].set_yscale('log')
        axes[2, col].set_ylim(LIMITS['tau'])
        axes[2, col].set_xlabel(r'Frequency $\omega/2\pi$ (THz)')
        if col == 0: axes[2, col].set_ylabel(r'$\tau$ (ps)')

    except Exception as e:
        print(f"Error processing {mat['name']}: {e}")

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.15)

plt.savefig('Phonon_BTE_Tau_Analysis.png', bbox_inches='tight')
print("Plotting complete. Check 'Phonon_BTE_Tau_Analysis.png'.")