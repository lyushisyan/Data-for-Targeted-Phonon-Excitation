import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================
# 0. 全局风格设置 (Arial, 无加粗, 刻度朝内)
# ==========================================
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
    'figure.dpi': 300
})

COLORS_E = ['#104E8B', '#D9534F', '#20B2AA']
MARKERS_E = ['o', 's', 'D']
COLORS_MAP = {'low': '#D9534F', 'high': '#104E8B'}
MARKERS_MAP = {'low': 's', 'high': '^'}
LEGEND_LABELS = [
    r'$\Delta E_1$',
    r'$\Delta E_2$',
    r'$\Delta E_3$'
]

# ==========================================
# 1. 配置参数
# ==========================================
materials = [
    {
        'name': 'Ge',
        'bulk_dir': 'Germanium/Ge_kappa_bulk', 
        'film_root': 'Germanium/Results_cross_Lx1000_target_300K',
        'target_fn': 'result_S1_Lx1000.txt',
        'freq_lim': (0, 8), 
        'data_range': (0.6, 7.6), 
        'size_freqs': [0.8, 5.6], 
        'size_list': [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000],
    },
    {
        'name': 'Si',
        'bulk_dir': 'Silicon/Si_kappa_bulk',
        'film_root': 'Silicon/Results_cross_Lx1000_target_300K',
        'target_fn': 'result_S1_Lx1000.txt',
        'freq_lim': (0, 14),
        'data_range': (0.8, 13.6),
        'size_freqs': [1.2, 10.0],
        'size_list': [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000],
    },
    {
        'name': 'SiC',
        'bulk_dir': '3C-SiC/SiC_kappa_bulk',
        'film_root': '3C-SiC/Results_cross_Lx1000_target_300K',
        'target_fn': 'result_S1_Lx1000.txt',
        'freq_lim': (0, 28),
        'data_range': (2.0, 27.0),
        'size_freqs': [2.0, 15.0],
        'size_list': [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000],
    }
]

HIGH_TEMP_ROOTS = {"Ge": "Ge-1000K", "Si": "Si-1000K", "SiC": "3C-SiC-1000K"}
temp_list = {100: '100K', 200: '200K', 300: '300K', 400: '400K', 500: '500K', 600: '600K', 1000: '1000K', 2000: '2000K'}

# MFP 加载
mfp_csv = "MFP/mfp_summary.csv"
mfp_info = {}
NAME2CSV = {"Ge": "Ge", "Si": "Si", "SiC": "SiC"}
if os.path.exists(mfp_csv):
    try:
        df_mfp = pd.read_csv(mfp_csv)
        for _, r in df_mfp.iterrows():
            key = str(r["Material"]).strip()
            mfp_info[key] = {"avg": float(r["avg_MFP_nm"]), "max": float(r["max_MFP_nm"])}
    except: pass

# ==========================================
# 2. 绘制图 1: Frequency Dependence (2x3)
# ==========================================
fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8.5))
for col, mat in enumerate(materials):
    axes1[0, col].set_title(f"{mat['name']}", fontweight='normal', fontsize=20, y=1.02)
    
    # Bulk Data
    try:
        df_bulk = pd.read_csv(os.path.join(mat['bulk_dir'], 'Kappa_Excitation_300K.csv'))
        df_plot = df_bulk[(df_bulk['Frequency_THz'] >= mat['data_range'][0]) & (df_bulk['Frequency_THz'] <= mat['data_range'][1])]
        for i, E_col in enumerate(['Ratio_E1', 'Ratio_E2', 'Ratio_E3']):
            axes1[0, col].plot(df_plot['Frequency_THz'], (df_plot[E_col]-1)*100, color=COLORS_E[i], 
                                marker=MARKERS_E[i], markerfacecolor='none', linestyle='None', label=LEGEND_LABELS[i])
    except: pass
    
    # Film Data
    try:
        base_p = os.path.join(mat['film_root'], 'Baseline', mat['target_fn'])
        if not os.path.exists(base_p): base_p = base_p.replace('Baseline', 'Bseline')
        k_base = pd.read_csv(base_p)['Kappa_Sim'].iloc[0]
        for i, E in enumerate(['E1', 'E2', 'E3']):
            f_dir = os.path.join(mat['film_root'], E); pts = []
            if os.path.exists(f_dir):
                for sub in [d for d in os.listdir(f_dir) if d.startswith('Freq_')]:
                    fp = os.path.join(f_dir, sub, mat['target_fn'])
                    if os.path.exists(fp):
                        f_val = float(sub.split('_')[1]); rel = (pd.read_csv(fp)['Kappa_Sim'].iloc[0]/k_base)-1
                        if mat['data_range'][0] <= f_val <= mat['data_range'][1]: pts.append((f_val, rel * 100))
            if pts:
                pts.sort(); freqs, rels = zip(*pts)
                axes1[1, col].plot(freqs, rels, color=COLORS_E[i], marker=MARKERS_E[i], 
                                    markerfacecolor='none', linestyle='None', label=LEGEND_LABELS[i])
    except: pass

    for row in range(2):
        ax = axes1[row, col]; ax.set_xlim(mat['freq_lim']); ax.set_ylim(-43, 43) 
        ax.axhline(0, color='black', alpha=0.3, linewidth=1)
        
        # Legend 只在 (a) 标注
        if row == 0 and col == 0:
            h, l = ax.get_legend_handles_labels()
            if h: ax.legend(loc='lower left', frameon=False, fontsize=14)
        
        ax.text(0.04, 0.94, f"({chr(97 + row*3 + col)})", transform=ax.transAxes, 
                fontsize=18, fontweight='normal', va='top', ha='left')
        ax.text(0.96, 0.94, f"{'Bulk' if row == 0 else 'Film 100nm'}", transform=ax.transAxes, 
                fontsize=16, fontweight='normal', va='top', ha='right')
        
        if row == 0: ax.tick_params(labelbottom=False)
        else: ax.set_xlabel(r'$\omega_{\mathrm{t}}/2\pi$ (THz)')
        if col > 0: ax.tick_params(labelleft=False)
        else: ax.set_ylabel(r'$(\kappa - \kappa_0) / \kappa_0$ (%)')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig('Frequency_Dependence.png', bbox_inches='tight')
fig1.savefig('Frequency_Dependence.pdf', bbox_inches='tight')

# ==========================================
# 3. 绘制图 2: Scaling Dependence (2x3) 
# ==========================================
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 9))
for col, mat in enumerate(materials):
    f_low, f_high = mat['size_freqs']
    mat_prefix = mat['film_root'].split('/')[0]
    axes2[0, col].set_title(f"{mat['name']}", fontweight='normal', fontsize=20, y=1.02)

    # --- Size Effect (第一行) ---
    try:
        df_bulk_ref = pd.read_csv(os.path.join(mat['bulk_dir'], 'Kappa_Excitation_300K.csv'))
        for f, k_key in zip([f_low, f_high], ['low', 'high']):
            x_sz, y_sz = [], []
            for sz in mat['size_list']:
                root = os.path.join(mat_prefix, f"Results_cross_Lx{int(sz)}_target_300K")
                fn = f"result_S1_Lx{int(sz)}.txt"
                bp = os.path.join(root, 'Baseline', fn); bp = bp.replace('Baseline', 'Bseline') if not os.path.exists(bp) else bp
                fp = os.path.join(root, 'E3', f'Freq_{f}', fn)
                if os.path.exists(bp) and os.path.exists(fp):
                    x_sz.append(sz / 10.0)
                    ratio = (pd.read_csv(fp)['Kappa_Sim'].iloc[0]/pd.read_csv(bp)['Kappa_Sim'].iloc[0])-1
                    y_sz.append(ratio * 100)
            if x_sz:
                axes2[0, col].plot(x_sz, y_sz, color=COLORS_MAP[k_key], marker=MARKERS_MAP[k_key], markerfacecolor='none', label=f'{f} THz')
                idx = (df_bulk_ref['Frequency_THz'] - f).abs().idxmin()
                ref_y = (df_bulk_ref.loc[idx, 'Ratio_E3']-1)*100
                axes2[0, col].axhline(ref_y, color=COLORS_MAP[k_key], linestyle='--', alpha=0.5)
        
        # MFP 标注
        csv_key = NAME2CSV.get(mat["name"], mat["name"])
        if csv_key in mfp_info:
            m_avg, m_max = mfp_info[csv_key]["avg"], mfp_info[csv_key]["max"]
            for m_val, lab in zip([m_avg, m_max], [r'$\langle \ell \rangle$', r'$\ell_{\mathrm{max}}$']):
                axes2[0, col].axvline(x=m_val, ymin=0.94, ymax=1.0, color='gray', linewidth=1.5, alpha=0.8)
                axes2[0, col].text(m_val, 0.92, lab, transform=axes2[0, col].get_xaxis_transform(), 
                                    va='top', ha='center', fontsize=18, color='black', fontweight='normal')
        axes2[0, col].set_xscale('log')
    except: pass

    # --- Temp Effect (第二行) ---
    try:
        for f, k_key in zip([f_low, f_high], ['low', 'high']):
            x_t, y_t, x_bt, y_bt = [], [], [], [] 
            for T, sfx in temp_list.items():
                current_root = HIGH_TEMP_ROOTS.get(mat['name']) if T >= 1000 else mat_prefix
                folder = os.path.join(current_root, f"Results_cross_Lx1000_target_{sfx}")
                bp = os.path.join(folder, 'Baseline', mat['target_fn']); bp = bp.replace('Baseline', 'Bseline') if not os.path.exists(bp) else bp
                fp = os.path.join(folder, 'E3', f'Freq_{f}', mat['target_fn'])
                
                if os.path.exists(bp) and os.path.exists(fp):
                    x_t.append(T)
                    ratio = (pd.read_csv(fp)['Kappa_Sim'].iloc[0]/pd.read_csv(bp)['Kappa_Sim'].iloc[0])-1
                    y_t.append(ratio * 100)
                
                if T >= 1000:
                    bulk_root = os.path.join(HIGH_TEMP_ROOTS.get(mat['name']), os.path.basename(mat['bulk_dir']))
                else:
                    bulk_root = mat['bulk_dir']
                
                bulk_f = os.path.join(bulk_root, f"Kappa_Excitation_{sfx}.csv")
                
                if os.path.exists(bulk_f):
                    df_b = pd.read_csv(bulk_f)
                    idx = (df_b['Frequency_THz'] - f).abs().idxmin()
                    x_bt.append(T)
                    y_bt.append((df_b.loc[idx, 'Ratio_E3']-1)*100)
            
            axes2[1, col].plot(x_t, y_t, color=COLORS_MAP[k_key], marker=MARKERS_MAP[k_key], markerfacecolor='none', label=f'{f} THz')
            axes2[1, col].plot(x_bt, y_bt, color=COLORS_MAP[k_key], linestyle='--', alpha=0.5)
            
            axes2[1, col].set_xscale('log')
        
    except Exception as e: 
        print(f"Error Temp {mat['name']}: {e}")

    for row in range(2):
        ax = axes2[row, col]; ax.set_ylim(-43, 43) 
        ax.axhline(0, color='black', alpha=0.3, linewidth=1)
        if row == 1:
            bulk_h = Line2D([0], [0], color='k', linestyle='--', alpha=0.5, label='Bulk (excited)')
            h, l = ax.get_legend_handles_labels()
            if h: ax.legend(h + [bulk_h], l + ['Bulk (excited)'], loc='lower right', frameon=False, fontsize=14)
        
        ax.text(0.04, 0.94, f"({chr(97 + row*3 + col)})", transform=ax.transAxes, 
                fontsize=18, fontweight='normal', va='top', ha='left')
        
        # --- 关键修改：将 L 修改为 H ---
        ax.set_xlabel(r'$H$ (nm)' if row == 0 else r'$T$ (K)')
        
        if col > 0: ax.tick_params(labelleft=False)
        else: ax.set_ylabel(r'$(\kappa - \kappa_0) / \kappa_0$ (%)')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig('Scaling_Dependence.png', bbox_inches='tight')
fig2.savefig('Scaling_Dependence.pdf', bbox_inches='tight')
