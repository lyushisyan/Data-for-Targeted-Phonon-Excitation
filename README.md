# Supporting data for Track Phonon 3D

This deposit contains the numerical supporting data used by the analysis and plotting scripts in the associated manuscript. It intentionally excludes the manuscript and figure image files, as requested by the Nature Portfolio submission system.

## Contents

| Figure | Data file |
|---|---|
| `Heatmap_Materials_MiniModel_2x2.png/.pdf` | `data/fig_heatmap_materials.csv`; `data/fig_heatmap_minimodel.csv` |
| `Frequency_Dependence.png/.pdf` | `data/fig_frequency_dependence.csv` |
| `Scaling_Dependence.png/.pdf` and `Kappa_vs_H_Scale.png` | `data/fig_scaling_and_kappa_vs_H.csv` |
| `Scaling_Dependence.png/.pdf` and `Kappa_vs_T_Bulk_Film100nm.png` | `data/fig_scaling_and_kappa_vs_T.csv` |
| `Phonon_Analysis_Internal_Labels.png` | `data/fig_tau_and_cumulative_kappa.csv.gz` |
| `Phonon_BTE_Tau_Analysis.png` | `data/fig_phonon_properties.csv` |

`code/` contains the analysis/plotting scripts and the dataset-building script. The gzip file is a compressed UTF-8 CSV; all other data files are plain UTF-8 CSV.

## Notes

- Film thickness is `H = Lx / 10` nm.
- The plotted spectral thermal-conductivity contribution is multiplied by 2 before cumulative summation, matching `plot_tau.py`.
- For the SiC heatmap, the plotting script sets ratios from 18 to 24 THz to 1. The table retains both raw and plotted values and marks these rows in `Was_overridden_in_plot`.
- No manuscript, figure image, or sensitive human data is included.
