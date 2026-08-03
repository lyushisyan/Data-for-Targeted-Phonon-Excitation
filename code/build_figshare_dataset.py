#!/usr/bin/env python3
"""Build a compact, review-ready Figshare supporting-data deposit.

Run with the project conda Python:
    /opt/miniconda3/bin/python build_figshare_dataset.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "Figshare_Dataset"
SIZES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
         20000, 50000, 100000, 200000]
TEMPS = [100, 200, 300, 400, 500, 600, 1000, 2000]
MATERIALS = [
    dict(name="Ge", root="Germanium", high="Ge-1000K", bulk="Ge_kappa_bulk",
         debye=7.5, freqs=(0.8, 5.6), sheng="Ge_from_shengbte"),
    dict(name="Si", root="Silicon", high="Si-1000K", bulk="Si_kappa_bulk",
         debye=13.0, freqs=(1.2, 10.0), sheng="Si_from_shengbte"),
    dict(name="SiC", root="3C-SiC", high="3C-SiC-1000K", bulk="SiC_kappa_bulk",
         debye=20.0, freqs=(2.0, 15.0), sheng="SiC_from_shengbte"),
]


class Builder:
    def __init__(self, out: Path):
        self.out = out
        self.data = out / "data"
        self.code = out / "code"
        self.sources: set[Path] = set()

    def source(self, path: str | Path) -> Path:
        p = ROOT / path
        if not p.is_file():
            raise FileNotFoundError(p)
        self.sources.add(p)
        return p

    @staticmethod
    def kappa(path: Path) -> float:
        return float(pd.read_csv(path)["Kappa_Sim"].iloc[0])

    def baseline(self, folder: Path, size: int) -> Path:
        fn = f"result_S1_Lx{size}.txt"
        for spelling in ("Baseline", "Bseline"):
            p = folder / spelling / fn
            if p.is_file():
                return self.source(p.relative_to(ROOT))
        raise FileNotFoundError(f"No baseline in {folder}")

    def freq_file(self, folder: Path, energy: str, freq: float, size: int) -> Path:
        e_root = folder / energy
        candidates = []
        for d in e_root.glob("Freq_*"):
            try:
                candidates.append((abs(float(d.name.split("_", 1)[1]) - freq), d))
            except ValueError:
                pass
        if not candidates or min(candidates)[0] >= 1e-6:
            raise FileNotFoundError(f"Frequency {freq} not found in {e_root}")
        p = min(candidates)[1] / f"result_S1_Lx{size}.txt"
        return self.source(p.relative_to(ROOT))

    def bulk_file(self, mat: dict, temp: int) -> Path:
        base = mat["high"] if temp >= 1000 else mat["root"]
        p = Path(base) / mat["bulk"] / f"Kappa_Excitation_{temp}K.csv"
        return self.source(p)

    def result_root(self, mat: dict, temp: int, size: int = 1000) -> Path:
        base = mat["high"] if temp >= 1000 else mat["root"]
        return ROOT / base / f"Results_cross_Lx{size}_target_{temp}K"

    def build_heatmap(self):
        mfp_path = self.source("MFP/mfp_summary.csv")
        mfp = pd.read_csv(mfp_path).set_index("Material")
        rows = []
        for mat in MATERIALS:
            available = []
            for size in SIZES:
                folder = ROOT / mat["root"] / f"Results_cross_Lx{size}_target_300K"
                k0 = self.kappa(self.baseline(folder, size))
                for d in sorted((folder / "E2").glob("Freq_*"),
                                key=lambda x: float(x.name.split("_", 1)[1])):
                    freq = float(d.name.split("_", 1)[1])
                    p = d / f"result_S1_Lx{size}.txt"
                    if p.is_file():
                        available.append((size, freq, k0, self.kappa(self.source(p.relative_to(ROOT)))))
            freq_set = sorted({x[1] for x in available})
            selected = freq_set[::2] if mat["name"] == "SiC" else freq_set[:-1]
            selected_set = set(selected)
            avg_mfp = float(mfp.loc[mat["name"], "avg_MFP_nm"])
            for size, freq, k0, ke in available:
                if freq not in selected_set:
                    continue
                raw_ratio = ke / k0
                overridden = mat["name"] == "SiC" and 18 <= freq <= 24
                plotted_ratio = 1.0 if overridden else raw_ratio
                h_nm = size / 10.0
                rows.append(dict(Material=mat["name"], Size_Lx_A=size, H_nm=h_nm,
                                 Average_MFP_nm=avg_mfp, Knudsen_number=avg_mfp / h_nm,
                                 Frequency_THz=freq, Debye_frequency_THz=mat["debye"],
                                 Normalized_frequency=freq / mat["debye"],
                                 Kappa_unexcited_W_mK=k0, Kappa_excited_W_mK=ke,
                                 Ratio_raw=raw_ratio, Ratio_plotted=plotted_ratio,
                                 Relative_change_plotted_percent=100 * (plotted_ratio - 1),
                                 Was_overridden_in_plot=overridden))
        pd.DataFrame(rows).to_csv(self.data / "fig_heatmap_materials.csv", index=False)

        omega = np.linspace(0, 1, 1500)
        targets = np.linspace(0, 1, 260)
        kns = np.logspace(-2, 2, 240)
        q = (2 / np.pi) * np.arcsin(omega)
        velocity = np.maximum(np.sqrt(1 - omega**2), 1e-4)
        density = q**2 / velocity
        heat_capacity = np.exp(-1.2 * omega)
        ell = velocity * (omega + 0.05) ** -1.4
        weight = density * heat_capacity * velocity * ell
        weight /= np.trapezoid(weight, omega)
        ell_norm = ell / np.trapezoid(weight * ell, omega)
        og, tg = omega[:, None], targets[None, :]
        alpha = 0.75 * np.exp(-2.8 * tg) * np.exp(-((og - tg) ** 2) / (2 * 0.055**2))
        beta = 0.13 * (1 + 4.5 * tg**2) * np.exp(-((og - tg) ** 2) / (2 * 0.16**2))
        out = []
        for kn in kns:
            surface = kn * ell_norm[:, None]
            response = (1 + alpha) * (1 + surface) / (1 + beta + surface) - 1
            eta = 100 * np.trapezoid(weight[:, None] * response, omega, axis=0)
            out.append(pd.DataFrame({"Normalized_target_frequency": targets,
                                     "Knudsen_number": kn,
                                     "Relative_change_percent": eta}))
        pd.concat(out, ignore_index=True).to_csv(self.data / "fig_heatmap_minimodel.csv", index=False)

    def build_frequency_and_scaling(self):
        freq_rows, h_rows, t_rows = [], [], []
        mfp = pd.read_csv(self.source("MFP/mfp_summary.csv")).set_index("Material")
        for mat in MATERIALS:
            bulk300 = pd.read_csv(self.bulk_file(mat, 300))
            low, high = mat["freqs"]
            data_range = {"Ge": (0.6, 7.6), "Si": (0.8, 13.6), "SiC": (2, 27)}[mat["name"]]
            for _, r in bulk300.iterrows():
                f = float(r["Frequency_THz"])
                if data_range[0] <= f <= data_range[1]:
                    for energy in ("E1", "E2", "E3"):
                        freq_rows.append(dict(Material=mat["name"], Geometry="Bulk", H_nm=np.nan,
                                              Temperature_K=300, Energy_level=energy,
                                              Frequency_THz=f, Kappa_unexcited_W_mK=r["Original"],
                                              Kappa_excited_W_mK=r[energy], Ratio=r[f"Ratio_{energy}"],
                                              Relative_change_percent=100 * (r[f"Ratio_{energy}"] - 1)))
            folder = self.result_root(mat, 300)
            k0 = self.kappa(self.baseline(folder, 1000))
            for energy in ("E1", "E2", "E3"):
                for d in sorted((folder / energy).glob("Freq_*"),
                                key=lambda x: float(x.name.split("_", 1)[1])):
                    f = float(d.name.split("_", 1)[1])
                    if data_range[0] <= f <= data_range[1]:
                        p = d / "result_S1_Lx1000.txt"
                        if p.is_file():
                            ke = self.kappa(self.source(p.relative_to(ROOT)))
                            freq_rows.append(dict(Material=mat["name"], Geometry="Film", H_nm=100,
                                                  Temperature_K=300, Energy_level=energy,
                                                  Frequency_THz=f, Kappa_unexcited_W_mK=k0,
                                                  Kappa_excited_W_mK=ke, Ratio=ke/k0,
                                                  Relative_change_percent=100*(ke/k0-1)))

            for size in SIZES:
                folder = ROOT / mat["root"] / f"Results_cross_Lx{size}_target_300K"
                base = self.kappa(self.baseline(folder, size))
                for f in (low, high):
                    excited = self.kappa(self.freq_file(folder, "E3", f, size))
                    idx = (bulk300["Frequency_THz"] - f).abs().idxmin()
                    h_rows.append(dict(Material=mat["name"], H_nm=size/10, Size_Lx_A=size,
                                       Frequency_THz=f, Energy_level="E3",
                                       Kappa_unexcited_W_mK=base, Kappa_excited_W_mK=excited,
                                       Relative_change_percent=100*(excited/base-1),
                                       Bulk_reference_change_percent=100*(bulk300.loc[idx,"Ratio_E3"]-1),
                                       Average_MFP_nm=mfp.loc[mat["name"],"avg_MFP_nm"],
                                       Maximum_MFP_nm=mfp.loc[mat["name"],"max_MFP_nm"]))
            for temp in TEMPS:
                folder = self.result_root(mat, temp)
                base = self.kappa(self.baseline(folder, 1000))
                bulk = pd.read_csv(self.bulk_file(mat, temp))
                for f in (low, high):
                    excited = self.kappa(self.freq_file(folder, "E3", f, 1000))
                    idx = (bulk["Frequency_THz"] - f).abs().idxmin()
                    t_rows.append(dict(Material=mat["name"], Temperature_K=temp, H_nm=100,
                                       Frequency_THz=f, Energy_level="E3",
                                       Film_Kappa_unexcited_W_mK=base,
                                       Film_Kappa_excited_W_mK=excited,
                                       Film_relative_change_percent=100*(excited/base-1),
                                       Bulk_Kappa_unexcited_W_mK=bulk.loc[idx,"Original"],
                                       Bulk_Kappa_excited_W_mK=bulk.loc[idx,"E3"],
                                       Bulk_relative_change_percent=100*(bulk.loc[idx,"Ratio_E3"]-1)))
        pd.DataFrame(freq_rows).to_csv(self.data / "fig_frequency_dependence.csv", index=False)
        pd.DataFrame(h_rows).to_csv(self.data / "fig_scaling_and_kappa_vs_H.csv", index=False)
        pd.DataFrame(t_rows).to_csv(self.data / "fig_scaling_and_kappa_vs_T.csv", index=False)

    def build_tau_cumulative(self):
        rows = []
        for mat in MATERIALS:
            low, high = mat["freqs"]
            for temp in (100, 300):
                folder = self.result_root(mat, temp)
                configs = [("Unexcited", np.nan, folder / "Baseline"),
                           ("Low frequency", low, folder / "E3" / f"Freq_{low}"),
                           ("High frequency", high, folder / "E3" / f"Freq_{high}")]
                for state, target, d in configs:
                    p = self.source((d / "spectral_ibz_S1_Lx1000.csv").relative_to(ROOT))
                    df = pd.read_csv(p).sort_values("Frequency_THz")
                    df = df[df["Tau_Sim_ps"] >= 1e-5].copy()
                    df["Kappa_Sim_W_mK"] *= 2.0
                    df["Cumulative_Kappa_W_mK"] = df["Kappa_Sim_W_mK"].cumsum()
                    df.insert(0, "Target_frequency_THz", target)
                    df.insert(0, "State", state)
                    df.insert(0, "Temperature_K", temp)
                    df.insert(0, "H_nm", 100)
                    df.insert(0, "Material", mat["name"])
                    rows.append(df)
        pd.concat(rows, ignore_index=True).to_csv(
            self.data / "fig_tau_and_cumulative_kappa.csv.gz", index=False, compression="gzip")

    def build_phonon_properties(self):
        rows = []
        for mat in MATERIALS:
            d = Path(mat["root"]) / mat["sheng"]
            omega = np.loadtxt(self.source(d / "BTE.omega"))
            qpts = np.loadtxt(self.source(d / "BTE.qpoints"))
            velocity = np.loadtxt(self.source(d / "BTE.v"))
            gamma_data = np.loadtxt(self.source(d / "BTE.w_T300K_original"))
            n_q, n_br = omega.shape
            omega_flat = omega.flatten(order="F") / (2*np.pi)
            weights = np.tile(qpts[:, 3], (n_br, 1)).flatten(order="C")
            vmag = np.sqrt(np.sum(velocity**2, axis=1)).flatten()
            gamma = gamma_data[:, 1].flatten()
            tau = np.divide(1.0, gamma, out=np.full_like(gamma, np.nan), where=gamma != 0)
            rows.append(pd.DataFrame({"Material": mat["name"],
                                      "Mode_index": np.arange(len(omega_flat)),
                                      "Frequency_THz": omega_flat,
                                      "Qpoint_weight": weights,
                                      "Group_velocity_km_s": vmag,
                                      "Scattering_rate_ps_inverse": gamma,
                                      "Relaxation_time_ps": tau}))
        pd.concat(rows, ignore_index=True).to_csv(self.data / "fig_phonon_properties.csv", index=False)

    def copy_code_and_write_docs(self):
        scripts = ["plot_heatmap_minimodel_2x2.py", "plot_kappa_ratio_Si_Ge_SiC.py",
                   "plot_kappa_vs_T_and_H.py", "plot_tau.py", "plot_phonon.py", "minimodel.py"]
        for name in scripts:
            shutil.copy2(self.source(name), self.code / name)
        shutil.copy2(Path(__file__), self.code / Path(__file__).name)

        (self.out / "README.md").write_text("""# Supporting data for Track Phonon 3D

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
""", encoding="utf-8")

    def clean_metadata(self):
        for p in self.out.rglob("._*"):
            p.unlink()

    def run(self):
        self.data.mkdir(parents=True)
        self.code.mkdir()
        self.build_heatmap()
        self.build_frequency_and_scaling()
        self.build_tau_cumulative()
        self.build_phonon_properties()
        self.copy_code_and_write_docs()
        self.clean_metadata()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        if not args.force:
            raise SystemExit(f"Output already exists: {out}. Use --force to replace it.")
        def ignore_vanished(_func, _path, excinfo):
            if not isinstance(excinfo, FileNotFoundError):
                raise excinfo
        shutil.rmtree(out, onexc=ignore_vanished)
    Builder(out).run()
    print(out)


if __name__ == "__main__":
    main()
