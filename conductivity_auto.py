"""
Non-interactive conductivity calculator.

Reads simulation metadata (volume, timestep, temperature) from GK_info.txt
and Li/PF6 counts from in.lammps, fits diffusion coefficients from MSD files,
then reports conductivity via the Nernst–Einstein relation.
"""
from __future__ import annotations

import argparse
import re
from typing import Dict, Tuple

import numpy as np

from msd_diffusion import load_msd, default_window_ps

kB = 1.380649e-23  # J/K
e = 1.602176634e-19  # C


def parse_gk_info(path: str) -> Dict[str, float]:
    info = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "BOXVOLUME" in line:
                info["volume_A3"] = float(line.split("=")[-1].strip())
            elif "TIMESTEP" in line:
                info["dt_fs"] = float(line.split("=")[-1].strip())
            elif "TEMPERATURE" in line:
                info["temperature_K"] = float(line.split("=")[-1].strip())
    missing = {k for k in ("volume_A3", "dt_fs", "temperature_K") if k not in info}
    if missing:
        raise ValueError(f"Missing keys in {path}: {', '.join(sorted(missing))}")
    return info


def parse_counts_from_in(path: str) -> Dict[str, int]:
    pattern = re.compile(r"group\s+(Li|PF6)\s+molecule\s+(\d+)\s*:\s*(\d+)")
    counts: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                name, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
                counts[name] = hi - lo + 1
    missing = {k for k in ("Li", "PF6") if k not in counts}
    if missing:
        raise ValueError(f"Could not find counts for: {', '.join(sorted(missing))}")
    return counts


def fit_diffusion(msd_path: str, dt_fs: float, window: Tuple[float, float] | None, mode: str = "total") -> Dict[str, float]:
    df = load_msd(msd_path, dt_fs=dt_fs)
    if window is None:
        t1_ps, t2_ps = default_window_ps(df, frac_last=0.4)
    else:
        t1_ps, t2_ps = window
    t_ps = df["t_ps"].values
    t_s = df["t_s"].values
    y = df["msd_total"].values if mode == "total" else df[f"msd_{mode}"]
    mask = (t_ps >= t1_ps) & (t_ps <= t2_ps)
    if mask.sum() < 2:
        raise ValueError(f"Window {t1_ps}-{t2_ps} ps has insufficient points in {msd_path}")
    slope, intercept = np.polyfit(t_s[mask], y[mask], 1)
    factor = 6 if mode == "total" else 2
    D_m2_s = (slope * 1e-20) / factor
    return {
        "slope_A2_per_s": float(slope),
        "intercept": float(intercept),
        "t1_ps": float(t1_ps),
        "t2_ps": float(t2_ps),
        "D_m2_s": float(D_m2_s),
    }


def sigma_ne(Ds, zs, Ns, V_m3: float, T_K: float) -> float:
    return sum((N / V_m3) * (z * e) ** 2 * D for D, z, N in zip(Ds, zs, Ns)) / (kB * T_K)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute conductivity directly from MSD files and metadata.")
    p.add_argument("--gk-info", default="GK_info.txt", help="Path to GK_info.txt with volume, timestep, temperature.")
    p.add_argument("--in-file", default="in.lammps", help="LAMMPS input file containing group counts for Li and PF6.")
    p.add_argument("--msd-li", default="msd_Li.out", help="MSD file for Li.")
    p.add_argument("--msd-pf6", default="msd_PF6.out", help="MSD file for PF6.")
    p.add_argument("--window", nargs=2, type=float, default=None, metavar=("t1_ps", "t2_ps"), help="Optional fit window in ps.")
    args = p.parse_args()

    info = parse_gk_info(args.gk_info)
    counts = parse_counts_from_in(args.in_file)

    window = tuple(args.window) if args.window else None
    res_li = fit_diffusion(args.msd_li, dt_fs=info["dt_fs"], window=window, mode="total")
    res_pf6 = fit_diffusion(args.msd_pf6, dt_fs=info["dt_fs"], window=window, mode="total")

    V_m3 = info["volume_A3"] * 1e-30
    D_list = [res_li["D_m2_s"], res_pf6["D_m2_s"]]
    z_list = [1.0, -1.0]
    N_list = [counts["Li"], counts["PF6"]]
    sigma_S_m = sigma_ne(D_list, z_list, N_list, V_m3, info["temperature_K"])

    print("=== Metadata ===")
    print(f"Volume: {info['volume_A3']:.6e} Å^3 ({V_m3:.6e} m^3)")
    print(f"Timestep: {info['dt_fs']} fs")
    print(f"Temperature: {info['temperature_K']} K")
    print(f"Counts: Li={N_list[0]}, PF6={N_list[1]}")
    print("\n=== Diffusion coefficients ===")
    print(f"Li  D = {res_li['D_m2_s']:.6e} m^2/s  (fit {res_li['t1_ps']:.2f}-{res_li['t2_ps']:.2f} ps)")
    print(f"PF6 D = {res_pf6['D_m2_s']:.6e} m^2/s  (fit {res_pf6['t1_ps']:.2f}-{res_pf6['t2_ps']:.2f} ps)")
    print("\n=== Conductivity (Nernst–Einstein) ===")
    print(f"sigma = {sigma_S_m:.6e} S/m  = {sigma_S_m*1e4:.6e} mS/cm")


if __name__ == "__main__":
    main()
