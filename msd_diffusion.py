# msd_diffusion.py
# 读取同目录 MSD.txt，画图 -> 交互选择口径与拟合区间 -> 计算扩散系数 D
# 兼容列格式：
# 1) step  msd_x  msd_y  msd_z  msd_total
# 2) step  ...  msd_total（最后一列总是 total）

import argparse
import os
import sys
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- IO & PREP -----------------------------
def load_msd(path: str, dt_fs: float = 1.0) -> pd.DataFrame:
    """
    读取 MSD 文件。默认单位：real（长度 Å，时间 fs）。
    支持：step + [msd_x, msd_y, msd_z] + msd_total；或 step + msd_total。
    返回列：step, t_fs, t_ps, t_ns, t_s, msd_total, (可选) msd_x, msd_y, msd_z
    """
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    ncol = arr.shape[1]
    if ncol < 2:
        raise ValueError("MSD.txt 至少需要两列：step 和 msd_total")

    step = arr[:, 0]
    msd_total = arr[:, -1]
    df = pd.DataFrame({"step": step, "msd_total": msd_total})

    # 如果有 x/y/z 列（常见为 5 列）
    if ncol >= 5:
        df["msd_x"] = arr[:, 1]
        df["msd_y"] = arr[:, 2]
        df["msd_z"] = arr[:, 3]
    elif ncol == 4:
        # 少数导出可能只有 x, y, total（此时 z 缺失也可运作）
        df["msd_x"] = arr[:, 1]
        df["msd_y"] = arr[:, 2]

    # 时间换算
    df["t_fs"] = df["step"] * dt_fs
    df["t_ps"] = df["t_fs"] * 1e-3
    df["t_ns"] = df["t_fs"] * 1e-6
    df["t_s"] = df["t_fs"] * 1e-15
    return df


def ensure_outdir(path: str) -> str:
    outdir = os.path.abspath(path)
    os.makedirs(outdir, exist_ok=True)
    return outdir


# ----------------------------- PLOTTING -----------------------------
def plot_overview(df: pd.DataFrame, outdir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    画总 MSD 与分量 MSD，返回图片路径。
    """
    total_png = os.path.join(outdir, "MSD_total.png")
    plt.figure()
    plt.plot(df["t_ns"], df["msd_total"])
    plt.xlabel("Time (ns)")
    plt.ylabel("MSD_total (Å$^2$)")
    plt.title("Total MSD vs Time")
    plt.tight_layout()
    plt.savefig(total_png, dpi=160)
    plt.close()

    comp_png = None
    if all(c in df.columns for c in ["msd_x", "msd_y", "msd_z"]):
        comp_png = os.path.join(outdir, "MSD_components.png")
        plt.figure()
        plt.plot(df["t_ns"], df["msd_x"], label="MSD_x")
        plt.plot(df["t_ns"], df["msd_y"], label="MSD_y")
        plt.plot(df["t_ns"], df["msd_z"], label="MSD_z")
        plt.xlabel("Time (ns)")
        plt.ylabel("MSD component (Å$^2$)")
        plt.title("MSD components vs Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(comp_png, dpi=160)
        plt.close()

    return total_png, comp_png


def plot_fit(df: pd.DataFrame, y: np.ndarray, t1_ps: float, t2_ps: float,
             slope_A2_per_s: float, intercept: float, mode_label: str,
             out_png: str) -> None:
    """画带拟合线的图（以 ps 作横坐标）。"""
    t_ps = df["t_ps"].values
    t_s = df["t_s"].values
    y_fit = slope_A2_per_s * t_s + intercept

    plt.figure()
    plt.plot(t_ps, y, label=mode_label)
    plt.plot(t_ps, y_fit, "--", label=f"Linear fit ({t1_ps:.1f}-{t2_ps:.1f} ps)")
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD (Å$^2$)")
    plt.title(f"{mode_label} with linear fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ----------------------------- DIFFUSION -----------------------------
def pick_series(df: pd.DataFrame, mode: str) -> Tuple[np.ndarray, str, int]:
    """
    根据 mode 选择 y 序列与维度因子：
    - total：总 MSD，D = slope/6
    - x/y/z：单方向 MSD，D = slope/2
    返回：y, label, factor (2*d)
    """
    m = mode.lower()
    if m in ["total", "msd", "msd_total"]:
        return df["msd_total"].values, "MSD_total", 6
    elif m in ["x", "msdx", "msd_x"]:
        if "msd_x" not in df.columns:
            raise ValueError("文件里没有 msd_x 列")
        return df["msd_x"].values, "MSD_x", 2
    elif m in ["y", "msdy", "msd_y"]:
        if "msd_y" not in df.columns:
            raise ValueError("文件里没有 msd_y 列")
        return df["msd_y"].values, "MSD_y", 2
    elif m in ["z", "msdz", "msd_z"]:
        if "msd_z" not in df.columns:
            raise ValueError("文件里没有 msd_z 列")
        return df["msd_z"].values, "MSD_z", 2
    else:
        raise ValueError("mode 需为 total/x/y/z")


def default_window_ps(df: pd.DataFrame, frac_last: float = 0.4) -> Tuple[float, float]:
    n = len(df)
    i0 = max(0, int((1.0 - frac_last) * n))
    return float(df["t_ps"].iloc[i0]), float(df["t_ps"].iloc[-1])


def parse_user_window(s: str, dt_fs: float, df: pd.DataFrame) -> Tuple[float, float]:
    """
    支持两种输入：
    - 'ps 1100 1160'  或  'ns 1.10 1.16'
    - 'step 1100000 1160000'
    返回 (t1_ps, t2_ps)
    """
    toks = s.strip().split()
    if len(toks) != 3:
        raise ValueError("格式应为：'ps t1 t2' 或 'ns t1 t2' 或 'step i1 i2'")
    kind, a, b = toks[0].lower(), toks[1], toks[2]
    if kind == "ps":
        t1_ps, t2_ps = float(a), float(b)
    elif kind == "ns":
        t1_ps, t2_ps = float(a) * 1e3, float(b) * 1e3
    elif kind == "step":
        i1, i2 = float(a), float(b)
        t1_ps = i1 * dt_fs * 1e-3
        t2_ps = i2 * dt_fs * 1e-3
    else:
        raise ValueError("首个词必须是 ps/ns/step")
    if t1_ps >= t2_ps:
        raise ValueError("t1 必须小于 t2")
    # 裁剪在数据范围内
    lo, hi = float(df["t_ps"].iloc[0]), float(df["t_ps"].iloc[-1])
    t1_ps = max(lo, min(hi, t1_ps))
    t2_ps = max(lo, min(hi, t2_ps))
    return t1_ps, t2_ps


def linear_fit_D(df: pd.DataFrame, mode: str, t1_ps: float, t2_ps: float, outdir: str) -> Dict:
    """
    在 [t1_ps, t2_ps] 区间对指定 MSD 做线性拟合，返回包含 D 的结果字典。
    """
    y, label, factor = pick_series(df, mode)
    t_ps = df["t_ps"].values
    t_s = df["t_s"].values
    mask = (t_ps >= t1_ps) & (t_ps <= t2_ps)
    if mask.sum() < 2:
        raise ValueError("选定区间点数太少")

    coeff = np.polyfit(t_s[mask], y[mask], 1)  # y = slope * t + intercept
    slope_A2_per_s = float(coeff[0])
    intercept = float(coeff[1])

    # 爱因斯坦关系：<r^2> = 2*d*D*t  => D = slope / (2*d)
    D_m2_s = (slope_A2_per_s * 1e-20) / factor   # 1 Å^2 = 1e-20 m^2
    D_cm2_s = D_m2_s * 1e4

    # 画图
    out_png = os.path.join(outdir, f"MSD_fit_{mode.lower()}.png")
    plot_fit(df, y, t1_ps, t2_ps, slope_A2_per_s, intercept, label, out_png)

    return {
        "mode": mode,
        "label": label,
        "t1_ps": t1_ps,
        "t2_ps": t2_ps,
        "n_points": int(mask.sum()),
        "slope_A2_per_s": slope_A2_per_s,
        "D_m2_s": D_m2_s,
        "D_cm2_s": D_cm2_s,
        "fit_png": os.path.abspath(out_png),
    }


# ----------------------------- CONDUCTIVITY (NE: sigma = N q^2 D / (V k_B T)) -----------------------------
# 物理常数（直接预定义）
kB = 1.380649e-23        # J/K
e  = 1.602176634e-19     # C

def sigma_ne_counts(D_list, z_list, N_list, V_m3, T_K) -> float:
    """
    Nernst–Einstein（按粒子个数 N_i）:
      sigma = sum_i [ (N_i/V) * (z_i*e)^2 * D_i / (kB*T) ]    [S/m]
    参数:
      - D_list : 各物种 D_i (m^2/s)
      - z_list : 各物种价数 z_i（如 Li+ = +1, PF6- = -1）
      - N_list : 各物种个数 N_i（粒子个数）
      - V_m3   : 体积 V（m^3）
      - T_K    : 温度 T（K）
    """
    if not (len(D_list) == len(z_list) == len(N_list)):
        raise ValueError("D/z/N 三个列表长度必须一致")

    n_list = [N / V_m3 for N in N_list]  # 数密度 n_i = N_i / V (m^-3)
    sigma_S_per_m = sum(n * (z * e) ** 2 * D for n, z, D in zip(n_list, z_list, D_list)) / (kB * T_K)  # S/m
    sigma_mS_per_um = sigma_S_per_m * 1e-3  # 1 S/m = 1e-3 mS/μm

    return sigma_S_per_m, sigma_mS_per_um

def volume_from_lengths_ang(Lx_A: float, Ly_A: float, Lz_A: float) -> float:
    """
    由盒子边长（单位 Å）计算体积（m^3）： V = Lx*Ly*Lz * 1e-30
    """
    return (Lx_A * 1e-10) * (Ly_A * 1e-10) * (Lz_A * 1e-10)


# ----------------------------- MAIN (CLI + INTERACTIVE) -----------------------------
def main():
    p = argparse.ArgumentParser(description="从 MSD.txt 计算扩散系数 (Einstein 关系)，并保存拟合图。")
    p.add_argument("--file", default="MSD.txt", help="输入文件（同目录）")
    p.add_argument("--dt-fs", type=float, default=1.0, help="时间步长，fs（real 单位；默认 1）")
    p.add_argument("--mode", default=None, help="非交互模式：total/x/y/z（默认进入交互选择）")
    p.add_argument("--t1-ps", type=float, default=None, help="非交互模式：拟合区间起点（ps）")
    p.add_argument("--t2-ps", type=float, default=None, help="非交互模式：拟合区间终点（ps）")
    p.add_argument("--outdir", default="msd_output", help="图片输出目录")
    args = p.parse_args()

    if not os.path.exists(args.file):
        print(f"[错误] 找不到输入文件：{args.file}", file=sys.stderr)
        sys.exit(1)

    outdir = ensure_outdir(args.outdir)
    df = load_msd(args.file, dt_fs=args.dt_fs)

    # 概览图
    total_png, comp_png = plot_overview(df, outdir)
    print(f"[保存] 总 MSD 图：{os.path.abspath(total_png)}")
    if comp_png:
        print(f"[保存] 分量 MSD 图：{os.path.abspath(comp_png)}")

    # 交互/非交互选择
    if args.mode is None:
        mode = input("选择统计口径 [total/x/y/z]（回车默认 total）：").strip() or "total"
    else:
        mode = args.mode

    # 默认窗口：最后 40%
    dft1, dft2 = default_window_ps(df, frac_last=0.4)
    print(f"默认拟合窗口（最后40%）：{dft1:.3f} ps → {dft2:.3f} ps")

    if args.t1_ps is None or args.t2_ps is None:
        prompt = ("输入拟合区间（两种格式其一，留空使用默认）：\n"
                  "  1) ps 1100 1160    或  ns 1.10 1.16\n"
                  "  2) step 1100000 1160000\n> ")
        s = input(prompt).strip()
        if s == "":
            t1_ps, t2_ps = dft1, dft2
        else:
            t1_ps, t2_ps = parse_user_window(s, args.dt_fs, df)
    else:
        t1_ps, t2_ps = args.t1_ps, args.t2_ps

    # 拟合并输出
    res = linear_fit_D(df, mode=mode, t1_ps=t1_ps, t2_ps=t2_ps, outdir=outdir)
    print("\n====== 扩散系数结果 ======")
    print(f"口径       : {res['label']} ({res['mode']})")
    print(f"拟合区间   : {res['t1_ps']:.3f} → {res['t2_ps']:.3f} ps  "
          f"({res['n_points']} 点)")
    print(f"slope      : {res['slope_A2_per_s']:.6e} Å^2/s")
    print(f"D          : {res['D_m2_s']:.6e} m^2/s  =  {res['D_cm2_s']:.6e} cm^2/s")
    print(f"[保存] 拟合图：{res['fit_png']}")
    print("================================\n")

    ans = input("是否按 NE 关系计算电导率 σ = N q^2 D / (V kB T)？[y/N] ").strip().lower()
    if ans == "y":
        # 1) 温度
        T_K = float(input("温度 T (K): ").strip())

        # 2) 体积输入方式（两选一）
        howV = input("体积输入方式：输入盒子边长(Å)还是直接输入体积(Å^3)？[len/vol] ").strip().lower()
        if howV == "len":
            Lx_A = float(input("Lx (Å): ").strip())
            Ly_A = float(input("Ly (Å): ").strip())
            Lz_A = float(input("Lz (Å): ").strip())
            V_m3 = volume_from_lengths_ang(Lx_A, Ly_A, Lz_A)  # Å → m^3
            V_A3 = (Lx_A * Ly_A * Lz_A)                       # 仅用于打印
        else:
            V_A3 = float(input("体积 V (Å^3): ").strip())
            V_m3 = V_A3 * 1e-30  # Å^3 → m^3

        # 3) 物种信息
        m = int(input("参与导电的物种数量（例如 Li+ 与 PF6- 则填 2）: ").strip())
        z_list, N_list, D_list = [], [], []
        for i in range(m):
            print(f"—— 物种 {i+1} ——")
            z_i = float(input("  价数 z_i（例：+1 或 -1）: ").strip())
            N_i = float(input("  个数 N_i（粒子个数）: ").strip())

            use_fit_D = input("  D_i 使用本次拟合结果？[y/n] ").strip().lower()
            if use_fit_D == "y":
                entered = input(f"  请输入要采用的 D_i (m^2/s)（回车使用 {res['D_m2_s']:.6e}）: ").strip()
                D_i = res["D_m2_s"] if entered == "" else float(entered)
            else:
                D_i = float(input("  手动输入 D_i (m^2/s): ").strip())

            z_list.append(z_i)
            N_list.append(N_i)
            D_list.append(D_i)

        sigma_S_per_m, sigma_mS_per_um = sigma_ne_counts(D_list, z_list, N_list, V_m3, T_K)
        print(f"\n====== 电导率（Nernst–Einstein, 按个数 N/V）======")
        print(f"T = {T_K:.2f} K,  V = {V_A3:.6e} Å^3  =  {V_m3:.6e} m^3")
        for i, (z_i, N_i, D_i) in enumerate(zip(z_list, N_list, D_list), 1):
            print(f"物种{i}: z={z_i:g}, N={N_i:g}, D={D_i:.6e} m^2/s")
        print(f"σ = {sigma_S_per_m:.6e} S/m  =  {sigma_mS_per_um:.6e} mS/μm")
        print("=============================================\n")


if __name__ == "__main__":
    main()
