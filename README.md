# MSD Diffusion & Conductivity Helper

This repository contains a small command-line utility, `msd_diffusion.py`, for processing mean square displacement (MSD) data exported from molecular dynamics simulations. It can:

- Load MSD tables (total-only or with x/y/z components) and generate overview plots.
- Interactively or non-interactively fit the linear portion of MSD to obtain diffusion coefficients via the Einstein relation.
- Optionally compute ionic conductivity using the Nernst–Einstein relation with user-specified species counts, charges, and diffusion coefficients.

## Requirements

The script depends on Python 3 with the following packages:

- `numpy`
- `pandas`
- `matplotlib`

## Usage

```bash
python msd_diffusion.py --file MSD.txt --dt-fs 1.0 --outdir msd_output
```

- `--mode`, `--t1-ps`, and `--t2-ps` can be supplied to run fully non-interactively; otherwise, the script prompts for a fitting window.
- Set `--dt-fs` to match the timestep (in femtoseconds) used in the simulation.
- All plots are saved under the directory provided by `--outdir` (created automatically).

The conductivity helper is optional and is prompted after the diffusion fit completes.

## Direct conductivity calculation

If you already have MSD outputs (`msd_Li.out`, `msd_PF6.out`), `GK_info.txt`, and the
LAMMPS input file (`in.lammps` with the `group Li`/`group PF6` definitions), you can
compute conductivity non-interactively:

```bash
python conductivity_auto.py \
  --gk-info GK_info.txt \
  --in-file in.lammps \
  --msd-li msd_Li.out \
  --msd-pf6 msd_PF6.out
```

- The script reads volume, timestep, and temperature from `GK_info.txt`.
- Li/PF6 counts are parsed from the molecule index ranges in `in.lammps`.
- Diffusion coefficients are fitted from the MSD files using the last 40% of the data by default; pass `--window t1 t2` (ps) to override.
- Outputs Li/PF6 diffusion coefficients and the Nernst–Einstein conductivity directly to stdout.
