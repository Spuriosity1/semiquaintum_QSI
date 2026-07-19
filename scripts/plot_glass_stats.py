#!/usr/bin/env python3
"""
plot_glass_stats.py — Plot glass diagnostics <J> and stdev(J) as well as 
loop frustration fraction, as a fucntion of Jpm / Jzz

Reads glass_stats outputm

Usage:
    python scripts/plot_hhl.py output.h5
"""

import argparse
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

# a_cubic in lattice-integer units: each conventional cubic cell spans 8 steps
_A_CUBIC_INT = 8


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot glass diagnostics from glass_stats output"
    )
    p.add_argument('--use_weighted', action="store_true")
    p.add_argument("filename", help="HDF5 output file from glass_stats")
    p.add_argument("--title", help="Plot title (default: filename)")
    p.add_argument("--save", metavar="FILE", help="Save figure to FILE")
    return p.parse_args()


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------

def read_glass_stats(filename, frust='frust'):
    with h5py.File(filename, "r") as f:
        jpm_list = np.array(f["jpm"][:])
        L        = int(f["L"][()])
        nsweep = int(f["nsweep"][()])

        frust_cycle = {
                3: np.array(f[f"{frust}_3_cycle"][:]),
                4: np.array(f[f"{frust}_4_cycle"][:])
                }

        # characterises convergence in L, not really that physically interesting
        frust_cycle_stdev = {
                3: np.array(f[f"{frust}_3_cycle_stdev"][:]),
                4: np.array(f[f"{frust}_4_cycle_stdev"][:])
                }

        
        J_mean = np.array(f["J_mean"][:])
        J_stdev = np.sqrt(f["J_variance"][:])

    return jpm_list, L, nsweep, (J_mean, J_stdev), (frust_cycle, frust_cycle_stdev)


def main():
    args = parse_args()

    title=args.title if hasattr(args, "title") else args.filename

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(4, 3.5), sharex=True)

    dset = 'wfrust' if args.use_weighted else 'frust'

    jpm_list, L, nsweep, J_stats, cycle_stats = read_glass_stats(args.filename, dset)

    J, J_stdev = J_stats
    cyc, cyc_stdev = cycle_stats
    pos = jpm_list>0
    neg = jpm_list>0

    ax1.plot(jpm_list[pos], J[pos], '-', color='k', label=r"$\langle \mathcal{J}_{ij} \rangle/J_{zz}$")
    ax1.plot(jpm_list[pos], J_stdev[pos], ':', color='k', label=r"$\sigma_{\mathcal{J}_{ij}}/J_{zz}$")
    ax1.set_yscale('log')

    for c in [3, 4]:
        ax2.errorbar(jpm_list[pos], cyc[c][pos], yerr=cyc_stdev[c][pos], label=f"{c}-cycle frust.")



    title=args.title if args.title is not None else args.filename
    ax1.set_title(title)

    ax2.set_xlabel(r"$J_\pm/J_{zz}$")
    ax2.set_ylabel(r"$P$(frustrated)")

    ax1.legend()
    ax2.legend()

    fig.tight_layout()
    plt.show()





if __name__ == "__main__":
    main()
