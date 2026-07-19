#!/usr/bin/env python3
"""
Plot the fraction of spins in quantum clusters vs disorder concentration p.

Accepts hist CSV files produced by sq_pyrochlore_dump.  Multiple files at
different p values and/or different cluster definitions (nn2, nn24, nn2_mf)
are overlaid on the same axes; one line per (L, cluster_def) combination.

Usage:
    python scripts/plot_quantum_fraction.py out/hist_*.csv
    python scripts/plot_quantum_fraction.py out/hist_*_nn2.csv out/hist_*_nn24.csv

Generating the input data (example sweep over p values):
    for p in 0.01 0.02 0.05 0.10 0.15 0.20; do
        for cdef in nn2 nn24 nn2_mf; do
            ./build/percol/sq_pyrochlore_dump 4 $p \
                --cluster_def $cdef --nsweep 500 --output_dir out/
        done
    done
"""

import argparse
import os.path
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import h5py

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8,
    'font.sans-serif': 'cm'
})



def parse_filename(path):
    """Extract (L, p, seed, nsweep, clust_type) from a hist HDF5 filename."""
    name = os.path.basename(path)
    m = re.search(r"L(\d+)_p([0-9.]+)_s(\d+)_w(\d+)_([\w]+)\.h5$", name)
    if not m:
        raise ValueError(f"Filename not in expected format: {path}")
    L        = int(m.group(1))
    p        = float(m.group(2))
    seed     = int(m.group(3))
    nsweep   = int(m.group(4))
    clust_type = m.group(5)
    return L, p, seed, nsweep, clust_type


def parse_h5(h5_path):
    with h5py.File(h5_path) as hf:
        nsweep = int(hf["nsweep"][()])
        counts = np.array(hf["counts"][:])
        sizes = np.array(hf["sizes"][:])
    
    return sizes, counts, nsweep


def quantum_fraction(h5_path, L, nsweep):
    """Return fraction of spins residing in quantum clusters."""
    size, count, nsweep = parse_h5(h5_path)
    n_quantum = np.sum(size * count)   # size * count
    n_total   = L**3 * 16 * nsweep
    return n_quantum / n_total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("histfiles", nargs='+', help="hist CSV files from sq_pyrochlore_dump")
    parser.add_argument("--save", metavar="FILE", help="save figure instead of showing")
    parser.add_argument("--plot-remainder", nargs='+', help="ctypes to plot 1- of")
    parser.add_argument("--figsize", nargs=2, type=float, help="Figure size in inches", default=(3.5,2))
    args = parser.parse_args()

    # group files by (L, clust_type)
    groups = defaultdict(list)   # (L, clust_type) -> [(p, fraction)]
    for path in args.histfiles:
        try:
            L, p, seed, nsweep, ctype = parse_filename(path)
        except ValueError as e:
            print(f"Skipping {path}: {e}")
            continue
        frac = quantum_fraction(path, L, nsweep)
        if ctype == 'nn2_mf':
            continue

        groups[(L, ctype)].append((p, frac))

    if not groups:
        print("No valid files found.")
        return

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    styles = {
            'nn2': {
        'ls': None, 
        'color': 'g',
        'label': r'$O(J_\pm)$',
        'marker': None
        },
            'nn24': {
        'ls': None, 
        'color': 'b',
        'label': r'$O(J_{\pm}^{2}/J_{zz})$',
        'marker': None
        }
    }

    default_style = lambda ct : {
        'ls': None, 
        'color': 'r',
        'label': ct,
        'marker': 'o'
        }

#    cmap = plt.colormaps['tab10']
#    L_values = sorted({L for L, _ in groups})
#    L_color  = {L: cmap(i) for i, L in enumerate(L_values)}

    dstore= {}
    for (L, ctype), points in sorted(groups.items()):
        points.sort()
        ps, fracs = zip(*points)

        st = styles.get(ctype, default_style(ctype))
        fracs_corrected=fracs/(1-np.array(ps))
        ax.plot(ps, fracs_corrected, **st)

        if args.plot_remainder and ctype=='nn24':
            ax.plot(ps, 1.-fracs_corrected, 'k', label='ringflip only')

    ax.set_xlabel(r"Disorder $p$")
    ax.set_ylabel(r"$N_u/N_{sp}$")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0,decimals=0))

    fig.tight_layout()


    ax.set_xticks(np.arange(0,0.25, 0.1),minor=False)
    ax.set_yticks(np.arange(0,1.1, 0.25),minor=False)

    ax.set_xticks(np.arange(0,0.26, 0.01),minor=True)
    ax.set_yticks(np.arange(0,1, 0.02),minor=True)

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


main()
