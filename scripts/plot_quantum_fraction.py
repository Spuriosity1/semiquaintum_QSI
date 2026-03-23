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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


def parse_filename(path):
    """Extract (L, p, seed, nsweep, clust_type) from a hist CSV filename."""
    name = os.path.basename(path)
    m = re.search(r"L(\d+)_p([0-9.]+)_s(\d+)_w(\d+)_([\w]+)\.csv$", name)
    if not m:
        raise ValueError(f"Filename not in expected format: {path}")
    L        = int(m.group(1))
    p        = float(m.group(2))
    seed     = int(m.group(3))
    nsweep   = int(m.group(4))
    clust_type = m.group(5)
    return L, p, seed, nsweep, clust_type


def quantum_fraction(csv_path, L, nsweep):
    """Return fraction of spins residing in quantum clusters."""
    data = np.genfromtxt(csv_path, delimiter='\t', skip_header=1)
    if data.ndim < 2 or data.size == 0:
        return 0.0
    n_quantum = np.sum(data[:, 0] * data[:, 1])   # size * count
    n_total   = L**3 * 16 * nsweep
    return n_quantum / n_total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("histfiles", nargs='+', help="hist CSV files from sq_pyrochlore_dump")
    parser.add_argument("--save", metavar="FILE", help="save figure instead of showing")
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

    fig, ax = plt.subplots(figsize=(3.5, 3))

    styles = {
            'nn2': {
        'ls': None, 
        'color': 'k',
        'label': '2nn',
        'marker': '+'
        },
            'nn24': {
        'ls': None, 
        'color': 'r',
        'label': r'2nn~$\cup$~4nn',
        'marker': 'x'
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

    for (L, ctype), points in sorted(groups.items()):
        points.sort()
        ps, fracs = zip(*points)

        st = styles.get(ctype, default_style(ctype))
        ax.plot(ps, fracs/(1-np.array(ps)), **st)

    ax.set_xlabel("Disorder concentration $p$")
    ax.set_ylabel("Fraction of spins in quantum clusters")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0,decimals=0))

    fig.tight_layout()

    ax.set_xticks(np.arange(0,0.25, 0.01),minor=True)
    ax.set_yticks(np.arange(0,1, 0.02),minor=True)

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


main()
