#!/usr/bin/env python3

import matplotlib.pyplot as plt
import re
import numpy as np
import argparse
import os.path
import matplotlib as mpl
import h5py


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


def parse_filename(path):
    pattern = r"L(\d+)_p([0-9.]+)"
    match = re.search(pattern, path)
    if not match:
        raise ValueError(f"Filename not in expected format: {path}")
    
    L = int(match.group(1))
    p = float(match.group(2))
    return L, p


def log_rebin(sizes, counts, count_vars,
              n_bins=50, s_min=None, s_max=None):
    s_min = max(sizes.min(), 1) if s_min is None else s_min
    s_max = sizes.max() if s_max is None else s_max
    edges = np.logspace(np.log10(s_min), np.log10(s_max), n_bins + 1)
    binned_counts = np.zeros(n_bins)
    binned_count_vars = np.zeros(n_bins)
    binned_spin_count_vars = np.zeros(n_bins)
    binned_spin_counts = np.zeros(n_bins)
    for k in range(n_bins):
        mask = (sizes >= edges[k]) & (sizes < edges[k + 1])
        binned_counts[k] = counts[mask].sum()
        binned_count_vars[k] = count_vars[mask].sum()

        binned_spin_counts[k] = (counts[mask] * sizes[mask]).sum()
        binned_count_vars[k] = (
                count_vars[mask] * sizes[mask] * sizes[mask]).sum()

    
    return edges[:-1], edges[1:], binned_counts, binned_spin_counts, binned_count_vars, binned_spin_count_vars

def main():
    parser = argparse.ArgumentParser(
        description="Plots the histogram.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("histfile", nargs='+', help="hist CSV file")
    parser.add_argument("--logplot", action='store_true',help='plots on a log scale')
    parser.add_argument("--title")
#    parser.add_argument("--rebin")
    parser.add_argument("--plot_ratio", action='store_true',
                        help=r"plots fraction of spins in the cluster (i.e. cluster frequency * cluster size) ")
    parser.add_argument("--plot_legend", action='store_true',
                        help=r"plots a legend ")
    parser.add_argument("--plot_colorbar", action='store_true',
                        help=r"plots a colorbar legend ")

    args=parser.parse_args()

    fig, ax = plt.subplots(figsize=(3.5, 3))

    if args.plot_ratio:
        ax.set_ylabel(r"Cluster density")
    else:
        ax.set_ylabel("Probability density per spin")

    nfiles = len(args.histfile)
    cmap = plt.colormaps['jet']


    # --- collect p values first ---
    ps = []
    for f in args.histfile:
        _, p = parse_filename(os.path.basename(f))
        ps.append(p)

    # --- set up normalization based on p ---
    norm = mpl.colors.Normalize(vmin=min(ps), vmax=max(ps))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i, f in enumerate(args.histfile):
        L, p = parse_filename(os.path.basename(f))

        with h5py.File(f, "r") as f:
            sizes=np.array(f["sizes"][()])
            counts=np.array(f["counts"][()])
            nsweep=np.array(f["nsweep"][()])
            count_vars=np.array(f["var"][()])
            N = int(np.array(f["N"]))

        max_decade = np.ceil(10*np.log10(N))/10
        lefts, rights, bc, bsc, v_bc, v_bsc = log_rebin(sizes, counts, count_vars,
                                       s_min=1, s_max=np.pow(10,max_decade),
                                       n_bins=int(max_decade*10+1))
        color = cmap(norm(p))
        if args.plot_ratio:
            ys = bsc / (N * nsweep)
            ys_error = np.sqrt(v_bsc) / (N * nsweep)
        else:
            ys = bc / (N * nsweep)
            ys_error = np.sqrt(v_bc) / (N * nsweep)

        x = np.empty(lefts.size *3, dtype=lefts.dtype)
        err = np.zeros(lefts.size *3, dtype=lefts.dtype)

        middles = np.sqrt(lefts*rights)
        x[0::3] = lefts
        x[1::3] = middles
        x[2::3] = rights
        err[1::3] = ys_error
        ax.errorbar(x, np.repeat(ys,3), yerr=err, fmt='-',
                color=color, lw=1, label=fr'${p*100:.0f}\%$')

        percolmask = counts>500
        total_percolaing = np.sum(counts[percolmask]*sizes[percolmask])
        print(f"p={p} average {100.0*total_percolaing/nsweep/N}% of spins in percolating cluster")

    if (args.title is not None):
        ax.set_title(args.title)
    ax.set_xlabel("Cluster Size")

    if args.plot_legend:
        fig.legend()
    elif args.plot_colorbar:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Disorder fraction $p$")

    if args.logplot:
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.set_xlim(0,20)
        ax.set_xticks(np.arange(0,21),minor=True)
        ax.set_xticks(np.arange(0,21,2),minor=False)
        ax.set_ylim(0,None)

    fig.tight_layout()
    plt.show()

main()
