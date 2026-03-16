#!/usr/bin/env python3

import matplotlib.pyplot as plt
import re
import numpy as np
import argparse
import os.path


def parse_filename(path):
    pattern = r"L(\d+)_p([0-9.]+)_s([0-9]+)_w([0-9]+)"
    match = re.search(pattern, path)
    if not match:
        raise ValueError(f"Filename not in expected format: {path}")
    
    L = int(match.group(1))
    p = float(match.group(2))
    seed = int(match.group(3))
    nsweep = int(match.group(4))
    return L, p, seed, nsweep

def main():
    parser = argparse.ArgumentParser(
        description="Plots the histogram.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("histfile", nargs='+', help="hist CSV file")
    parser.add_argument("--logplot", action='store_true',help='plots on a log scale')
#    parser.add_argument("--rebin")
    parser.add_argument("--plot_ratio", action='store_true',
                        help=r"plots percent of spins in the cluster ")

    args=parser.parse_args()

    fig, ax = plt.subplots()

    if args.plot_ratio:
        yfunc = lambda data, N : 100*data[:,1]*data[:,0]/N
        ax.set_ylabel(r"% of spins in this cluster")
    else:
        yfunc = lambda data, N : data[:,1]/N
        ax.set_ylabel("Probability")

    nfiles = len(args.histfile)
    cmap = plt.colormaps['jet']

    for i, f in enumerate(args.histfile):
        L, p, seed, nsweep = parse_filename(os.path.basename(f))
        data=np.genfromtxt(f,delimiter='\t',skip_header=1)
        if (data.ndim < 2):
            continue

        N=L*L*L*16
        ax.plot(data[:,0], yfunc(data, N*nsweep) ,'+-',label=f'L={L} p={p}', lw=1,
                color=cmap(i/(nfiles-1)))

        percolmask = data[:,0]>500
        total_percolaing = np.sum(data[percolmask,1]*data[percolmask,0])
        print(f"p={p} average {100.0*total_percolaing/nsweep/N}% of spins in percolating cluster")

    ax.set_title("Cluster scaling")
    ax.set_xlabel("Cluster Size")

    fig.legend()

    if args.logplot:
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.set_xlim(0,20)
        ax.set_xticks(np.arange(0,21),minor=True)
        ax.set_xticks(np.arange(0,21,2),minor=False)
        ax.set_ylim(0,None)


    plt.show()

main()
