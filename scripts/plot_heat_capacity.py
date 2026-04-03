#!/usr/bin/env python3

import sys
import h5py
import numpy as np
import os.path
import argparse
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


def load_metadata(fname):
    """Return N, the (mean) number of non-deleted spins in the realisation."""
    with h5py.File(fname, "r") as f:
        N = float(np.array(f["/geometry/n_spins"]))
    return N


def load_E_data(fname):
    """
    Returns T, E_mean, var_mean, se_E, se_C_raw.

    For accumulated files (MC and disorder avg), ending in .davg.h5:
      - var_mean = ⟨Var_k(E)⟩_disorder  (= ⟨C⟩ · T²)
      - se_E, se_C_raw computed from inter-seed scatter via Bessel-corrected std / √K

    For single-trial files:
      - var_mean = Var(E) = E2/n − (E/n)²
      - se_E = √(var/n)
      - se_C_raw = None  (higher moments not stored)
    """
    with h5py.File(fname, "r") as f:
        g = f["/energy"]
        assert isinstance(g, h5py.Group)
        T = np.array(g["T_list"])
        E = np.array(g["E"])

        if "var" in g:
            # --- disorder-averaged file ---
            K        = np.array(g["n_disorder"]).astype(float)
            var_sum  = np.array(g["var"])
            var_sq   = np.array(g["var_sq"])
            e_sq     = np.array(g["E_sq"])

            E_mean   = E / K
            var_mean = var_sum / K

            def _se(sum_x, sum_x2, K):
                return np.where(
                    K > 1,
                    np.sqrt(np.maximum(sum_x2 - sum_x**2 / K, 0) / (K * (K - 1))),
                    np.nan,
                )

            se_E          = _se(E, e_sq, K)
            se_C_times_T2 = _se(var_sum, var_sq, K)
            return T, E_mean, var_mean, se_E, se_C_times_T2

        else:
            # --- per-disorder (or raw single-seed) file ---
            E2     = np.array(g["E2"])
            nsamp  = np.array(g["n_samples"])
            E_mean = E / nsamp
            var    = E2 / nsamp - E_mean**2
            se_E   = np.sqrt(np.maximum(var, 0) / nsamp)

            se_C_times_T2 = None
            if "var_sq" in g and "n_mc_seeds" in g:
                K      = float(np.array(g["n_mc_seeds"]))
                var_sq = np.array(g["var_sq"])
                # Bessel-corrected SE on the mean of K per-seed variance estimates
                se_C_times_T2 = np.where(
                    K > 1,
                    np.sqrt(np.maximum(var_sq - K * var**2, 0) / (K * (K - 1))),
                    np.nan,
                )
            return T, E_mean, var, se_E, se_C_times_T2


def compute_entropy(T, C_per_N):
    """
    Integrate C(T)/T dT from T_min upward (trapezoidal rule in linear T).
    Returns S_per_N with S=0 at T_min.
    """
    idx = np.argsort(T)
    T_s = T[idx]
    C_s = C_per_N[idx]
    integrand = C_s / T_s
    S = np.zeros_like(T_s)
    for i in range(1, len(T_s)):
        S[i] = S[i-1] + 0.5 * (integrand[i-1] + integrand[i]) * (T_s[i] - T_s[i-1])
    # map back to original ordering
    S_out = np.empty_like(S)
    S_out[idx] = S
    return S_out


def plot_file(fname, axes_map, color, label):
    N = load_metadata(fname)

    T, E_mean, var_mean, se_E, se_C_raw = load_E_data(fname)

    C    = var_mean / (T**2 * N)
    se_C = se_C_raw / (T**2 * N) if se_C_raw is not None else None

    idx  = np.argsort(T)
    T    = T[idx];  C = C[idx];  E_mean = E_mean[idx];  se_E = se_E[idx]
    if se_C is not None:
        se_C = se_C[idx]

    if 'C' in axes_map:
        ax = axes_map['C']
        ax.plot(T, C, marker="o", markersize=3, color=color, label=label)
        if se_C is not None:
            ax.fill_between(T, C - se_C, C + se_C, alpha=0.2, color=color)

    if 'E' in axes_map:
        ax = axes_map['E']
        E_per_N = E_mean / N
        se_E_per_N = se_E / N
        ax.plot(T, E_per_N, marker="o", markersize=3, color=color, label=label)
        ax.fill_between(T, E_per_N - se_E_per_N, E_per_N + se_E_per_N,
                        alpha=0.2, color=color)

    if 'S' in axes_map:
        ax = axes_map['S']
        S = compute_entropy(T, C)
        ax.plot(T, S, marker="o", markersize=3, color=color, label=label)


def main(args):

    fnames = args.files
    labels = args.labels
    plots  = args.plot

    cmap   = plt.colormaps["tab10"]
    colors = [cmap(i) for i in range(len(fnames))]

    if labels is None:
        labels = [os.path.basename(fname) for fname in fnames]

    axes_map = {}

    if 'C' in plots:
        fig_C, ax_C = plt.subplots(figsize=(3.5, 3))
        axes_map['C'] = ax_C

    if 'E' in plots:
        fig_E, ax_E = plt.subplots(figsize=(3.5, 3))
        axes_map['E'] = ax_E

    if 'S' in plots:
        fig_S, ax_S = plt.subplots(figsize=(3.5, 3))
        axes_map['S'] = ax_S

    for fname, color, label in zip(fnames, colors, labels):
        plot_file(fname, axes_map, color, label)

    if 'C' in axes_map:
        ax = axes_map['C']
        ax.set_ylabel("Specific heat $C/N$")
        ax.set_xlabel("Temperature $T$")
        ax.set_xscale("log")
        if args.y_logscale:
            ax.set_yscale("log")
        ax.grid(True)
        ax.legend(fontsize=7)
        if args.xlim is not None:
            ax.set_xlim(args.xlim)
        ax.figure.tight_layout()

    if 'E' in axes_map:
        ax = axes_map['E']
        ax.set_ylabel(r"Mean energy $\langle E \rangle / N$")
        ax.set_xlabel("Temperature $T$")
        ax.set_xscale("log")
        ax.grid(True)
        ax.legend(fontsize=7)
        if args.xlim is not None:
            ax.set_xlim(args.xlim)
        ax.figure.tight_layout()

    if 'S' in axes_map:
        ax = axes_map['S']
        ax.set_ylabel(r"Entropy $S/N$ (integrated from $T_\mathrm{min}$)")
        ax.set_xlabel("Temperature $T$")
        ax.set_xscale("log")
        ax.grid(True)
        ax.legend(fontsize=7)
        if args.xlim is not None:
            ax.set_xlim(args.xlim)
        ax.figure.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots heat capacity of one or many merged seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("files", nargs='+', help="output hdf5 files")
    parser.add_argument("--labels", nargs='+', help="labels for plot")
    parser.add_argument("--xlim", nargs=2, help="x limits", type=float)
    parser.add_argument("--y_logscale", action="store_true")
    parser.add_argument("--plot", nargs='+', default=['C'],
                        choices=['C', 'E', 'S'],
                        metavar='{C,E,S}',
                        help="figures to show: C=heat capacity, E=mean energy, S=integrated entropy (default: C)")
    args = parser.parse_args()

    main(args)
