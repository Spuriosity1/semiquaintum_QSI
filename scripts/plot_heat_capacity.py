#!/usr/bin/env python3

import re
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
        N = np.array(f["/geometry/n_spins"])
        if hasattr(N, "__len__"):
            N = np.mean(N)
        else:
            N = float(N)
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


def csv_stem(fname):
    """Strip .h5 (including .davg.h5) suffix from basename."""
    return re.sub(r'\.h5$', '', os.path.basename(fname))


def load_theory_data(fname):
    """Load pre-computed Cv curves from a generate_theory output file.

    Returns T_list, Cv_classical, Cv_theory, Cv_total (all per spin, ascending T).
    """
    with h5py.File(fname, "r") as f:
        grp = f["heat_capacity"]
        T     = np.array(grp["T_list"])
        C_cl  = np.array(grp["Cv_classical"])
        C_th  = np.array(grp["Cv_theory"])
        C_tot = np.array(grp["Cv_total"])
    idx   = np.argsort(T)
    return T[idx], C_cl[idx], C_th[idx], C_tot[idx]


def plot_theory_file(fname, axes_map, color, label, save_directory=None):
    """Overlay theory curves on the C axis (dashed lines, no markers)."""
    if 'C' not in axes_map:
        return

    T, C_cl, C_th, C_tot = load_theory_data(fname)
    ax = axes_map['C']

    ax.plot(T, C_tot, linestyle="--",  color=color, label=label)

    if save_directory is not None:
        stem = csv_stem(fname)
        np.savetxt(
            os.path.join(save_directory, stem + '.C.csv'),
            np.column_stack([T, C_tot, C_cl, C_th]),
            delimiter=',', header='T,Cv_total,Cv_classical,Cv_theory', comments='',
        )


def plot_file(fname, axes_map, color, label, save_directory=None):
    N = load_metadata(fname)

    T, E_mean, var_mean, se_E, se_C_raw = load_E_data(fname)

    C    = var_mean / (T**2 * N)
    se_C = se_C_raw / (T**2 * N) if se_C_raw is not None else None

    idx  = np.argsort(T)
    T    = T[idx];  C = C[idx];  E_mean = E_mean[idx];  se_E = se_E[idx]
    if se_C is not None:
        se_C = se_C[idx]

    E_per_N    = E_mean / N
    se_E_per_N = se_E / N
    S          = compute_entropy(T, C)

    if 'C' in axes_map:
        ax = axes_map['C']
        ax.plot(T, C, marker="o", markersize=3, color=color, label=label)
        if se_C is not None:
            ax.fill_between(T, C - se_C, C + se_C, alpha=0.2, color=color)

    if 'E' in axes_map:
        ax = axes_map['E']
        ax.plot(T, E_per_N, marker="o", markersize=3, color=color, label=label)
        ax.fill_between(T, E_per_N - se_E_per_N, E_per_N + se_E_per_N,
                        alpha=0.2, color=color)

    if 'S' in axes_map:
        ax = axes_map['S']
        ax.plot(T, S, marker="o", markersize=3, color=color, label=label)

    if save_directory is not None:
        stem   = csv_stem(fname)
        nan_col = np.full_like(T, np.nan)

        if 'C' in axes_map:
            se_col = se_C if se_C is not None else nan_col
            np.savetxt(
                os.path.join(save_directory, stem + '.C.csv'),
                np.column_stack([T, C, se_col]),
                delimiter=',', header='T,C_per_N,se_C_per_N', comments='',
            )

        if 'E' in axes_map:
            np.savetxt(
                os.path.join(save_directory, stem + '.E.csv'),
                np.column_stack([T, E_per_N, se_E_per_N]),
                delimiter=',', header='T,E_per_N,se_E_per_N', comments='',
            )

        if 'S' in axes_map:
            np.savetxt(
                os.path.join(save_directory, stem + '.S.csv'),
                np.column_stack([T, S, nan_col]),
                delimiter=',', header='T,S_per_N,se_S_per_N', comments='',
            )


def main(args):

    fnames  = args.files
    labels  = args.labels
    tnames  = args.theory  or []
    tlabels = args.theory_labels
    plots   = args.plot

    n_total = len(fnames) + len(tnames)
    cmap    = plt.colormaps["tab10"]
    colors  = [cmap(i) for i in range(n_total)]

    if labels is None:
        labels = [os.path.basename(fname) for fname in fnames]
    if tlabels is None:
        tlabels = [os.path.basename(tn) for tn in tnames]

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

    save_to = args.save_to
    if save_to is not None and not os.path.isdir(save_to):
        raise IOError(f"Cannot save to {save_to}: not a directory") 

    for fname, color, label in zip(fnames, colors, labels):
        plot_file(fname, axes_map, color, label, save_to)

    for tname, color, label in zip(tnames, colors[len(fnames):], tlabels):
        plot_theory_file(tname, axes_map, color, label, save_to)

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
    parser.add_argument("--theory", nargs='+', metavar="THEORY_H5",
                        help="theory HDF5 files from generate_theory (overlaid as dashed lines on C plot)")
    parser.add_argument("--theory_labels", nargs='+', help="labels for theory curves")
    parser.add_argument("--save_to", help="saves plot data to specified directory", default=None)
    parser.add_argument("--xlim", nargs=2, help="x limits", type=float)
    parser.add_argument("--y_logscale", action="store_true")
    parser.add_argument("--plot", nargs='+', default=['C'],
                        choices=['C', 'E', 'S'],
                        metavar='{C,E,S}',
                        help="figures to show: C=heat capacity, E=mean energy, S=integrated entropy (default: C)")
    args = parser.parse_args()

    main(args)
