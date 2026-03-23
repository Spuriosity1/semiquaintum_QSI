#!/usr/bin/env python3

import sys
import h5py
import numpy as np
import os.path
import matplotlib.pyplot as plt





def load_metadata(fname):
    """Return N, the (mean) number of non-deleted spins in the realisation."""
    with h5py.File(fname, "r") as f:
        N = float(np.array(f["/geometry/n_spins"]))
    return N


def load_E_data(fname):
    """
    Returns T, E_mean, var_mean, se_E, se_C_raw.

    For accumulated files (produced by acc_heat_capacity):
      - var_mean = ⟨Var_k(E)⟩_disorder  (= ⟨C⟩ · T²)
      - se_E, se_C_raw computed from inter-seed scatter via Bessel-corrected std / √K

    For single-seed files:
      - var_mean = Var(E) = E2/n − (E/n)²
      - se_E = √(var/n)
      - se_C_raw = None  (higher moments not stored)
    """
    with h5py.File(fname, "r") as f:
        g = f["/energy"]
        assert isinstance(g, h5py.Group)
        T     = np.array(g["T_list"])
        E     = np.array(g["E"])
        nsamp = np.array(g["n_samples"])

        if "var" in g:
            # --- accumulated file ---
            K        = nsamp.astype(float)
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
            # --- single-seed file ---
            E2     = np.array(g["E2"])
            E_mean = E / nsamp
            var    = E2 / nsamp - E_mean**2
            se_E   = np.sqrt(np.maximum(var, 0) / nsamp)
            return T, E_mean, var, se_E, None


def plot_file(fname, axes, color, label):
    N = load_metadata(fname)

    T, E_mean, var_mean, se_E, se_C_raw = load_E_data(fname)

    C    = var_mean / (T**2 * N)
    se_C = se_C_raw / (T**2 * N) if se_C_raw is not None else None

    idx  = np.argsort(T)
    T    = T[idx];  C = C[idx];  E_mean = E_mean[idx];  se_E = se_E[idx]
    if se_C is not None:
        se_C = se_C[idx]

    ax_C, ax_E = axes

    ax_C.plot(T, C, marker="o", markersize=3, color=color, label=label)
    # ax_C.plot(0.5*(T[1:]+T[:-1]), np.diff(E_mean)/np.diff(T) / N )
    if se_C is not None:
        ax_C.fill_between(T, C - se_C, C + se_C, alpha=0.2, color=color)

    ax_E.plot(T, E_mean / N, marker="o", markersize=3, color=color, label=label)
    ax_E.fill_between(T, (E_mean - se_E) / N, (E_mean + se_E) / N, alpha=0.2, color=color)


def main(fnames):
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

    cmap   = plt.colormaps["tab10"]
    colors = [cmap(i) for i in range(len(fnames))]

    for fname, color in zip(fnames, colors):
        label = os.path.basename(fname)
        plot_file(fname, axes, color, label)

    ax_C, ax_E = axes
    ax_C.set_ylabel("Specific heat $C/N$")
    ax_C.set_xscale("log")
    ax_C.set_yscale("log")
    ax_C.grid(True)
    ax_C.legend(fontsize=7)

    ax_E.set_ylabel("Mean energy $\\langle E \\rangle / N$")
    ax_E.set_xlabel("Temperature $T$")
    ax_E.set_xscale("log")
    ax_E.grid(True)
    ax_E.legend(fontsize=7)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_heat_capacity.py FILE1.h5 [FILE2.h5 ...]")
        sys.exit(1)

    main(sys.argv[1:])
