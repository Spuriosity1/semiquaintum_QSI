#!/usr/bin/env python3
"""
plot_ssf.py — Plot the static structure factor S(q) in the (h,h,l) plane.

Reads the /ssf group from sq_pyrochlore HDF5 output.  Q-points are expressed in
conventional cubic reciprocal lattice units (r.l.u.), where (1,0,0) = 2π/a_cubic.

Usage examples:
    python scripts/plot_ssf.py output.h5
    python scripts/plot_ssf.py output.h5 --dataset Szz --log
    python scripts/plot_ssf.py output.h5 --T -1 --save hhl.png
    python scripts/plot_ssf.py output.h5 --all --save all_T.png
"""

import argparse
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot SSF S(q) in the (h,h,l) plane from sq_pyrochlore HDF5 output."
    )
    p.add_argument("filename", help="HDF5 output file from sq_pyrochlore")
    p.add_argument(
        "--dataset", default="Sqq",
        help="Which SSF dataset to plot: Sqq (neutron, default) or Szz (local-axis Ising)",
    )
    p.add_argument(
        "--T", type=int, default=0, metavar="INDEX",
        help="Temperature index to plot (default: 0 = lowest T; negative indices wrap)",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Plot all temperatures as a tiled grid of panels",
    )
    p.add_argument(
        "--log", action="store_true",
        help="Use a logarithmic colour scale",
    )
    p.add_argument(
        "--clim", nargs=2, type=float, metavar=("VMIN", "VMAX"),
        help="Fix colour scale limits (applied to every panel)",
    )
    p.add_argument(
            "--e0", nargs=3, type=int, metavar=("QX","QY","QZ"),
            help="first vector to extract S(q) along (integers)",
            default=(1,0,0))
    p.add_argument(
            "--e1", nargs=3, type=int, metavar=("QX","QY","QZ"),
            help="first vector to extract S(q) along (integers)",
            default=(0,1,0))
    p.add_argument(
        "--save", metavar="FILE",
        help="Save figure to FILE instead of opening an interactive window",
    )
    return p.parse_args()


def read_ssf(filename, dataset):
    with h5py.File(filename, "r") as f:
        grp = f["/ssf"]
        T_list   = grp["T_list"][:]       # (n_T,)
        n_samples = grp["n_samples"][:]   # (n_T,)
        data     = grp[dataset][:]        # (n_T, n_k)
        n_spins  = int(grp.attrs["n_spins"])
        k_dims   = grp.attrs["k_dims"]    # [Lx, Ly, Lz]
    return T_list, n_samples, data, n_spins, tuple(int(d) for d in k_dims)

def read_recip_latvecs(filename):
    with h5py.File(filename, "r") as f:
        grp = f["/geometry"]

        B = np.array(grp["recip_vectors"][:])
    return B


def extract_plane(data, n_spins, k_dims, e0, e1):
    """
    Normalise, reshape, and extract a slice along a particular plane.

    Args
    ----
    data : ndarray, shape (n_T, n_k)
    n_spins : int (only used as a normalisation constant)
    k_dims: tuple (Lx, Ly, Lz)
    e0 : plane basis vector 0, integer, units of r.l.v.
    e1 : plane basis vector 1, integer, units of r.l.v.

    Returns
    -------
    S_slice : ndarray, shape (n_T, N0, N1)
        S(Q(x,y)) / site, with the BZ centred (Γ at the middle of each axis),
        Q(x,y) = e0 * x + e1 * y
    axis_1 : ndarray, shape (N0,)
        values in r.l.u.  (Γ at 0)
    axis_2 : ndarray, shape (N1,)
        values in r.l.u.  (Γ at 0)
    """
    Lx, Ly, Lz = k_dims


    n_T = data.shape[0]

    # Normalise to S(q) per site
    S = data / n_spins                   # (n_T, n_k)

    # Reshape to 4D; row-major so index (k0,k1,k2) ↔ flat k0*Ly*Lz + k1*Lz + k2
    S4d = S.reshape(n_T, Lx, Ly, Lz)


    # indexing BS: Figure out the periodicity in e0, e1
    # Assert that e0 * p0 = a recip. latt. vector for some M.
    # e0 * p0  = (Lx nx, Ly ny, Lz nz)^T for integer nx,ny,nz
    # => need to solve simultaneously
    # p0 e0[a] = 0 mod L[a]
    # divide by gcd(e0[a], L[a])
    # p0 e0[a]/gcd = 0 mod L[a]/gcd 
    # so periodicity p0 = lcm( e0[a]/gcd(e0[a], L[a]) , a=1,2,3)

    p0 = 1
    p1 = 1
    for a in range(3):
        L = k_dims[a]
        p0 = np.lcm(p0, L // np.gcd(int(e0[a]), L)) 
        p1 = np.lcm(p1, L // np.gcd(int(e1[a]), L)) 

    # Build index arrays: Q(x,y) = e0*x + e1*y, x in [0,p0), y in [0,p1)
    xs = np.arange(p0)
    ys = np.arange(p1)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')  # (p0, p1)

    # Map to 3D k-indices mod L for each component
    i0 = (e0[0] * xx + e1[0] * yy) % Lx   # (p0, p1)
    i1 = (e0[1] * xx + e1[1] * yy) % Ly
    i2 = (e0[2] * xx + e1[2] * yy) % Lz

    # Extract plane via fancy indexing
    S_slice = S4d[:, i0, i1, i2]              # (n_T, p0, p1)

    # fftshift each spatial axis so Γ sits at the centre
    S_slice = np.fft.fftshift(S_slice, axes=(1, 2))

    # Build axes in r.l.u., centred at Γ=0
    # The step along axis 0 is |e0| r.l.u., etc.
    axis_1 = (xs - p0 // 2) * np.linalg.norm(e0)   # shape (p0,)
    axis_2 = (ys - p1 // 2) * np.linalg.norm(e1)   # shape (p1,)

    return S_slice, axis_1, axis_2

def make_edges(axis):
    """Convert midpoint coordinates to bin edges for pcolormesh."""
    d = axis[1] - axis[0]
    return np.append(axis - d / 2, axis[-1] + d / 2)


def plot_panel(ax, S2d, h_axis, l_axis, T, dataset, log_scale, clim,
               transform=((1,0),(0,1)) ):
    """Draw one 2D colour-map panel on *ax*."""
    if clim is not None:
        vmin, vmax = clim
    else:
        # Exclude the q=0 Gamma-point spike (scales as N^2) from the colour limits
        # so the rest of the BZ is visible on a linear scale.
        flat = S2d.ravel()
        flat_pos = flat[flat > 0]
        vmax = np.percentile(flat_pos, 99) if flat_pos.size else flat.max()
        vmin = flat_pos.min() if log_scale and flat_pos.size else flat.min()

    if log_scale:
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    h_edges = make_edges(h_axis)
    l_edges = make_edges(l_axis)

#    x_points, y_points = np.einsum('ab,bcd->acd',np.array(transform) , np.meshgrid(h_edges, l_edges))

    # pcolormesh(x_edges, y_edges, C) where C has shape (len(y)-1, len(x)-1)
    mesh = ax.pcolormesh(
        h_edges, l_edges,
        S2d.T,
        norm=norm, cmap="inferno", shading="auto",
    )
    plt.colorbar(mesh, ax=ax, label=f"{dataset} / site")

    ax.set_xlabel("$h$  (r.l.u.)")
    ax.set_ylabel("$l$  (r.l.u.)")
    ax.set_title(f"$S(h,h,l)$  [{dataset}],  $T = {T:.4g}$")
    ax.set_aspect("equal")


def main():
    args = parse_args()

    T_list, n_samples, data, n_spins, k_dims = read_ssf(args.filename, args.dataset)
    n_T = len(T_list)

    e0 = np.array(args.e0, dtype=int)
    e1 = np.array(args.e1, dtype=int)

    S_hhl, axis_0, axis_1 = extract_plane(data, n_spins, k_dims, e0, e1)

    if args.all:
        ncols = min(4, n_T)
        nrows = (n_T + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
        axes = np.array(axes).flatten()
        for i in range(n_T):
            plot_panel(axes[i], S_hhl[i], axis_0, axis_1,
                       T_list[i], args.dataset, args.log, args.clim)
        for i in range(n_T, len(axes)):
            axes[i].set_visible(False)
    else:
        t_idx = args.T
        if t_idx < -n_T or t_idx >= n_T:
            print(
                f"error: --T {t_idx} out of range for {n_T} temperature(s) "
                f"[−{n_T}, {n_T-1}]",
                file=sys.stderr,
            )
            sys.exit(1)
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        plot_panel(ax, S_hhl[t_idx], axis_0, axis_1,
                   T_list[t_idx], args.dataset, args.log, args.clim)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
