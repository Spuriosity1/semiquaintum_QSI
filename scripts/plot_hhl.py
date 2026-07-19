#!/usr/bin/env python3
"""
plot_hhl.py — Plot S(q) in the (h,h,l) plane from sq_pyrochlore HDF5 output.

Unlike plot_ssf.py, this script:
  • evaluates S(q) at arbitrary q on the HHL plane (not restricted to the
    first cubic BZ) by folding q back to the BZ for the correlator lookup
    while keeping the full extended-zone phase for sublattice sums;
  • derives the reciprocal lattice matrix B from the file's k_dims (no
    hardcoded L=8 assumption);
  • covers ±2 cubic r.l.u. in both h and l, which spans the full FCC BZ.

Usage:
    python scripts/plot_hhl.py output.h5
    python scripts/plot_hhl.py output.h5 --dataset Szz --log
    python scripts/plot_hhl.py output.h5 --T -1 --save hhl.png
    python scripts/plot_hhl.py output.h5 --all --save all_T.png
"""

import argparse
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

# a_cubic in lattice-integer units: each conventional cubic cell spans 8 steps
_A_CUBIC_INT = 8


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot SSF S(q) in the (h,h,l) plane from sq_pyrochlore HDF5 output."
    )
    p.add_argument("filename", help="HDF5 output file from sq_pyrochlore")
    p.add_argument(
        "--dataset", default="Szz",
        help="Szz (local-axis Ising, default), Sqq (neutron), or Spm (spin-flip)",
    )
    p.add_argument(
        "--T", type=int, default=0, metavar="INDEX",
        help="Temperature index to plot (default: 0 = lowest T; negative wraps)",
    )
    p.add_argument("--all", action="store_true", help="Tile all temperatures")
    p.add_argument("--log", action="store_true", help="Logarithmic colour scale")
    p.add_argument(
        "--clim", nargs=2, type=float, metavar=("VMIN", "VMAX"),
        help="Fix colour scale limits",
    )
    p.add_argument("--title", help="Plot title (default: filename)")
    p.add_argument("--save", metavar="FILE", help="Save figure to FILE")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_ssf(filename):
    """Read SSF correlator; always returns corr shape (n_T, n_sl, n_sl, n_kpoints).

    Handles two formats:
      old: dataset 'corr'        shape (n_T, n_sl, n_sl, n_kpoints, 2)
      new: dataset 'static_corr' shape (n_seeds, n_T, n_kpoints, n_sl, n_sl, 2)
           → averaged over seeds, transposed to (n_T, n_sl, n_sl, n_kpoints)
    """
    with h5py.File(filename, "r") as f:
        grp = f["/ssf"]
        T_list    = grp["T_list"][:]
        n_samples = grp["n_samples"][:]
        sl_pos    = grp["sl_positions"][:]
        k_dims    = tuple(int(d) for d in grp.attrs["k_dims"])
        n_spins   = int(grp.attrs["n_spins"]) if "n_spins" in grp.attrs else 0

        if "static_corr" in grp:
            raw = grp["static_corr"][:]              # (n_seeds, n_T, n_k, n_sl, n_sl, 2)
            c   = (raw[..., 0] + 1j * raw[..., 1]).mean(axis=0)  # (n_T, n_k, n_sl, n_sl)
            corr = c.transpose(0, 2, 3, 1)           # (n_T, n_sl, n_sl, n_k)
        else:
            raw  = grp["corr"][:]                    # (n_T, n_sl, n_sl, n_k, 2)
            corr = raw[..., 0] + 1j * raw[..., 1]   # (n_T, n_sl, n_sl, n_k)
            # corr is stored as a sum over MC samples; normalise to per-sample mean
            corr /= n_samples[:, None, None, None]

        if n_spins == 0:
            n_spins = int(sl_pos.shape[0])           # fallback: sublattice count

    return T_list, n_samples, corr, sl_pos, n_spins, k_dims


def read_tcm(filename):
    with h5py.File(filename, "r") as f:
        grp = f["/transverse_corr"]
        T_list          = grp["T_list"][:]
        n_samples       = grp["n_samples"][:]
        corr            = grp["corr"][:]
        disp_vectors    = grp["disp_vectors"][:]
        n_pairs         = grp["n_pairs_per_sample"][:]
        n_quantum_spins = int(grp.attrs["n_quantum_spins"])
    return T_list, n_samples, corr, disp_vectors, n_pairs, n_quantum_spins


def make_B(k_dims):
    """Reciprocal lattice matrix for a supercell of L×L×L conventional cubic cells.

    Integer real-space coordinates are in units of a_cubic / 8, so the
    physical reciprocal lattice step is 2π / (8 * L).
    """
    Lx, Ly, Lz = k_dims
    return 2 * np.pi * np.diag([1.0 / (_A_CUBIC_INT * Lx),
                                 1.0 / (_A_CUBIC_INT * Ly),
                                 1.0 / (_A_CUBIC_INT * Lz)])


# ---------------------------------------------------------------------------
# Structure factor contractions at arbitrary q-lists
# ---------------------------------------------------------------------------

def contract_at_qlist(corr, k_dims, B, sl_pos, q_vecs):
    """Evaluate S^{zz}(q) at arbitrary q-vectors.

    Folds each q to the cubic BZ for the correlator lookup, but uses the
    full (extended-zone) q for the sublattice phase factors.

    Parameters
    ----------
    corr   : (n_T, n_sl, n_sl, n_kpoints) complex
    k_dims : (Lx, Ly, Lz)
    B      : (3, 3) diagonal reciprocal lattice matrix  (q = K @ B.T)
    sl_pos : (n_sl, 3) integer sublattice positions
    q_vecs : (n_q, 3) physical q-vectors (same units as B produces)

    Returns
    -------
    (n_T, n_q) real
    """
    Lx, Ly, Lz = k_dims

    # Map physical q to integer K, fold into [0, L)
    K_float = q_vecs / np.diag(B)                              # (n_q, 3)
    K_fold  = np.round(K_float).astype(int) % np.array([Lx, Ly, Lz])
    k_idx   = K_fold[:, 0] * Ly * Lz + K_fold[:, 1] * Lz + K_fold[:, 2]

    # Sublattice phase at full (extended-zone) q
    dr    = sl_pos[:, None, :].astype(float) - sl_pos[None, :, :]  # (n_sl, n_sl, 3)
    arg   = np.einsum('qi,mni->qmn', q_vecs, dr)                   # (n_q, n_sl, n_sl)
    phase = np.exp(1j * arg)                                        # (n_q, n_sl, n_sl)

    # Gather correlator slice for each q-point, then contract
    corr_sel = corr[:, :, :, k_idx]  # (n_T, n_sl, n_sl, n_q)
    return np.real(np.einsum('qmn,tmnq->tq', phase, corr_sel))     # (n_T, n_q)


_sl_z_axis = np.array(
    [[-1., -1., -1.], [-1., 1., 1.], [1., -1., 1.], [1., 1., -1.]]
) / np.sqrt(3)


def contract_sperp_at_qlist(corr, k_dims, B, sl_pos, q_vecs):
    """Evaluate neutron S_perp(q) at arbitrary q-vectors.

    Applies the transverse projector (I − q̂q̂) with the local-axis weights
    ẑ_μ · P(q) · ẑ_ν, using the full q for both phase and projector.
    """
    Lx, Ly, Lz = k_dims

    # Fold to BZ
    K_float = q_vecs / np.diag(B)
    K_fold  = np.round(K_float).astype(int) % np.array([Lx, Ly, Lz])
    k_idx   = K_fold[:, 0] * Ly * Lz + K_fold[:, 1] * Lz + K_fold[:, 2]

    # Transverse projector per q-point
    q_norm = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    q_norm = np.where(q_norm == 0, 1.0, q_norm)
    q_hat  = q_vecs / q_norm                                        # (n_q, 3)
    P      = np.eye(3)[None] - q_hat[:, :, None] * q_hat[:, None, :]  # (n_q, 3, 3)

    # Phase factors
    dr    = sl_pos[:, None, :].astype(float) - sl_pos[None, :, :]
    arg   = np.einsum('qi,mni->qmn', q_vecs, dr)
    phase = np.exp(1j * arg)                                        # (n_q, n_sl, n_sl)

    n_sl = sl_pos.shape[0]
    z      = np.array([_sl_z_axis[s % 4] for s in range(n_sl)])    # (n_sl, 3)
    weight = np.einsum('ma,qab,nb->qmn', z, P, z)                  # (n_q, n_sl, n_sl)

    corr_sel = corr[:, :, :, k_idx]                                 # (n_T, n_sl, n_sl, n_q)
    return np.real(np.einsum('qmn,qmn,tmnq->tq', phase, weight, corr_sel))


def compute_spm_at_qlist(corr, n_samples, disp_vectors, n_quantum_spins, q_vecs):
    """Evaluate S⁺⁻(q) at arbitrary q-vectors via direct Fourier sum."""
    arg   = np.einsum('qi,di->qd', q_vecs, disp_vectors.astype(float))
    phase = np.exp(1j * arg)                                        # (n_q, n_disp)
    C     = corr / n_samples[:, None]                               # (n_T, n_disp)
    S     = np.real(np.einsum('qd,td->tq', phase, C))
    return S / n_quantum_spins


# ---------------------------------------------------------------------------
# HHL grid
# ---------------------------------------------------------------------------

def make_hhl_qvecs(k_dims):
    """Build a dense (h,h,l) q-grid covering ±2 cubic r.l.u. in both axes.

    Returns
    -------
    q_vecs : (n_q, 3) physical q-vectors
    h_vals : (n_h,) array of h values in cubic r.l.u.
    l_vals : (n_l,) array of l values in cubic r.l.u.
    shape  : (n_h, n_l) for reshaping n_q back to 2D
    """
    Lx, _, Lz = k_dims
    # ±2 r.l.u. in h (= (h,h,0) direction) and l (= (0,0,l) direction)
    h_vals = np.arange(-2 * Lx, 2 * Lx) / Lx   # step 1/Lx, range [-2, 2)
    l_vals = np.arange(-2 * Lz, 2 * Lz) / Lz

    H, L_ax = np.meshgrid(h_vals, l_vals, indexing='ij')  # (n_h, n_l)

    # Physical q: 1 r.l.u. = 2π / a_cubic  (a_cubic = 8 integer units)
    scale = 2 * np.pi / _A_CUBIC_INT
    q_vecs = np.stack(
        [H.ravel() * scale, H.ravel() * scale, L_ax.ravel() * scale], axis=1
    )
    return q_vecs, h_vals, l_vals, H.shape


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_edges(axis):
    d = axis[1] - axis[0]
    return np.append(axis - d / 2, axis[-1] + d / 2)


def plot_panel(ax, S2d, h_vals, l_vals, T, dataset, log_scale, clim, label=None, cax=None):
    if clim is not None:
        vmin, vmax = clim
    else:
        flat     = S2d.ravel()
        flat_pos = flat[flat > 0]
        vmax = np.percentile(flat_pos, 99) if flat_pos.size else flat.max()
        vmin = flat_pos.min() if log_scale and flat_pos.size else flat.min()

    if log_scale:
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Wrap: repeat left column (h_min) at right and bottom row (l_min) at top
    # so the plot looks periodic at its boundaries.
    dh = h_vals[1] - h_vals[0]
    dl = l_vals[1] - l_vals[0]
    h_ext = np.append(h_vals, h_vals[-1] + dh)
    l_ext = np.append(l_vals, l_vals[-1] + dl)
    S2d_w = np.concatenate([S2d,      S2d[:1, :]  ], axis=0)  # left col at right
    S2d_w = np.concatenate([S2d_w,    S2d_w[:, :1]], axis=1)  # bottom row at top

    mesh = ax.pcolormesh(
        make_edges(h_ext), make_edges(l_ext), S2d_w.T,
        norm=norm, cmap="inferno", shading="auto",
        rasterized=True
    )
    cb_label = {
        "Szz": r"$\langle S^{z}S^{z} \rangle$ ",
        "Sqq": r"$S_\perp$",
        "Spm": r"$\langle S^{+}S^{-}\rangle$ ",
    }.get(dataset, f"{dataset} / site")

    if cax is not None:
        cb = plt.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.set_label(cb_label, fontsize=6, labelpad=1)
        cax.tick_params(labelsize=5, pad=1)
        cax.xaxis.set_ticks_position("bottom")

    ax.tick_params(which="both", direction="out", top=True, right=True, color='k')
    l = [-2,-1,0,1,2]
    ax.set_xticks(l, labels=[rf'$\overline{-x}$' if x<0 else f'${x}$' for x in l])
    ax.set_yticks(l, labels=[rf'$\overline{-x}$' if x<0 else f'${x}$' for x in l])

    ax.set_xlabel(r"$(h,h,*)$  (r.l.u.)")
    ax.set_ylabel(r"$(*,*,l)$  (r.l.u.)")


    if label is None:
        label = {"Szz": r"$S^{zz}$", "Sqq": r"$S_\perp$", "Spm": r"$S^{+-}$"}.get(dataset, dataset)
        label = f"{label}  ($T = {T:.4g}$)"

    print("Using label: ",label)
    ax.set_title(label)
    ax.set_aspect("equal")

    return mesh


def main():
    args = parse_args()


    title=args.title if hasattr(args, "title") else args.filename


    if args.dataset == 'Spm':
        T_list, n_samples, corr_tcm, disp_vectors, n_pairs, n_quantum_spins = \
            read_tcm(args.filename)
        with h5py.File(args.filename, "r") as f:
            k_dims = tuple(int(d) for d in f["/ssf"].attrs["k_dims"])
        B = make_B(k_dims)
        q_vecs, h_vals, l_vals, grid_shape = make_hhl_qvecs(k_dims)
        data_flat = compute_spm_at_qlist(
            corr_tcm, n_samples, disp_vectors, n_quantum_spins, q_vecs
        )
        norm_factor = 1
    else:
        T_list, n_samples, corr, sl_pos, n_spins, k_dims = read_ssf(args.filename)
        B = make_B(k_dims)
        q_vecs, h_vals, l_vals, grid_shape = make_hhl_qvecs(k_dims)
        if args.dataset == 'Szz':
            data_flat = contract_at_qlist(corr, k_dims, B, sl_pos, q_vecs)
        elif args.dataset == 'Sqq':
            data_flat = contract_sperp_at_qlist(corr, k_dims, B, sl_pos, q_vecs)
        else:
            raise ValueError("--dataset must be Szz, Sqq, or Spm")
        norm_factor = n_spins

    # Reshape and normalise.
    # n_T comes from data_flat (corr may store fewer temps than T_list covers).
    n_T = data_flat.shape[0]
    T_list = T_list[:n_T]                                          # trim to match
    S_hhl = data_flat.reshape(n_T, *grid_shape) / norm_factor  # (n_T, n_h, n_l)

    if args.all:
        ncols = min(4, n_T)
        nrows = (n_T + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.7 * nrows))
        axes = np.array(axes).flatten()
        for i in range(n_T):
            plot_panel(axes[i], S_hhl[i], h_vals, l_vals,
                       T_list[i], args.dataset, args.log, args.clim, label=title)
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
        fig, ax = plt.subplots(figsize=(4, 3.5))
        plot_panel(ax, S_hhl[t_idx], h_vals, l_vals,
                   T_list[t_idx], args.dataset, args.log, args.clim, label=title)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
