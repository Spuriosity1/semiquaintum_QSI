#!/usr/bin/env python3
"""
plot_ssf.py — Plot the static structure factor S(q) in the (h,h,l) plane.

Reads the /ssf or /transverse_corr group from sq_pyrochlore HDF5 output.
Q-points are expressed in conventional cubic reciprocal lattice units (r.l.u.),
where (1,0,0) = 2π/a_cubic.

Usage examples:
    python scripts/plot_ssf.py output.h5
    python scripts/plot_ssf.py output.h5 --dataset Szz --log
    python scripts/plot_ssf.py output.h5 --dataset Spm
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
        "--dataset", default="Szz",
        help="Which structure factor to plot: Szz (local-axis Ising, default), "
             "Sqq (neutron), or Spm (spin-flip ⟨S⁺S⁻⟩ from quantum clusters)",
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
            help="second vector to extract S(q) along (integers)",
            default=(0,1,0))
    p.add_argument(
        "--save", metavar="FILE",
        help="Save figure to FILE instead of opening an interactive window",
    )
    return p.parse_args()


def read_ssf(filename):
    """Read raw correlator data from the /ssf HDF5 group.

    Returns
    -------
    T_list    : (n_T,) float
    n_samples : (n_T,) int
    corr      : (n_T, n_sl, n_sl, n_kpoints) complex128
    sl_pos    : (n_sl, 3) int64  — sublattice positions in lattice integer coords
    n_spins   : int
    k_dims    : (Lx, Ly, Lz) tuple
    """
    with h5py.File(filename, "r") as f:
        grp = f["/ssf"]
        T_list    = grp["T_list"][:]        # (n_T,)
        n_samples = grp["n_samples"][:]     # (n_T,)
        raw       = grp["corr"][:]          # (n_T, n_sl, n_sl, n_kpoints, 2)
        sl_pos    = grp["sl_positions"][:]  # (n_sl, 3)
        n_spins   = int(grp.attrs["n_spins"])
        k_dims    = tuple(int(d) for d in grp.attrs["k_dims"])
    corr = raw[..., 0] + 1j * raw[..., 1]  # (n_T, n_sl, n_sl, n_kpoints)
    return T_list, n_samples, corr, sl_pos, n_spins, k_dims


def read_recip_latvecs(filename):
    with h5py.File(filename, "r") as f:
        B = np.array(f["/geometry/recip_vectors"][:])  # (3, 3), q = B @ K_centered
    return B


def read_tcm(filename):
    """Read real-space spin-flip correlator from the /transverse_corr HDF5 group.

    Returns
    -------
    T_list          : (n_T,) float
    n_samples       : (n_T,) int
    corr            : (n_T, n_disp) float  — unnormalised sum over samples
    disp_vectors    : (n_disp, 3) int64    — Δr = r_j - r_i in lattice integer coords
    n_pairs         : (n_disp,) float      — mean pairs per sample for each displacement
    n_quantum_spins : int
    """
    with h5py.File(filename, "r") as f:
        grp = f["/transverse_corr"]
        T_list          = grp["T_list"][:]
        n_samples       = grp["n_samples"][:]
        corr            = grp["corr"][:]
        disp_vectors    = grp["disp_vectors"][:]
        n_pairs         = grp["n_pairs_per_sample"][:]
        n_quantum_spins = int(grp.attrs["n_quantum_spins"])
    return T_list, n_samples, corr, disp_vectors, n_pairs, n_quantum_spins


def compute_spm(corr, n_samples, disp_vectors, n_quantum_spins, k_dims, B):
    """Fourier-transform the real-space spin-flip correlator to get S⁺⁻(q).

    Computes  S⁺⁻(q_k) = (1/N) Σ_d corr[T,d]/n_samples[T] · exp(i q_k · Δr_d)

    where the sum runs over all catalogued displacements d, q_k is the k-th
    reciprocal lattice point, and N = n_quantum_spins.

    Parameters
    ----------
    corr            : (n_T, n_disp) — accumulated, unnormalised ⟨S⁺S⁻⟩ sums
    n_samples       : (n_T,)
    disp_vectors    : (n_disp, 3) int
    n_quantum_spins : int
    k_dims          : (Lx, Ly, Lz)
    B               : (3, 3) reciprocal lattice matrix  (q = K @ B.T)

    Returns
    -------
    S : (n_T, n_k) real float
    """
    Lx, Ly, Lz = k_dims
    K = np.indices((Lx, Ly, Lz)).reshape(3, -1).T.astype(float)  # (n_k, 3)
    K[:, 0] -= Lx * (K[:, 0] > Lx // 2)
    K[:, 1] -= Ly * (K[:, 1] > Ly // 2)
    K[:, 2] -= Lz * (K[:, 2] > Lz // 2)
    q = K @ B.T  # (n_k, 3)

    # Phase: exp(i q_k · Δr_d), shape (n_k, n_disp)
    arg = np.einsum('ki,di->kd', q, disp_vectors.astype(float))
    phase = np.exp(1j * arg)

    # Sample-averaged correlator, shape (n_T, n_disp)
    C = corr / n_samples[:, None]

    # S⁺⁻(q) = Re Σ_d C[T,d] · exp(i q · Δr_d),  normalised per site
    S = np.real(np.einsum('kd,td->tk', phase, C))
    return S / n_quantum_spins


def _phase_weights(k_dims, B, sl_pos):
    """phase[k, mu, nu] = exp(i q(k) · (r_mu - r_nu))

    k_dims : (Lx, Ly, Lz)
    B      : (3,3) reciprocal lattice matrix; q = B @ K_centered
    sl_pos : (n_sl, 3) integer sublattice positions

    Returns shape (n_kpoints, n_sl, n_sl).
    """
    Lx, Ly, Lz = k_dims
    # k-point indices in row-major order matching lil2's flat_from_idx3
    K = np.indices((Lx, Ly, Lz)).reshape(3, -1).T.astype(float)  # (n_k, 3)
    # 0-centre: wrap K[a] > L[a]//2 to K[a] - L[a]
    K[:, 0] -= Lx * (K[:, 0] > Lx // 2)
    K[:, 1] -= Ly * (K[:, 1] > Ly // 2)
    K[:, 2] -= Lz * (K[:, 2] > Lz // 2)
    q = K @ B.T  # (n_k, 3)

    dr = sl_pos[:, None, :].astype(float) - sl_pos[None, :, :]  # (n_sl, n_sl, 3)
    arg = np.einsum('ki,mni->kmn', q, dr)  # (n_k, n_sl, n_sl)
    return np.exp(1j * arg)


def contract_szz(corr, k_dims, B, sl_pos):
    """Contract (n_T, n_sl, n_sl, n_kpoints) correlator → (n_T, n_kpoints) real.
    """
    phase = _phase_weights(k_dims, B, sl_pos)  # (n_k, n_sl, n_sl)

    # result[t, k] = Re Σ_{μν} weight[k,μ,ν] · corr[t,μ,ν,k]
    return np.real(np.einsum('kmn,tmnk->tk', phase, corr))  # (n_T, n_k)



_sl_z_axis = np.array(
        [[ -1., -1., -1.], [-1., 1., 1.], [1., -1., 1.], [1., 1., -1.]]
        )/np.sqrt(3)

def contract_sperp(corr, k_dims, B, sl_pos):
    """Contract (n_T, n_sl, n_sl, n_kpoints) correlator → (n_T, n_kpoints) real.

    Computes the neutron S_perp(q) = Σ_{μν} (ẑ_μ·ẑ_ν − (q̂·ẑ_μ)(q̂·ẑ_ν)) exp(iq·Δr_{μν}) C_{μν}(q)
    with the transverse projector (I − q̂q̂) evaluated at each k-point individually.
    """
    Lx, Ly, Lz = k_dims
    K = np.indices((Lx, Ly, Lz)).reshape(3, -1).T.astype(float)  # (n_k, 3)
    K[:, 0] -= Lx * (K[:, 0] > Lx // 2)
    K[:, 1] -= Ly * (K[:, 1] > Ly // 2)
    K[:, 2] -= Lz * (K[:, 2] > Lz // 2)
    q_vecs = K @ B.T                                               # (n_k, 3)
    q_norm = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    q_norm = np.where(q_norm == 0, 1.0, q_norm)                   # guard Γ point
    q_hat = q_vecs / q_norm                                        # (n_k, 3)

    # Transverse projector per k-point: P[k,a,b] = δ_ab − q̂_k^a q̂_k^b
    P = np.eye(3)[None] - q_hat[:, :, None] * q_hat[:, None, :]   # (n_k, 3, 3)

    phase = _phase_weights(k_dims, B, sl_pos)                      # (n_k, n_sl, n_sl)
    n_sl = sl_pos.shape[0]

    # Local axes for each sublattice
    z = np.array([_sl_z_axis[s % 4] for s in range(n_sl)])        # (n_sl, 3)
    # weight[k,μ,ν] = ẑ_μ · P(k) · ẑ_ν
    weight = np.einsum('ma,kab,nb->kmn', z, P, z)                  # (n_k, n_sl, n_sl)

    return np.real(np.einsum('kmn,kmn,tmnk->tk', phase, weight, corr))  # (n_T, n_k)


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

    print(k_dims)

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


def format_ax_label(vector, letter, sep=' '):
    s = []
    for v in vector:
        if int(v) == 0:
            s.append('0')
        elif int(v) == 1:
            s.append( letter)
        elif int(v) == -1:
            s.append('-' + letter)
        else:
            s.append(str(int(v)) + letter)
    return sep.join(s)

def plot_panel(ax, S2d, axis_0, axis_1, e0, e1, T, dataset, log_scale, clim):
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

    h_edges = make_edges(axis_0)
    l_edges = make_edges(axis_1)

    # pcolormesh(x_edges, y_edges, C) where C has shape (len(y)-1, len(x)-1)
    mesh = ax.pcolormesh(
        h_edges, l_edges,
        S2d.T,
        norm=norm, cmap="inferno", shading="auto",
    )
    cb_label = {"Szz": r"$S^{zz}$ / site", "Sqq": r"$S_\perp$ / site",
                "Spm": r"$S^{+-}$ / site"}.get(dataset, f"{dataset} / site")
    plt.colorbar(mesh, ax=ax, label=cb_label)

    ax.set_xlabel("$(%s)$  (r.l.u.)" % format_ax_label(e0, 'h', '~'))
    ax.set_ylabel("$(%s)$  (r.l.u.)" % format_ax_label(e1, 'l', '~'))
    label = {"Szz": r"$S^{zz}$", "Sqq": r"$S_\perp$", "Spm": r"$S^{+-}$"}.get(dataset, dataset)
    ax.set_title(f"{label}  ($T = {T:.4g}$)")
    ax.set_aspect("equal")


def main():
    args = parse_args()

    B = read_recip_latvecs(args.filename)
    e0 = np.array(args.e0, dtype=int)
    e1 = np.array(args.e1, dtype=int)

    if args.dataset == 'Spm':
        T_list, n_samples, corr, disp_vectors, n_pairs, n_quantum_spins = \
            read_tcm(args.filename)
        with h5py.File(args.filename, "r") as f:
            k_dims = tuple(int(d) for d in f["/ssf"].attrs["k_dims"])
        # compute_spm already normalises per quantum spin; pass norm_factor=1
        data = compute_spm(corr, n_samples, disp_vectors, n_quantum_spins, k_dims, B)
        norm_factor = 1
    else:
        T_list, n_samples, corr, sl_pos, n_spins, k_dims = read_ssf(args.filename)
        if args.dataset == 'Szz':
            data = contract_szz(corr, k_dims, B, sl_pos)  # (n_T, n_k)
        elif args.dataset == 'Sqq':
            data = contract_sperp(corr, k_dims, B, sl_pos)
        else:
            raise ValueError("--dataset must be Szz, Sqq, or Spm")
        norm_factor = n_spins

    n_T = len(T_list)
    S_hhl, axis_0, axis_1 = extract_plane(data, norm_factor, k_dims, e0, e1)

    if args.all:
        ncols = min(4, n_T)
        nrows = (n_T + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
        axes = np.array(axes).flatten()
        for i in range(n_T):
            plot_panel(axes[i], S_hhl[i], axis_0, axis_1, args.e0, args.e1,
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
        plot_panel(ax, S_hhl[t_idx], axis_0, axis_1, args.e0, args.e1,
                   T_list[t_idx], args.dataset, args.log, args.clim)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
