#!/usr/bin/env python3

import re
import sys
import os.path
import numpy as np
import h5py

# Usage: acc_heat_capacity ../tmp/run_L8_p0.050_Jzz1.000_Jxx0.100_Jyy0.100_s?.h5
#
# Accumulates per-seed thermal variances across disorder seeds.
# Output file has the same name with the _sN part removed.
#
# Output datasets (all length = n_temperatures):
#   T_list    — temperature points
#   E         — Σ_k ⟨E⟩_k          (divide by n_samples for disorder mean)
#   E_sq      — Σ_k ⟨E⟩_k²         (for SE on mean energy)
#   var       — Σ_k Var_k(E)        (divide by n_samples for disorder-mean thermal variance)
#   var_sq    — Σ_k Var_k(E)²       (for SE on heat capacity)
#   n_samples — K (number of seeds, not number of MC sweeps)


def load_raw(fname):
    """Return T_list, E, E2, n_samples, n_spins without dividing by n_samples."""
    with h5py.File(fname, "r") as f:
        g = f["/energy"]
        assert isinstance(g, h5py.Group)
        T       = np.array(g["T_list"])
        E       = np.array(g["E"])
        E2      = np.array(g["E2"])
        n       = np.array(g["n_samples"])
        n_spins = int(np.array(f["/geometry/n_spins"]))
    return T, E, E2, n, n_spins


def output_name(first_file):
    """Strip _sN from the filename to get the accumulated output name."""
    base = os.path.basename(first_file)
    out_base = re.sub(r"_s\d+", "", base)
    if out_base == base:
        raise ValueError(f"Could not find _sN seed tag in filename: {base}")
    return os.path.join(os.path.dirname(first_file), out_base)


def main(fnames):
    if not fnames:
        print("Usage: acc_heat_capacity FILE_s0.h5 FILE_s1.h5 ...")
        sys.exit(1)

    T_ref, E0, E20, n0, n_spins0 = load_raw(fnames[0])
    em  = E0 / n0
    var = E20 / n0 - em**2        # thermal variance for seed 0

    e_sum      = em.copy()
    e_sq_sum   = em**2
    var_sum    = var.copy()
    var_sq_sum = var**2
    n_spins_sum = n_spins0
    n_seeds    = 1

    for fname in fnames[1:]:
        T, E, E2, n, n_spins = load_raw(fname)
        if not np.allclose(T, T_ref):
            raise ValueError(f"T_list mismatch in {fname}")
        em  = E / n
        var = E2 / n - em**2
        e_sum       += em
        e_sq_sum    += em**2
        var_sum     += var
        var_sq_sum  += var**2
        n_spins_sum += n_spins
        n_seeds     += 1

    out = output_name(fnames[0])
    K = np.full(len(T_ref), n_seeds, dtype=np.uint64)
    with h5py.File(out, "w") as f:
        g = f.create_group("energy")
        g.create_dataset("T_list",    data=T_ref)
        g.create_dataset("E",         data=e_sum)
        g.create_dataset("E_sq",      data=e_sq_sum)
        g.create_dataset("var",       data=var_sum)
        g.create_dataset("var_sq",    data=var_sq_sum)
        g.create_dataset("n_samples", data=K)
        gg = f.create_group("geometry")
        gg.create_dataset("n_spins",  data=n_spins_sum / n_seeds)

    print(f"Accumulated {len(fnames)} seeds → {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
