#!/usr/bin/env python3
"""
acc_runs.py — Merge sq_pyrochlore output files across MC and disorder seeds.

Groups input files by disorder seed (_dsN), merges MC seeds (_msM) within
each group for energy, SSF, and transverse-correlator observables, then
averages energy and SSF over disorder seeds.

Writes:
  _dsX_mergeN.mavg.h5  — one per disorder seed (N = number of MC seeds merged)
  _mergeK.davg.h5      — disorder average over K seeds

Energy (/energy group):
  Per-disorder: T_list, E (sum), E2 (corrected sum), n_samples, var_sq, n_mc_seeds
  Disorder-avg: T_list, E, E_sq, var, var_sq, n_disorder

SSF (/ssf group, if present):
  Per-disorder: T_list, n_samples (sum), corr (sum), sl_positions, attrs n_spins/k_dims
  Disorder-avg: T_list, n_disorder, corr (sum of per-seed means), sl_positions,
                attrs n_spins/k_dims
  Note: corr/n_disorder gives the disorder-averaged mean correlator.

Transverse correlator (/transverse_corr group, if present):
  Per-disorder only (displacement catalogs differ between disorder seeds):
    T_list, n_samples (sum), corr (sum), disp_vectors, sublat_i, sublat_j,
    n_pairs_per_sample, attr n_quantum_spins
"""

import re
import sys
import os.path
import numpy as np
import h5py
from collections import defaultdict


# ── file validation ────────────────────────────────────────────────────────────

def check_file(fname):
    try:
        with h5py.File(fname, "r") as f:
            pass
    except OSError as e:
        print(f'Problem with file "{fname}": {e}', file=sys.stderr)
        return False
    return True


# ── filename parsing ───────────────────────────────────────────────────────────

_COMPATIBLE_TAGS = {
    "L":   (r"_L(\d+)[_.]",               int),
    "p":   (r"_p(\d+\.?\d*)[_.]",         float),
    "Jzz": (r"_Jzz([-\+\d]+\.?\d*)[_.]",  float),
    "Jxx": (r"_Jxx([-\+\d]+\.?\d*)[_.]",  float),
    "Jyy": (r"_Jyy([-\+\d]+\.?\d*)[_.]",  float),
}


def parse_tags(fname):
    base = os.path.basename(fname)
    tags = {}
    for name, (pattern, cast) in _COMPATIBLE_TAGS.items():
        m = re.search(pattern, base)
        if m is None:
            raise ValueError(f"No _{name}<N> tag in: {base}")
        tags[name] = cast(m.group(1))
    return tags


def parse_dseed(fname):
    m = re.search(r"_ds(\d+)[_.]", os.path.basename(fname))
    if m is None:
        raise ValueError(f"No _dsN tag in: {os.path.basename(fname)}")
    return int(m.group(1))


def check_filename_compatibility(fnames):
    ref = parse_tags(fnames[0])
    for fname in fnames[1:]:
        tags = parse_tags(fname)
        mismatches = [f"{k}: {ref[k]} vs {tags[k]}" for k in ref if ref[k] != tags[k]]
        if mismatches:
            raise ValueError(
                f"Incompatible tags in {os.path.basename(fname)}: "
                + ", ".join(mismatches)
            )


def per_disorder_name(representative_file, n_merged):
    base = os.path.basename(representative_file)
    out = re.sub(r"_ms\d+", f"_merge{n_merged}", base)
    out = re.sub(r"\.h5$", ".mavg.h5", out)
    if out == base:
        raise ValueError(f"No _msN tag in: {base}")
    return os.path.join(os.path.dirname(representative_file), out)


def disorder_avg_name(representative_file, n_disorder):
    base = os.path.basename(representative_file)
    out = re.sub(r"_ds\d+", f"_merge{n_disorder}", base)
    out = re.sub(r"_ms\d+", "", out)
    out = re.sub(r"\.h5$", ".davg.h5", out)
    if out == base:
        raise ValueError(f"No _dsN tag in: {base}")
    return os.path.join(os.path.dirname(representative_file), out)


# ── energy ─────────────────────────────────────────────────────────────────────

def load_energy(fname):
    """Return T_list, E_sum, E2_sum, n_samples."""
    with h5py.File(fname, "r") as f:
        g = f["/energy"]
        T  = np.array(g["T_list"])
        E  = np.array(g["E"])
        E2 = np.array(g["E2"])
        n  = np.array(g["n_samples"])
    return T, E, E2, n


def load_n_spins(fname):
    with h5py.File(fname, "r") as f:
        return float(np.array(f["/geometry/n_spins"]))


# ── SSF ────────────────────────────────────────────────────────────────────────

def has_ssf(fname):
    with h5py.File(fname, "r") as f:
        return "/ssf" in f


def load_ssf(fname):
    """Return T_list, n_samples, corr, sl_positions, n_spins, k_dims."""
    with h5py.File(fname, "r") as f:
        g = f["/ssf"]
        assert isinstance(g, h5py.Group)
        T      = np.array(g["T_list"])       # (n_T,)
        n      = np.array(g["n_samples"])    # (n_T,)
        corr   = np.array(g["corr"])         # (n_T, n_sl, n_sl, n_k, 2)
        sl_pos = np.array(g["sl_positions"]) # (n_sl, 3)
        n_spins = int(g.attrs["n_spins"])
        k_dims  = tuple(int(d) for d in g.attrs["k_dims"])
    return T, n, corr, sl_pos, n_spins, k_dims


def check_ssf_compatibility(fnames, T_ref):
    """Raise if SSF metadata is inconsistent across files."""
    _, _, corr0, sl_pos0, _, k_dims0 = load_ssf(fnames[0])
    T0, *_ = load_ssf(fnames[0])
    if not np.allclose(T0, T_ref):
        raise ValueError(f"SSF T_list differs from energy T_list in {fnames[0]}")
    for fname in fnames[1:]:
        T, _, corr, sl_pos, _, k_dims = load_ssf(fname)
        if not np.allclose(T, T_ref):
            raise ValueError(f"SSF T_list mismatch in {fname}")
        if not np.array_equal(sl_pos, sl_pos0):
            raise ValueError(f"SSF sl_positions mismatch in {fname}")
        if k_dims != k_dims0:
            raise ValueError(f"SSF k_dims mismatch in {fname}: {k_dims} vs {k_dims0}")
        if corr.shape[1:] != corr0.shape[1:]:
            raise ValueError(f"SSF corr shape mismatch in {fname}")


# ── transverse correlator ──────────────────────────────────────────────────────

def has_tcm(fname):
    with h5py.File(fname, "r") as f:
        return "/transverse_corr" in f


def load_tcm(fname):
    """Return T_list, n_samples, corr, disp_vectors, n_pairs_per_sample, n_quantum_spins."""
    with h5py.File(fname, "r") as f:
        g = f["/transverse_corr"]
        assert isinstance(g, h5py.Group)
        T      = np.array(g["T_list"])               # (n_T,)
        n      = np.array(g["n_samples"])            # (n_T,)
        corr   = np.array(g["corr"])                 # (n_T, n_disp)
        disp   = np.array(g["disp_vectors"])         # (n_disp, 3)
        pairs  = np.array(g["n_pairs_per_sample"])   # (n_disp,)
        n_qsp  = int(g.attrs["n_quantum_spins"])
    return T, n, corr, disp, pairs, n_qsp


def check_tcm_compatibility(fnames, T_ref):
    """Within a disorder group: raise if TCM metadata is inconsistent."""
    T0, _, corr0, disp0, _, nq0 = load_tcm(fnames[0])
    if not np.allclose(T0, T_ref):
        raise ValueError(f"TCM T_list differs from energy T_list in {fnames[0]}")
    for fname in fnames[1:]:
        T, _, corr, disp, _, nq = load_tcm(fname)
        if not np.allclose(T, T_ref):
            raise ValueError(f"TCM T_list mismatch in {fname}")
        if not np.array_equal(disp, disp0):
            raise ValueError(f"TCM disp_vectors mismatch in {fname}")
        if nq != nq0:
            raise ValueError(f"TCM n_quantum_spins mismatch in {fname}: {nq} vs {nq0}")


# ── HDF5 writers ───────────────────────────────────────────────────────────────

def write_ssf_group(hf, group_name, T_list, n_samples, corr, sl_positions, n_spins, k_dims):
    g = hf.create_group(group_name)
    g.create_dataset("T_list",       data=T_list)
    g.create_dataset("n_samples",    data=n_samples)
    g.create_dataset("corr",         data=corr)
    g.create_dataset("sl_positions", data=sl_positions)
    g.attrs["n_spins"] = n_spins
    g.attrs["k_dims"]  = list(k_dims)


def write_ssf_davg_group(hf, group_name, T_list, corr_sum, n_disorder,
                         sl_positions, n_spins, k_dims):
    """Disorder-averaged SSF: corr_sum = Σ_ds (corr_ds / n_samples_ds).
    Divide by n_disorder to recover the disorder-averaged mean correlator.
    """
    g = hf.create_group(group_name)
    g.create_dataset("T_list",       data=T_list)
    g.create_dataset("n_disorder",   data=np.uint64(n_disorder))
    g.create_dataset("corr",         data=corr_sum)
    g.create_dataset("sl_positions", data=sl_positions)
    g.attrs["n_spins"] = n_spins
    g.attrs["k_dims"]  = list(k_dims)


def write_tcm_group(hf, group_name, T_list, n_samples, corr, disp_vectors,
                    n_pairs_per_sample, n_quantum_spins):
    g = hf.create_group(group_name)
    g.create_dataset("T_list",             data=T_list)
    g.create_dataset("n_samples",          data=n_samples)
    g.create_dataset("corr",               data=corr)
    g.create_dataset("disp_vectors",       data=disp_vectors)
    g.create_dataset("n_pairs_per_sample", data=n_pairs_per_sample)
    g.attrs["n_quantum_spins"] = n_quantum_spins


# ── main ───────────────────────────────────────────────────────────────────────

def main(fnames):
    if not fnames:
        print(
            "Usage: acc_runs.py FILE_ds0_ms0.h5 [FILE_ds0_ms1.h5 ...] FILE_ds1_ms0.h5 ...",
            file=sys.stderr,
        )
        sys.exit(1)

    check_filename_compatibility(fnames)

    # Global energy T_list check
    T_ref = None
    for fname in fnames:
        T, *_ = load_energy(fname)
        if T_ref is None:
            T_ref = T
        elif not np.allclose(T, T_ref):
            raise ValueError(f"Energy T_list mismatch in {fname}")

    # SSF presence and global metadata check
    ssf_presence = {f: has_ssf(f) for f in fnames}
    n_with_ssf = sum(ssf_presence.values())
    if 0 < n_with_ssf < len(fnames):
        missing = [f for f, v in ssf_presence.items() if not v]
        print(f"Warning: {len(missing)} file(s) lack /ssf — SSF skipped for those.")
    ssf_files_all = [f for f in fnames if ssf_presence[f]]
    if ssf_files_all:
        check_ssf_compatibility(ssf_files_all, T_ref)
        _, _, _, sl_pos_ref, _, k_dims_ref = load_ssf(ssf_files_all[0])

    # Group by disorder seed
    groups = defaultdict(list)
    for fname in fnames:
        groups[parse_dseed(fname)].append(fname)

    disorder_E           = []
    disorder_var         = []
    disorder_n_spins     = []
    disorder_corr_ssf    = []   # per-seed mean SSF correlator (n_T, n_sl, n_sl, n_k, 2)
    disorder_ssf_n_spins = []   # n_spins attribute from /ssf per disorder seed

    for ds, files in sorted(groups.items()):
        if not check_file(files[0]):
            sys.exit(1)

        # ── energy merge ─────────────────────────────────────────────────────
        T_this, E_sum, E2_j, n_sum = load_energy(files[0])
        if not np.allclose(T_this, T_ref):
            raise ValueError(f"T_list mismatch in {files[0]}")
        within_sum = E2_j - E_sum**2 / n_sum
        var_j      = within_sum / n_sum
        var_sq_sum = var_j**2
        n_spins    = load_n_spins(files[0])

        for fname in files[1:]:
            if not check_file(fname):
                continue
            T, E_j, E2_j, n_j = load_energy(fname)
            if not np.allclose(T, T_ref):
                raise ValueError(f"T_list mismatch in {fname}")
            w_j         = E2_j - E_j**2 / n_j
            within_sum += w_j
            E_sum      += E_j
            n_sum      += n_j
            var_j       = w_j / n_j
            var_sq_sum += var_j**2
            n_spins     = load_n_spins(fname)

        K_mc         = len(files)
        E2_corrected = within_sum + E_sum**2 / n_sum
        e_mean       = E_sum / n_sum
        var          = within_sum / n_sum

        # ── SSF merge ─────────────────────────────────────────────────────────
        ds_ssf_files = [f for f in files if ssf_presence[f]]
        merged_ssf_corr = None
        merged_ssf_n    = None
        ssf_n_spins     = None

        if ds_ssf_files:
            _, n0, corr0, _, ssf_n_spins, _ = load_ssf(ds_ssf_files[0])
            merged_ssf_corr = corr0.copy()
            merged_ssf_n    = n0.copy()
            for fname in ds_ssf_files[1:]:
                _, n_j, corr_j, _, ssf_n_spins, _ = load_ssf(fname)
                merged_ssf_corr += corr_j
                merged_ssf_n    += n_j

        # ── TCM merge ─────────────────────────────────────────────────────────
        ds_tcm_files = [f for f in files if has_tcm(f)]
        merged_tcm_corr = None
        merged_tcm_n    = None
        tcm_disp        = None
        tcm_pairs       = None
        tcm_n_qsp       = None

        if ds_tcm_files:
            check_tcm_compatibility(ds_tcm_files, T_ref)
            _, n0, corr0, disp0, pairs0, nq0 = load_tcm(ds_tcm_files[0])
            merged_tcm_corr = corr0.copy()
            merged_tcm_n    = n0.copy()
            tcm_disp        = disp0
            tcm_pairs       = pairs0
            tcm_n_qsp       = nq0
            for fname in ds_tcm_files[1:]:
                _, n_j, corr_j, *_ = load_tcm(fname)
                merged_tcm_corr += corr_j
                merged_tcm_n    += n_j

        # ── write per-disorder file ───────────────────────────────────────────
        out = per_disorder_name(files[0], K_mc)
        with h5py.File(out, "w") as hf:
            g = hf.create_group("energy")
            g.create_dataset("T_list",     data=T_ref)
            g.create_dataset("E",          data=E_sum)
            g.create_dataset("E2",         data=E2_corrected)
            g.create_dataset("n_samples",  data=n_sum)
            g.create_dataset("var_sq",     data=var_sq_sum)
            g.create_dataset("n_mc_seeds", data=np.uint64(K_mc))
            gg = hf.create_group("geometry")
            gg.create_dataset("n_spins",   data=n_spins)

            if merged_ssf_corr is not None:
                write_ssf_group(hf, "ssf", T_ref, merged_ssf_n, merged_ssf_corr,
                                sl_pos_ref, ssf_n_spins, k_dims_ref)

            if merged_tcm_corr is not None:
                write_tcm_group(hf, "transverse_corr", T_ref, merged_tcm_n,
                                merged_tcm_corr, tcm_disp, tcm_pairs, tcm_n_qsp)

        print(f"  ds={ds}: {K_mc} MC seed(s) → {out}")

        disorder_E.append(e_mean)
        disorder_var.append(var)
        disorder_n_spins.append(n_spins)

        if merged_ssf_corr is not None:
            # Normalize: sum across T of mean correlator per disorder seed
            mean_corr = merged_ssf_corr / merged_ssf_n[:, None, None, None, None]
            disorder_corr_ssf.append(mean_corr)
            disorder_ssf_n_spins.append(ssf_n_spins)

    # ── disorder average ──────────────────────────────────────────────────────
    K       = len(disorder_E)
    E_arr   = np.array(disorder_E)    # (K, n_T)
    var_arr = np.array(disorder_var)  # (K, n_T)

    first_file = groups[min(groups)][0]
    out_avg = disorder_avg_name(first_file, K)
    with h5py.File(out_avg, "w") as hf:
        g = hf.create_group("energy")
        g.create_dataset("T_list",     data=T_ref)
        g.create_dataset("E",          data=E_arr.sum(axis=0))
        g.create_dataset("E_sq",       data=(E_arr**2).sum(axis=0))
        g.create_dataset("var",        data=var_arr.sum(axis=0))
        g.create_dataset("var_sq",     data=(var_arr**2).sum(axis=0))
        g.create_dataset("n_disorder", data=np.uint64(K))
        gg = hf.create_group("geometry")
        gg.create_dataset("n_spins",   data=disorder_n_spins)

        if disorder_corr_ssf:
            corr_sum    = np.sum(disorder_corr_ssf, axis=0)
            n_spins_ssf = int(np.round(np.mean(disorder_ssf_n_spins)))
            write_ssf_davg_group(hf, "ssf", T_ref, corr_sum, len(disorder_corr_ssf),
                                 sl_pos_ref, n_spins_ssf, k_dims_ref)

    print(f"Disorder average of {K} seed(s) → {out_avg}")


if __name__ == "__main__":
    main(sys.argv[1:])
