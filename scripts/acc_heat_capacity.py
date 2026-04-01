#!/usr/bin/env python3

import re
import sys
import os.path
import numpy as np
import h5py
from collections import defaultdict

# Usage: acc_heat_capacity run_L8_..._ds0_ms0.h5 run_L8_..._ds0_ms1.h5 run_L8_..._ds1_ms0.h5 ...
#
# Groups input files by disorder seed (_dsN), merges MC seeds (_msM) within each group,
# computes the thermal variance Var(e) = <e²> - <e>² per disorder realization (where
# e = E / <n_spins>, <n_spins> = (1-p)*16*L³), then averages over disorder seeds.
#
# Writes:
#   _dsX_mergeN.h5    — one per disorder seed (N = number of MC seeds merged)
#   _mergeK.davg.h5   — disorder average over K seeds
#
# Heat capacity C/N = Var(e) / T² is left to post-processing.
#
# Output datasets in /energy:
#   Per-disorder file (_dsX_mergeN.h5) — same format as raw simulation output:
#     T_list, E (sum), E2 (sum), n_samples
#   Disorder-average file (_mergeK.davg.h5) — intensive quantities normalised by <n_spins>:
#     T_list, E, E_sq, var, var_sq, n_disorder


def load_raw(fname):
    """Return T_list, E_sum, E2_sum, n_samples (raw MC sums, not divided by n)."""
    with h5py.File(fname, "r") as f:
        g = f["/energy"]
        assert isinstance(g, h5py.Group)
        T  = np.array(g["T_list"])
        E  = np.array(g["E"])
        E2 = np.array(g["E2"])
        n  = np.array(g["n_samples"])
    return T, E, E2, n


def parse_dseed(fname):
    m = re.search(r"_ds(\d+)[_.]", os.path.basename(fname))
    if m is None:
        raise ValueError(f"No _dsN tag in: {os.path.basename(fname)}")
    return int(m.group(1))


# Tags that must be identical across all input files.
_COMPATIBLE_TAGS = {
    "L":   (r"_L(\d+)[_.]",            int),
    "p":   (r"_p(\d+\.?\d*)[_.]",      float),
    "Jzz": (r"_Jzz(\d+\.?\d*)[_.]",   float),
    "Jxx": (r"_Jxx(\d+\.?\d*)[_.]",   float),
    "Jyy": (r"_Jyy(\d+\.?\d*)[_.]",   float),
}


def parse_tags(fname):
    """Return dict of physical tags parsed from filename."""
    base = os.path.basename(fname)
    tags = {}
    for name, (pattern, cast) in _COMPATIBLE_TAGS.items():
        m = re.search(pattern, base)
        if m is None:
            raise ValueError(f"No _{name}<N> tag in: {base}")
        tags[name] = cast(m.group(1))
    return tags


def check_compatibility(fnames):
    """Raise ValueError if any file has tags that differ from the first file."""
    ref = parse_tags(fnames[0])
    for fname in fnames[1:]:
        tags = parse_tags(fname)
        mismatches = [
            f"{k}: {ref[k]} vs {tags[k]}"
            for k in ref if ref[k] != tags[k]
        ]
        if mismatches:
            raise ValueError(
                f"Incompatible tags in {os.path.basename(fname)}: "
                + ", ".join(mismatches)
            )


def parse_L_p(fname):
    tags = parse_tags(fname)
    return tags["L"], tags["p"]


def expected_n_spins(fname):
    L, p = parse_L_p(fname)
    return (1.0 - p) * 16 * L**3


def per_disorder_name(representative_file, n_merged):
    """Keep _dsX, replace _msM with _mergeN."""
    base = os.path.basename(representative_file)
    out = re.sub(r"_ms\d+", f"_merge{n_merged}", base)
    if out == base:
        raise ValueError(f"No _msN tag in: {base}")
    return os.path.join(os.path.dirname(representative_file), out)


def disorder_avg_name(representative_file, n_disorder):
    """Strip _dsX and _msM; replace _dsX with _mergeN, change extension to .davg.h5."""
    base = os.path.basename(representative_file)
    out = re.sub(r"_ds\d+", f"_merge{n_disorder}", base)
    out = re.sub(r"_ms\d+", "", out)
    out = re.sub(r"\.h5$", ".davg.h5", out)
    if out == base:
        raise ValueError(f"No _dsN tag in: {base}")
    return os.path.join(os.path.dirname(representative_file), out)


def load_n_spins(fname):
    """Return N, the (mean) number of non-deleted spins in the realisation."""
    with h5py.File(fname, "r") as f:
        N = float(np.array(f["/geometry/n_spins"]))
    return N

def main(fnames):
    if not fnames:
        print("Usage: acc_heat_capacity FILE_ds0_ms0.h5 [FILE_ds0_ms1.h5 ...] FILE_ds1_ms0.h5 ...")
        sys.exit(1)

    check_compatibility(fnames)

    # Group by disorder seed
    groups = defaultdict(list)
    for fname in fnames:
        groups[parse_dseed(fname)].append(fname)

    T_ref        = None
    disorder_E   = []   # intensive <e> per disorder seed
    disorder_var = []   # intensive Var(e) per disorder seed

    for ds, files in sorted(groups.items()):
        T_this, E_sum, E2_j, n_sum = load_raw(files[0])
        if T_ref is None:
            T_ref = T_this
        elif not np.allclose(T_this, T_ref):
            raise ValueError(f"T_list mismatch for ds={ds}")

        # within_sum = Σ_j (E2_j - E_j²/N_j): sum of within-seed weighted variances.
        # Using this instead of E2_total avoids inflating the variance with between-seed
        # scatter when seeds are stuck in different metastable states at low T.
        within_sum = E2_j - E_sum**2 / n_sum


        n_spins = load_n_spins(fnames[0])

        # var_j for seed 0 (intensive), accumulated for SE estimate across MC seeds
        var_j      = within_sum / n_sum / n_spins**2
        var_sq_sum = var_j**2

        for fname in files[1:]:
            n_spins = load_n_spins(fnames[0])

            T, E_j, E2_j, n_j = load_raw(fname)
            if not np.allclose(T, T_ref):
                raise ValueError(f"T_list mismatch in {fname}")
            w_j         = E2_j - E_j**2 / n_j
            within_sum += w_j
            E_sum      += E_j
            n_sum      += n_j
            var_j       = w_j / n_j / n_spins**2
            var_sq_sum += var_j**2

        K_mc = len(files)

        # E2_corrected: stored so that E2/n - (E/n)² recovers within-seed variance.
        # For a single seed this equals the original E2 exactly.
        E2_corrected = within_sum + E_sum**2 / n_sum

        e_mean = E_sum  / n_sum / n_spins
        var    = within_sum / n_sum / n_spins**2

        # Write per-disorder file
        out = per_disorder_name(files[0], len(files))
        with h5py.File(out, "w") as f:
            g = f.create_group("energy")
            g.create_dataset("T_list",     data=T_ref)
            g.create_dataset("E",          data=E_sum)
            g.create_dataset("E2",         data=E2_corrected)
            g.create_dataset("n_samples",  data=n_sum)
            g.create_dataset("var_sq",     data=var_sq_sum)
            g.create_dataset("n_mc_seeds", data=np.uint64(K_mc))
            gg = f.create_group("geometry")
            gg.create_dataset("n_spins",   data=n_spins)
        print(f"  ds={ds}: {len(files)} MC seed(s) → {out}")

        disorder_E.append(e_mean)
        disorder_var.append(var)

    # Disorder average
    K       = len(disorder_E)
    E_arr   = np.array(disorder_E)    # (K, n_T)
    var_arr = np.array(disorder_var)  # (K, n_T)

    first_file = groups[min(groups)][0]
    out_avg = disorder_avg_name(first_file, K)
    with h5py.File(out_avg, "w") as f:
        g = f.create_group("energy")
        g.create_dataset("T_list",     data=T_ref)
        g.create_dataset("E",          data=E_arr.mean(axis=0))
        g.create_dataset("E_sq",       data=(E_arr**2).mean(axis=0))
        g.create_dataset("var",        data=var_arr.mean(axis=0))
        g.create_dataset("var_sq",     data=(var_arr**2).mean(axis=0))
        g.create_dataset("n_disorder", data=np.uint64(K))
        gg = f.create_group("geometry")
        gg.create_dataset("n_spins",   data=n_spins)
    print(f"Disorder average of {K} seed(s) → {out_avg}")


if __name__ == "__main__":
    main(sys.argv[1:])
