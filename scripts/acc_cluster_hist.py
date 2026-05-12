#!/usr/bin/env python3
"""
Merge cluster-histogram HDF5 files produced by sq_pyrochlore_dump.

All input files must share the same L, p, cluster_def, and N.
Counts are summed bin-by-bin; nsweep values are accumulated.

Writes one merged file:
    hist_L{L}_p{p:.3f}_w{total_nsweep}_merge{K}_{clust_type}.h5

Datasets in the output:
    sizes        int64[M]   — distinct cluster sizes (union across all inputs)
    counts       int64[M]   — Σ_k counts_k[s]          (sum across files)
    counts_sq    int64[M]   — Σ_k counts_sq_k[s]       (sum of per-sweep Σx² across files)
    nsweep       int64      — total sweeps accumulated
    N            int64      — spins in the supercell (must match across inputs)
    n_files      int64      — number of input files merged (K)
    rate_sum     float64[M] — Σ_k r_k[s],   r_k = counts_k / nsweep_k
    rate_sq_sum  float64[M] — Σ_k r_k[s]²
    var          float64[M] — Bessel-corrected sample variance of r_k across files:
                              (rate_sq_sum - rate_sum²/K) / (K-1); NaN when K=1.
    var_sweep    float64[M] — Bessel-corrected pooled per-sweep variance:
                              (counts_sq - counts²/nsweep_total) / (nsweep_total-1);
                              NaN when counts_sq is absent from all inputs.

counts_sq, rate_sum, and rate_sq_sum are raw moments stored so that merged
files can themselves be re-merged by summing moments then recomputing variances.

Usage:
    acc_cluster_hist.py hist_L4_p0.050_s0_w100_nn2.h5 \\
                        hist_L4_p0.050_s1_w100_nn2.h5 ...
"""

import re
import sys
import os.path
import numpy as np
import h5py


_FILENAME_RE = re.compile(
    r"hist_L(\d+)_p([0-9.]+)_s([0-9]+)_w([0-9]+)_(\w+)\.h5$"
)


def parse_filename(path):
    m = _FILENAME_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Filename not in expected format: {os.path.basename(path)}")
    return {
        "L":          int(m.group(1)),
        "p":          float(m.group(2)),
        "seed":       int(m.group(3)),
        "nsweep":     int(m.group(4)),
        "clust_type": m.group(5),
    }


def load_hist(fname):
    """Return sizes, counts, counts_sq (or None), nsweep, N."""
    with h5py.File(fname, "r") as f:
        sizes      = np.array(f["sizes"],  dtype=np.int64)
        counts     = np.array(f["counts"], dtype=np.int64)
        counts_sq  = np.array(f["counts_sq"], dtype=np.int64) if "counts_sq" in f else None
        nsweep     = int(np.array(f["nsweep"]))
        N          = int(np.array(f["N"]))
    return sizes, counts, counts_sq, nsweep, N


def check_compatibility(metas, fnames):
    ref = metas[0]
    for meta, fname in zip(metas[1:], fnames[1:]):
        bad = [
            f"{k}: {ref[k]} vs {meta[k]}"
            for k in ("L", "clust_type")
            if ref[k] != meta[k]
        ]
        if abs(ref["p"] - meta["p"]) > 1e-6:
            bad.append(f"p: {ref['p']} vs {meta['p']}")
        if bad:
            raise ValueError(
                f"{os.path.basename(fname)} incompatible with "
                f"{os.path.basename(fnames[0])}: " + ", ".join(bad)
            )


def merged_outpath(fnames, total_nsweep):
    meta = parse_filename(fnames[0])
    name = (
        f"hist_L{meta['L']}_p{meta['p']:.3f}"
        f"_w{total_nsweep}_merge{len(fnames)}_{meta['clust_type']}.h5"
    )
    return os.path.join(os.path.dirname(os.path.abspath(fnames[0])), name)


def main(fnames):
    if not fnames:
        print(__doc__)
        sys.exit(1)

    metas = [parse_filename(f) for f in fnames]
    check_compatibility(metas, fnames)

    # Load all files; check N consistency.
    all_data: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, int]] = []
    total_nsweep = 0
    N: int | None = None

    for fname in fnames:
        sizes, counts, counts_sq, nsweep, n = load_hist(fname)
        if N is None:
            N = n
        elif N != n:
            raise ValueError(f"N mismatch in {os.path.basename(fname)}: {n} vs {N}")
        all_data.append((sizes, counts, counts_sq, nsweep))
        total_nsweep += nsweep

    assert N is not None

    # Build union of observed sizes; map each size to a dense index.
    size_set: set[int] = set()
    for sizes, _, _, _ in all_data:
        size_set.update(sizes.tolist())
    sizes_out = np.array(sorted(size_set), dtype=np.int64)
    size_idx  = {int(s): i for i, s in enumerate(sizes_out.tolist())}
    M = len(sizes_out)
    K = len(fnames)

    counts_sum    = np.zeros(M, dtype=np.int64)
    counts_sq_sum = np.zeros(M, dtype=np.int64)
    rate_sum      = np.zeros(M, dtype=np.float64)
    rate_sq_sum   = np.zeros(M, dtype=np.float64)
    have_counts_sq = False

    for sizes, counts, counts_sq, nsweep in all_data:
        # scatter sparse file arrays into the dense union grid
        counts_dense    = np.zeros(M, dtype=np.int64)
        counts_sq_dense = np.zeros(M, dtype=np.int64)
        for i, s in enumerate(sizes.tolist()):
            idx = size_idx[int(s)]
            counts_dense[idx] = counts[i]
            if counts_sq is not None:
                counts_sq_dense[idx] = counts_sq[i]

        counts_sum    += counts_dense
        if counts_sq is not None:
            counts_sq_sum += counts_sq_dense
            have_counts_sq = True

        r_k = counts_dense.astype(np.float64) / nsweep
        rate_sum    += r_k
        rate_sq_sum += r_k ** 2

    # Between-file Bessel-corrected variance of the per-file mean rate r_k.
    if K > 1:
        var_out = (rate_sq_sum - rate_sum ** 2 / K) / (K - 1)
    else:
        var_out = np.full(M, np.nan)

    # Pooled within-file (per-sweep) Bessel-corrected variance.
    # var_sweep = (Σx² - (Σx)²/N) / (N-1),  N = total_nsweep.
    if have_counts_sq and total_nsweep > 1:
        var_sweep = (
            counts_sq_sum.astype(np.float64)
            - counts_sum.astype(np.float64) ** 2 / total_nsweep
        ) / (total_nsweep - 1)
    else:
        var_sweep = np.full(M, np.nan)

    out = merged_outpath(fnames, total_nsweep)
    with h5py.File(out, "w") as f:
        f.create_dataset("sizes",        data=sizes_out)
        f.create_dataset("counts",       data=counts_sum)
        f.create_dataset("counts_sq",    data=counts_sq_sum)
        f.create_dataset("nsweep",       data=np.int64(total_nsweep))
        f.create_dataset("N",            data=np.int64(N))
        f.create_dataset("n_files",      data=np.int64(K))
        f.create_dataset("rate_sum",     data=rate_sum)
        f.create_dataset("rate_sq_sum",  data=rate_sq_sum)
        f.create_dataset("var",          data=var_out)
        f.create_dataset("var_sweep",    data=var_sweep)

    n_quantum = float(np.dot(sizes_out.astype(float), counts_sum.astype(float)))
    pct = 100.0 * n_quantum / (N * total_nsweep)
    print(
        f"Merged {K} file(s): total nsweep={total_nsweep}, "
        f"{pct:.2f}% of spins in clusters\n→ {out}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
