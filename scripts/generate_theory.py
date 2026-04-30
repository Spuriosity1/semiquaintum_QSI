#!/usr/bin/env python3
"""Generate theory heat capacity curve for a classical (Jxx=Jyy=0) sq_pyrochlore output file.

Usage:
    generate_theory <input.h5> [options]

The script:
  1. Parses L, p, Jzz, Jxx, Jyy from the input filename (errors if Jxx or Jyy != 0).
  2. Runs sq_pyrochlore_dump to obtain the defect-cluster size distribution.
  3. Reads the classical Cv from the input HDF5.
  4. Adds a theory contribution (stub — fill in cv_theory() per model).
  5. Writes output to <input_stem>.davg.theory_<model>.h5 alongside the input file.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"_L(\d+)_p([0-9.]+)_Jzz([0-9.]+)_Jxx([0-9.]+)_Jyy([0-9.]+)"
)


def parse_filename(path: str):
    m = _FNAME_RE.search(Path(path).name)
    if not m:
        sys.exit(
            f"Error: cannot parse L/p/Jzz/Jxx/Jyy from filename: {Path(path).name}\n"
            f"Expected pattern: ..._L<int>_p<float>_Jzz<float>_Jxx<float>_Jyy<float>..."
        )
    L, p, Jzz, Jxx, Jyy = m.groups()
    return int(L), float(p), float(Jzz), float(Jxx), float(Jyy)


# ---------------------------------------------------------------------------
# Cluster distribution from sq_pyrochlore_dump
# ---------------------------------------------------------------------------

def run_dump(L, p, ds, output_dir, nsweep, cluster_def):
    """Run sq_pyrochlore_dump and return its stdout."""
    cmd = [
        "build/percol/sq_pyrochlore_dump",
        str(L), str(p),
        "-o", output_dir,
        "-s", str(ds),
        "--nsweep", str(nsweep),
        "--cluster_def", cluster_def,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        sys.exit(
            f"Error: binary not found: {cmd[0]}\n"
            "Run this script from the repository root."
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error: sq_pyrochlore_dump failed:\n{e.stderr}")
    return result.stdout


def parse_cluster_dist(stdout: str) -> dict:
    """Parse cluster size → number-per-site from sq_pyrochlore_dump stdout."""
    dist = {}
    in_table = False
    for line in stdout.splitlines():
        if "Clust. Size" in line:
            in_table = True
            continue
        if not in_table:
            continue
        parts = line.split()
        if len(parts) < 2:
            break
        try:
            size = int(parts[0])
            count = float(parts[1])
        except ValueError:
            break  # hit the "X% of spins..." line or similar
        dist[size] = count
    return dist


# ---------------------------------------------------------------------------
# Theory models  — fill in cv_theory() for each model
# ---------------------------------------------------------------------------

def cv_theory(T_list: np.ndarray, cluster_dist: dict, p: float, Jpm: float,
              model: str) -> np.ndarray:
    """Return theory Cv contribution as a function of temperature.

    Parameters
    ----------
    T_list       : 1-D array of temperatures (ascending)
    cluster_dist : {cluster_size: number_per_site}
    p            : dilution probability
    Jpm          : +- coupling constant we are monkeypatching in
    model        : name of the theory model

    Returns
    -------
    cv : 1-D array, same shape as T_list
    """
    if model == "isolated_dimers":
        # adds tro-level systems witgh gap J_pm.
        # This is the situation of first order clusters 
        # subject to ice constraints
        n_dimers = cluster_dist.get(2, 0.0)
        beta = 1.0 / T_list
        x = Jpm * beta
        cv_dimer = x**2 * np.exp(-x) / (1.0 + np.exp(-x))**2
        return n_dimers * cv_dimer

    raise ValueError(f"Unknown model: '{model}'. Add it to cv_theory().")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate theory + classical Cv for a sq_pyrochlore classical output file."
    )
    parser.add_argument("input", help="Path to .h5 file produced by sq_pyrochlore (Jxx=Jyy=0)")
    parser.add_argument("--model", default="isolated_dimers",
                        help="Theory model name (used in output filename, default: isolated_dimers)")
    parser.add_argument("Jpm", help="Jpm to use for theory fit", type=float)
    parser.add_argument("--ds", type=int, default=0,
                        help="Disorder seed for sq_pyrochlore_dump (default: 0)")
    parser.add_argument("--dump_output_dir", default="../output/percol_CMC/",
                        help="Output directory passed to sq_pyrochlore_dump (default: ../output/percol_CMC/)")
    parser.add_argument("--nsweep", type=int, default=1000,
                        help="--nsweep for sq_pyrochlore_dump (default: 1000)")
    parser.add_argument("--cluster_def", default="nn2_mf",
                        choices=["nn2", "nn24", "nn2_mf"],
                        help="Cluster definition for sq_pyrochlore_dump (default: nn2)")
    args = parser.parse_args()

    # 1. Parse filename
    L, p, Jzz, Jxx, Jyy = parse_filename(args.input)

    if Jxx**2 + Jyy**2 > 1e-10:
        raise RuntimeError("Incorrect usage: this file adds a small dimer contribution to a classical (i.e. no off diagonal) curve")

    Jpm = args.Jpm

    print(f"Input:  L={L}, p={p}, Jzz={Jzz}, Jxx={Jxx}, Jyy={Jyy}")
    print(f"Theory: Jpm={Jpm} (model={args.model})")

    # 2. Run sq_pyrochlore_dump and parse cluster distribution
    print(f"Running sq_pyrochlore_dump (L={L}, p={p}, ds={args.ds}, nsweep={args.nsweep})...")
    Path(args.dump_output_dir).mkdir(parents=True, exist_ok=True)
    stdout = run_dump(L, p, args.ds, args.dump_output_dir, args.nsweep, args.cluster_def)
    cluster_dist = parse_cluster_dist(stdout)

    if not cluster_dist:
        sys.exit("Error: failed to parse any cluster distribution data from sq_pyrochlore_dump output.")

    print("  Cluster distribution (size: per-site count):")
    for size, count in sorted(cluster_dist.items()):
        print(f"    {size:4d}    {count:.6g}")

    # 3. Read input HDF5
    print(f"Reading {args.input}...")
    with h5py.File(args.input, "r") as f:
        g      = f["energy"]
        T_list = g["T_list"][:]
        N      = float(np.mean(np.array(f["/geometry/n_spins"])))

        if "var" in g:
            # disorder-averaged format: var stores sum of per-seed variances
            K        = np.array(g["n_disorder"]).astype(float)
            var_mean = np.array(g["var"]) / K
        else:
            # single-seed / raw format
            E2       = g["E2"][:]
            n        = g["n_samples"][:]
            E_mean   = g["E"][:] / n
            var_mean = E2 / n - E_mean**2

    # 4. Compute classical Cv per spin
    cv_classical = var_mean / (T_list**2 * N)   # β² Var(E) / N

    # 5. Theory contribution
    cv_th    = cv_theory(T_list, cluster_dist, p, Jpm, args.model)
    cv_total = cv_classical + cv_th

    # 6. Output path
    stem     = Path(args.input).stem   # strip .h5
    stem = re.sub(r'_Jxx0.000_Jyy0.000', f'_Jpm{Jpm:.3f}', stem) # remove old tags
    out_path = Path(args.input).parent / f"{stem}.davg.theory_{args.model}.h5"

    # 7. Write output HDF5
    print(f"Writing {out_path}...")
    with h5py.File(out_path, "w") as f:
        grp = f.create_group("heat_capacity")
        grp.attrs["L"]     = L
        grp.attrs["p"]     = p
        grp.attrs["Jzz"]   = Jzz
        grp.attrs["N"]     = N
        grp.attrs["model"] = args.model
        grp.attrs["ds"]    = args.ds

        grp.create_dataset("T_list",       data=T_list,       compression="gzip")
        grp.create_dataset("Cv_classical", data=cv_classical, compression="gzip")
        grp.create_dataset("Cv_theory",    data=cv_th,        compression="gzip")
        grp.create_dataset("Cv_total",     data=cv_total,     compression="gzip")

        # Store the cluster distribution for reference
        cgrp = grp.create_group("cluster_dist")
        sizes  = np.array(sorted(cluster_dist.keys()), dtype=np.int64)
        counts = np.array([cluster_dist[s] for s in sizes], dtype=np.float64)
        cgrp.create_dataset("sizes",  data=sizes)
        cgrp.create_dataset("counts", data=counts)

    print("Done.")


if __name__ == "__main__":
    main()
