# Semi-quantum spin ice

Simulation code for forthcoming work on the disordered XXZ problem on the pyrochlore lattice.


# Building

First, ensure that the following dependencies are available and findable by `pkgconfig`:

```
HDF5 (serial or parallel version)
fftw3
Eigen 3
aranav/argparse (header only)
```


```bash
git clone git@github.com:Spuriosity1/semiquaintum_QSI.git
cd semiquaintum_QSI
meson setup -C build
```


# Executables Provided


## sq_pyrochlore


Semi-quantum classical MC for disorderd pyrochlores.

The stages of the simulation are as follows:

$$H = \sum_{\langle ij \rangle} Jzz S^zS^z + \sum_{quantum clusters} Jxx S^xS^x + Jyy S^y S^y$$

1. Removes spins from the lattice at random.
2. Identifies 'quantum clusters' in the mean-field sense of spins (see earlier).
3. Simulates thermal statistics of this Hamiltonian, under simulated annealing.


## Arguments
```
Positional arguments:
  L                           Linear dimension of the system 
  p                           Dilution probability 

Optional arguments:
  -h, --help                  shows help message and exits 
  -v, --version               prints version information and exits 
  --Jzz                       ZZ (classical) coupling strength [nargs=0..1] [default: 1]
  --Jxx                       XX coupling strength on quantum clusters [nargs=0..1] [default: 0.1]
  --Jyy                       YY coupling strength on quantum clusters [nargs=0..1] [default: 0.1]
  -a, --include_second_order  Includes also second order processes 
  -s, --dseed                 Disorder seed (controls which spins are deleted) [nargs=0..1] [default: 0]
  -m, --mseed                 MC RNG seed (controls thermal Monte Carlo trajectory) [nargs=0..1] [default: 0]
  --prefix                    [nargs=0..1] [default: "run"]
  -o, --output_dir            Output directory (filenames automatically generated) [required]
  --nburn                     Burn-In Iterations in the RNG sweep [nargs=0..1] [default: 128]
  -w, --nsweep                Iterations in the sweep [nargs=0..1] [default: 16]
  --nsamp                     Sampling steps in the sweep [nargs=0..1] [default: 16]
  --nstep                     Iterations in the RNG sweep [nargs=0..1] [default: 100]
  --Thot                      Beginning Temperature [nargs=0..1] [default: 1]
  --Tcold                     Final Temperature [nargs=0..1] [default: 0.001]
  --mf_boundary               Use approximate eigenvalue shift for boundary spin flips (faster but less accurate than default) 
  --classical                 Skip quantum cluster identification — treat all spins as classical (benchmarking) 
  --moves                     MC moves to attempt, any combination of R(ing) C(lassical) W(orm) M(onopole) B(oundary) Q(uantum). [nargs=0..1] [default: "CRWBQ"]
  --n_replicas                Number of parallel-tempering replicas (1 = standard annealing) [nargs=0..1] [default: 1]
  --n_replica_swaps           Number of attempted replica exchanges per temperature. Ignored if n_replicas=1. [nargs=0..1] [default: 10]
  --ignore_ssf                Do not compute SSF correlators.
  --ignore_tcm                Do not compute transverse (S+S-) correlators.
  -v, --verbosity             Output verbosity: 0=silent, 1=normal, 2=+Q² spinon texture, 5=+cluster spectra [nargs=0..1] [default: 1]
```

## Output format:
```
/energy                  Group
/energy/E                Dataset {nstep}
/energy/E2               Dataset {nstep}
/energy/T_list           Dataset {nstep}
/energy/n_samples        Dataset {nstep}
```
Stores a raw sum over MC sweeps of `E`, `E^2` binned by temperature, stored as T_list.

```
/geometry                Group
/geometry/n_spins        Dataset {SCALAR}
```
The number of non-deleted spins (<= 16 L^3 in general) 

```
/ssf                     Group
/ssf/T_list              Dataset {1}
/ssf/corr                Dataset {1, 16, 16, L^3, 2}
/ssf/n_samples           Dataset {1}
/ssf/sl_positions        Dataset {16, 3}
```
The <S^z S^z> (local coordinates) correlators. Indexing:
t = temperature index
mu,nu = cubic SL index (order: 0123 0123 0123 0123, by FCC unit )
k = flat k-index (C-style row-major)
z = real/imag

`corr[t, mu, nu, k, :]  = <S^z(k)_mu S^z(-k)_nu>`



```
/transverse_corr         Group
/transverse_corr/T_list  Dataset {1}
/transverse_corr/corr    Dataset {1, n_pairs}
/transverse_corr/disp_vectors Dataset {n_pairs, 3}
/transverse_corr/n_pairs_per_sample Dataset {n_pairs}
/transverse_corr/n_samples Dataset {1}
```
n_pairs is a stand-in for the number of unique intra-cluster displacement vectors.
Stores real-space information. disp_vectors stores these diplacement vectors in a
canonical (but physically meaningless) order; with respect to which the 'corr' variables <S+_r S-_{r+\Delta}> are written. 
