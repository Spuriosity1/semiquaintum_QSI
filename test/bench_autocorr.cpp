#include "lattice_lib/supercell.hpp"
#include "lattice_lib/unitcell_types.hpp"
#include <argparse/argparse.hpp>
#include <unordered_set>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "quantum_cluster.hpp"
#include "monte_carlo.hpp"
#include "sim_bits.hpp"

using namespace std;

// Counts intact tetrahedra (all 4 members classical and present) with Q != 0.
static int count_monopoles(const std::vector<Tetra*>& intact_tetras) {
    int m = 0;
    for (Tetra* t : intact_tetras) {
        int q = 0;
        for (Spin* s : t->member_spins) q += s->ising_val;
        if (q != 0) ++m;
    }
    return m;
}

// Write a 1-D vector<double> as an HDF5 dataset inside an open group.
static void write_double_vec(hid_t group, const char* name,
                             const std::vector<double>& v) {
    hsize_t dims[1] = { v.size() };
    hid_t sp = H5Screate_simple(1, dims, nullptr);
    hid_t ds = H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, sp,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data());
    H5Dclose(ds);
    H5Sclose(sp);
}

static std::string make_filename(int L, float p,
                                 size_t dseed, size_t mseed,
                                 double T,
                                 const std::string& prefix = "autocorr",
                                 const std::string& ext    = "h5") {
    const auto& param = ModelParams::get();
    std::ostringstream oss;
    oss << prefix
        << "_L"   << L
        << "_p"   << std::fixed << std::setprecision(3) << p
        << "_Jzz" << param.Jzz
        << "_Jxx" << param.Jxx
        << "_Jyy" << param.Jyy
        << "_T"   << std::setprecision(4) << T
        << "_ds"  << dseed
        << "_ms"  << mseed
        << "."    << ext;
    return oss.str();
}


int main(int argc, char* argv[]) {

    auto ap = argparse::ArgumentParser("bench_autocorr");

    // Physical
    ap.add_argument("L")
        .help("Linear dimension of the system")
        .scan<'i', int>();
    ap.add_argument("p")
        .help("Dilution probability")
        .scan<'g', float>();

    ap.add_argument("--Jzz").default_value(1.0).scan<'g', double>();
    ap.add_argument("--Jxx").default_value(0.1).scan<'g', double>();
    ap.add_argument("--Jyy").default_value(0.1).scan<'g', double>();

    ap.add_argument("--T")
        .help("Temperature at which to measure autocorrelation")
        .required()
        .scan<'g', double>();

    // Seeds
    ap.add_argument("--dseed", "-s")
        .help("Disorder seed")
        .default_value(static_cast<size_t>(0))
        .scan<'i', size_t>();
    ap.add_argument("--mseed", "-m")
        .help("MC RNG seed")
        .default_value(static_cast<size_t>(0))
        .scan<'i', size_t>();

    // MC protocol
    ap.add_argument("--nburn")
        .help("Burn-in sweeps before recording")
        .default_value(static_cast<size_t>(1024))
        .scan<'i', size_t>();
    ap.add_argument("--nsweep")
        .help("Number of sweeps to record")
        .default_value(static_cast<size_t>(65536))
        .scan<'i', size_t>();

    ap.add_argument("--mf_boundary")
        .help("Use MF boundary-spin approximation (faster)")
        .default_value(false)
        .implicit_value(true);

    ap.add_argument("--moves")
        .help("MC moves to attempt (R/C/W/M/B/Q). Default: all.")
        .default_value(std::string("RCWMBQ"));

    ap.add_argument("--output_dir", "-o")
        .help("Output directory")
        .required();

    ap.add_argument("--prefix")
        .default_value(std::string("autocorr"));

    ap.add_argument("--verbosity", "-v")
        .default_value(1)
        .scan<'i', int>();

    ap.parse_args(argc, argv);

    // Parameters
    ModelParams::get().Jzz = ap.get<double>("--Jzz");
    ModelParams::get().Jxx = ap.get<double>("--Jxx");
    ModelParams::get().Jyy = ap.get<double>("--Jyy");

    int    L      = ap.get<int>("L");
    double p      = ap.get<float>("p");
    size_t dseed  = ap.get<size_t>("--dseed");
    size_t mseed  = ap.get<size_t>("--mseed");
    double T      = ap.get<double>("--T");
    size_t nburn  = ap.get<size_t>("--nburn");
    size_t nsweep = ap.get<size_t>("--nsweep");
    bool   exact_boundary = !ap.get<bool>("--mf_boundary");
    int    verbosity      = ap.get<int>("--verbosity");

    MCSettings params;
    params.beta  = 1.0 / T;
    params.moves = parse_moves(ap.get<std::string>("--moves"));

    auto hashf = std::hash<size_t>();
    std::mt19937 rng(hashf(dseed));
    params.rng.seed(hashf(mseed ^ (dseed * 105)));

    // Build lattice and disorder realization
    QClattice sc = initialise_lattice(L);
    std::unordered_set<Tetra*> seed_tetras;
    MCStateMF state;

    delete_spins(rng, sc, p, seed_tetras);
    identify_1o_clusters(seed_tetras, state.clusters);
    identify_flippable_hexas(sc, state.intact_plaqs);

    for (auto& qc : state.clusters) qc.initialise();
    state.partition_spins(sc.get_objects<Spin>());

    if (verbosity >= 1) {
        output_cluster_dist(std::cout, state.clusters, 1);
        std::cout << "\n==========================\n"
                  << state.classical_spins.size() << " Full classical spins\n"
                  << state.boundary_spins.size()  << " Boundary spins\n"
                  << state.clusters.size()         << " Quantum clusters\n"
                  << state.intact_tetras.size()    << " Intact tetrahedra\n"
                  << "T = " << T << "\n";
    }

    // Two-stage burn-in: first randomise at high T, then equilibrate at target T.
    const double T_hot = 10.0 * ModelParams::get().Jzz;
    if (verbosity >= 1)
        std::cout << "Burn-in stage 1: " << nburn << " sweeps at T=" << T_hot << "\n";
    params.beta = 1.0 / T_hot;
    for (size_t n = 0; n < nburn; ++n) {
        if (exact_boundary) state.sweep<true>(params);
        else                state.sweep<false>(params);
    }

    if (verbosity >= 1)
        std::cout << "Burn-in stage 2: " << nburn << " sweeps at T=" << T << "\n";
    params.beta = 1.0 / T;
    for (size_t n = 0; n < nburn; ++n) {
        if (exact_boundary) state.sweep<true>(params);
        else                state.sweep<false>(params);
    }
    params.reset_acceptance();
    if (verbosity >= 1) std::cout << "Done.\n";

    // Record time series
    std::vector<double> E_series(nsweep);
    std::vector<double> M_series(nsweep);

    for (size_t n = 0; n < nsweep; ++n) {
        if (exact_boundary) state.sweep<true>(params);
        else                state.sweep<false>(params);

        E_series[n] = state.energy();
        M_series[n] = static_cast<double>(count_monopoles(state.intact_tetras));
    }

    if (verbosity >= 1) {
        std::cout << "Recorded " << nsweep << " sweeps. "
                  << params.acceptance() << "\n";
    }

    // Write HDF5
    std::filesystem::path out_dir = ap.get<std::string>("--output_dir");
    const std::string prefix = ap.get<std::string>("--prefix");
    auto file_path = out_dir / make_filename(L, p, dseed, mseed, T, prefix);

    hid_t fid = H5Fcreate(file_path.string().c_str(),
                           H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0)
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());

    hid_t grp = H5Gcreate2(fid, "timeseries",
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_double_vec(grp, "E", E_series);
    write_double_vec(grp, "M", M_series);
    H5Gclose(grp);

    write_geometry_group(fid, sc);

    H5Fclose(fid);

    std::cout << file_path << "\n";
    return 0;
}
