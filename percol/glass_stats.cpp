#include <argparse/argparse.hpp>
#include <unordered_set>
#include <hdf5.h>

#include "glass_stats.hpp"
#include "sim_bits.hpp"

using namespace std;


// Semi-quantum classical MC for disorderd pyrochlores.
//
//
// 1. Removes spins from the lattice at random.
// 2. Identifies 'quantum clusters' of spins
//

// Jpm sweep: -1.0 to +1.0 inclusive in steps of 0.05 (41 values)
static constexpr double JPM_MIN  = -1.0;
static constexpr double JPM_MAX  =  1.0;
static constexpr double JPM_STEP =  0.05;
static constexpr int    N_JPM    = static_cast<int>((JPM_MAX - JPM_MIN) / JPM_STEP + 0.5) + 1;


std::string make_filestem(
    int L,
    float p,
    size_t seed,
    size_t nsweep,
    const std::string& prefix = "glass"
    )
{
    std::ostringstream oss;

    oss << prefix
        << "_L" << L
        << "_p" << std::fixed << std::setprecision(3) << p
        << "_s" << seed
        << "_w" << nsweep;

    return oss.str();
}


int main (int argc, char *argv[]) {

    auto ap = argparse::ArgumentParser("pyro_cmc");

    /// PHYSICAL
    ///
    ap.add_argument("L")
        .help("Linear dimension of the system")
        .scan<'i', int>();
    ap.add_argument("p")
        .help("Dilution probability")
        .scan<'g', float>();

    /// BOOK-KEEPING

    ap.add_argument("--seed", "-s")
        .help("RNG seed")
        .default_value(static_cast<size_t>(0))
        .scan<'i', size_t>();
    ap.add_argument("--output_dir", "-o")
        .help("Output directory (filenames automatically generated)")
        .required();
    ap.add_argument("--nsweep", "-w")
        .help("Iterations in the RNG sweep")
        .default_value(static_cast<size_t>(16))
        .scan<'i', size_t>();

    ap.parse_args(argc, argv);



    int L = ap.get<int>("L");
    double p = ap.get<float>("p"); // site deletion probability
    size_t seed = ap.get<size_t>("--seed");

    QClattice sc = initialise_lattice(L);

    size_t nsweep = ap.get<size_t>("--nsweep");

    std::mt19937 rng(seed);
    std::unordered_set<Tetra*> defect_tetras;

    // Per-Jpm running sums for frustration and J-distribution
    std::vector<double> frust3_sum(N_JPM, 0), frust3_sq(N_JPM, 0);
    std::vector<double> frust4_sum(N_JPM, 0), frust4_sq(N_JPM, 0);
    std::vector<JDistStats> jdist(N_JPM);

    for (size_t n = 0; n < nsweep; n++) {
        // Set about p*100% of the spins to "deleted" state (Bernoulli sample)
        delete_spins(rng, sc, p, defect_tetras);
        // defect_tetras is the set of 'quantum' pseudospins (half-integer charges in CGSM)
        TetraBondDFS dfs_data(defect_tetras, 6);

        // Expensive: enumerate cycles once per disorder realization
        CycleFrustration cf(dfs_data.data());
        auto cycles = cf.enumerate_cycles(3, 4);

        // Cheap: classify frustration and accumulate J-stats for each Jpm
        for (int i = 0; i < N_JPM; i++) {
            double jpm = JPM_MIN + i * JPM_STEP;

            auto cs = cf.classify(cycles, jpm, 3, 4);
            assert(cs[0].cycle_length == 3);
            assert(cs[1].cycle_length == 4);
            double f3 = cs[0].frust_fraction(), f4 = cs[1].frust_fraction();
            frust3_sum[i] += f3; frust3_sq[i] += f3*f3;
            frust4_sum[i] += f4; frust4_sq[i] += f4*f4;

            if (std::abs(jpm) > 1e-10)
                jdist[i].accumulate(cf.bonds(jpm), defect_tetras.size(), jpm);
        }
    }

    // Build output arrays
    const double bessel = 1.0 * nsweep / (nsweep - 1);
    std::vector<double> jpm_arr(N_JPM),    J_mean_arr(N_JPM),    J_var_arr(N_JPM);
    std::vector<double> frust3_arr(N_JPM), frust3_stdev(N_JPM);
    std::vector<double> frust4_arr(N_JPM), frust4_stdev(N_JPM);
    for (int i = 0; i < N_JPM; i++) {
        jpm_arr[i]    = JPM_MIN + i * JPM_STEP;
        J_mean_arr[i] = jdist[i].mean();
        J_var_arr[i]  = jdist[i].variance();

        double f3 = frust3_sum[i] / nsweep, f3sq = frust3_sq[i] / nsweep;
        double f4 = frust4_sum[i] / nsweep, f4sq = frust4_sq[i] / nsweep;
        frust3_arr[i]   = f3;
        frust4_arr[i]   = f4;
        frust3_stdev[i] = std::sqrt(bessel * (f3sq - f3*f3));
        frust4_stdev[i] = std::sqrt(bessel * (f4sq - f4*f4));
    }


    std::filesystem::path out_dir = ap.get<std::string>("--output_dir");

    const int64_t nsweep_i = static_cast<int64_t>(nsweep);

    std::string file_stem = make_filestem(L, p, seed, nsweep, "tglass");
    auto hist_fname = out_dir / (file_stem + ".h5");

    hid_t file_id = H5Fcreate(hist_fname.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
        throw std::runtime_error("Failed to create HDF5 file: " + hist_fname.string());

    // helpers
    auto write_1d_f64 = [&](const char* name, const double* data, hsize_t len) {
        hid_t space = H5Screate_simple(1, &len, nullptr);
        hid_t ds    = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE, space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds); H5Sclose(space);
    };

    // Jpm sweep arrays
    write_1d_f64("jpm",              jpm_arr.data(),    static_cast<hsize_t>(N_JPM));
    write_1d_f64("J_mean",           J_mean_arr.data(), static_cast<hsize_t>(N_JPM));
    write_1d_f64("J_variance",       J_var_arr.data(),  static_cast<hsize_t>(N_JPM));
    write_1d_f64("frust_3_cycle",    frust3_arr.data(), static_cast<hsize_t>(N_JPM));
    write_1d_f64("frust_4_cycle",    frust4_arr.data(), static_cast<hsize_t>(N_JPM));
    write_1d_f64("frust_3_cycle_stdev", frust3_stdev.data(), static_cast<hsize_t>(N_JPM));
    write_1d_f64("frust_4_cycle_stdev", frust4_stdev.data(), static_cast<hsize_t>(N_JPM));

    // scalars
    {
        hid_t space = H5Screate(H5S_SCALAR);
        auto write_i64 = [&](const char* name, int64_t val) {
            hid_t ds = H5Dcreate2(file_id, name, H5T_NATIVE_INT64, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);
            printf("%s\t%lld\n", name, val);
            H5Dclose(ds);
        };
        write_i64("nsweep", nsweep_i);
        write_i64("L",      static_cast<int64_t>(L));
        H5Sclose(space);
    }

    H5Fclose(file_id);


    return 0;
}


