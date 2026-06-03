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

    ap.add_argument("--jpm")
        .help("Ratio J_\\pm / J_zz")
        .default_value(0.3f)
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
        .default_value(static_cast<size_t>(100))
        .scan<'i', size_t>();

    ap.parse_args(argc, argv);



    int L = ap.get<int>("L");
    double p = ap.get<float>("p"); // site deletion probability
    double jpm = ap.get<float>("jpm"); 
    size_t seed = ap.get<size_t>("--seed");

    QClattice sc = initialise_lattice(L);

    size_t nsweep = ap.get<size_t>("--nsweep");

    std::mt19937 rng(seed);
    std::unordered_set<Tetra*> defect_tetras;

    // stats: proportion of 3- and 4-cycles on the interaction graph which
    // are frustrated
    double frust_fraction_3    = 0, frust_fraction_4    = 0;
    double frust_fraction_3_sq = 0, frust_fraction_4_sq = 0;

    JDistStats jdist;

    for (size_t n=0; n<nsweep; n++){
        // Set about p*100% of the spins to "deleted" state
        // (Bernoulli sample)
        delete_spins(rng, sc, p, defect_tetras);
        // defect_tetras is the set of 'quantum' psuedospins
        // (half-integer charges in CGSM)
        TetraBondDFS dfs_data(defect_tetras, 6);

        CycleFrustration cycle_calc(dfs_data.data(), jpm);

        jdist.accumulate(cycle_calc.bonds(), defect_tetras.size(), jpm);

        auto cycle_stats = cycle_calc.compute(3, 4);

        assert(cycle_stats[0].cycle_length == 3);
        assert(cycle_stats[1].cycle_length == 4);
        double f3 = cycle_stats[0].frust_fraction();
        double f4 = cycle_stats[1].frust_fraction();
        frust_fraction_3 += f3;
        frust_fraction_4 += f4;

        frust_fraction_3_sq += f3 * f3;
        frust_fraction_4_sq += f4 * f4;
    }

    frust_fraction_3 /= nsweep;
    frust_fraction_4 /= nsweep;

    frust_fraction_3_sq /= nsweep;
    frust_fraction_4_sq /= nsweep;

    double b = 1.0*nsweep / (nsweep - 1); // bessel correction factor

    double var_frust_fraction_3 = b * (frust_fraction_3_sq - frust_fraction_3*frust_fraction_3);
    double var_frust_fraction_4 = b * (frust_fraction_4_sq - frust_fraction_4*frust_fraction_4);


    std::filesystem::path out_dir = ap.get<std::string>("--output_dir");

    const int64_t nsweep_i = static_cast<int64_t>(nsweep);

    std::string file_stem = make_filestem(L, p, seed, nsweep, "bonds");
    auto hist_fname = out_dir / (file_stem + ".h5");

    hid_t file_id = H5Fcreate(hist_fname.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
        throw std::runtime_error("Failed to create HDF5 file: " + hist_fname.string());

    // helpers
    auto write_1d_i64 = [&](const char* name, const int64_t* data, hsize_t len) {
        hid_t space = H5Screate_simple(1, &len, nullptr);
        hid_t ds    = H5Dcreate2(file_id, name, H5T_NATIVE_INT64, space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds); H5Sclose(space);
    };
    auto write_1d_f64 = [&](const char* name, const double* data, hsize_t len) {
        hid_t space = H5Screate_simple(1, &len, nullptr);
        hid_t ds    = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE, space,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds); H5Sclose(space);
    };

    // J distribution arrays
    write_1d_i64("J_hist",        jdist.hist.data(),
                 static_cast<hsize_t>(JDistStats::NBINS));
    {
        auto edges = JDistStats::bin_edges();
        write_1d_f64("J_bin_edges", edges.data(),
                     static_cast<hsize_t>(edges.size()));
    }
    if (!jdist.degree_hist.empty())
        write_1d_i64("degree_hist", jdist.degree_hist.data(),
                     static_cast<hsize_t>(jdist.degree_hist.size()));

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
        auto write_f64 = [&](const char* name, double val) {
            hid_t ds = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);
            printf("%s\t%lf\n", name, val);
            H5Dclose(ds);
        };
        write_i64("nsweep",              nsweep_i);
        write_i64("L",                   static_cast<int64_t>(L));
        write_f64("frust_3_cycle",       frust_fraction_3);
        write_f64("frust_4_cycle",       frust_fraction_4);
        write_f64("frust_3_cycle_stdev", sqrt(var_frust_fraction_3));
        write_f64("frust_4_cycle_stdev", sqrt(var_frust_fraction_4));
        write_f64("J_mean",              jdist.mean());
        write_f64("J_variance",          jdist.variance());
        write_f64("J_neg_frac",          jdist.neg_frac());
        write_f64("jpm",                 jpm);
        H5Sclose(space);
    }

    H5Fclose(file_id);


    return 0;
}


