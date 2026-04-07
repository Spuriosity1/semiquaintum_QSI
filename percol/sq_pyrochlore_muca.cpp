#include "lattice_lib/supercell.hpp"
#include "lattice_lib/unitcell_types.hpp"
#include "energy_manager.hpp"
#include <argparse/argparse.hpp>
#include <chrono>
#include <numeric>
#include <unordered_set>

#include "quantum_cluster.hpp"
#include "monte_carlo.hpp"
#include "sim_bits.hpp"

using namespace std;

// ---- filename helpers ----

static std::string make_filename_muca(
    int L, float p, size_t dseed, size_t mseed, int n_bins,
    const std::string& prefix = "muca")
{
    std::ostringstream oss;
    const auto& param = ModelParams::get();
    oss << prefix
        << "_L" << L
        << "_p" << std::fixed << std::setprecision(3) << p
        << "_Jzz" << param.Jzz
        << "_Jxx" << param.Jxx
        << "_Jyy" << param.Jyy
        << "_bins" << n_bins
        << "_ds" << dseed
        << "_ms" << mseed
        << ".h5";
    return oss.str();
}

// ---- HDF5 write helpers ----

static hid_t make_group(hid_t parent, const char* name) {
    hid_t g = H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (g < 0) throw std::runtime_error(std::string("Failed to create HDF5 group: ") + name);
    return g;
}

static void write_1d(hid_t group, const char* name, hid_t dtype,
                     const void* data, hsize_t n)
{
    hid_t sp  = H5Screate_simple(1, &n, nullptr);
    hid_t ds  = H5Dcreate2(group, name, dtype, sp,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (ds < 0) throw std::runtime_error(std::string("Failed to create dataset: ") + name);
    H5Dwrite(ds, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds);
    H5Sclose(sp);
}

static void write_scalar(hid_t group, const char* name, hid_t dtype, const void* data) {
    hsize_t one = 1;
    write_1d(group, name, dtype, data, one);
}

// ---- Wang-Landau helpers ----

static bool histogram_is_flat(const MUCAContext& ctx, double flatness_threshold) {
    int64_t H_min  = std::numeric_limits<int64_t>::max();
    int64_t H_tot  = 0;
    int     n_vis  = 0;
    for (auto h : ctx.wl_H) {
        if (h == 0) continue;          // skip inaccessible bins
        H_min = std::min(H_min, h);
        H_tot += h;
        ++n_vis;
    }
    if (n_vis == 0) return false;
    double H_mean = static_cast<double>(H_tot) / n_vis;
    return static_cast<double>(H_min) / H_mean >= flatness_threshold;
}

// ---- main ----

int main(int argc, char* argv[]) {

    auto ap = argparse::ArgumentParser("pyro_muca");

    // Physical parameters (identical to sq_pyrochlore)
    ap.add_argument("L").help("Linear dimension").scan<'i', int>();
    ap.add_argument("p").help("Dilution probability").scan<'g', float>();
    ap.add_argument("--Jzz").default_value(1.0).scan<'g', double>();
    ap.add_argument("--Jxx").default_value(0.1).scan<'g', double>();
    ap.add_argument("--Jyy").default_value(0.1).scan<'g', double>();
    ap.add_argument("--include_second_order", "-a").default_value(false).implicit_value(true);

    // Bookkeeping
    ap.add_argument("--dseed", "-s").default_value(static_cast<size_t>(0)).scan<'i', size_t>();
    ap.add_argument("--mseed", "-m").default_value(static_cast<size_t>(0)).scan<'i', size_t>();
    ap.add_argument("--prefix").default_value("muca");
    ap.add_argument("--output_dir", "-o").required();
    ap.add_argument("--moves").default_value(std::string("CRWBQ"));
    ap.add_argument("--mf_boundary").default_value(false).implicit_value(true);
    ap.add_argument("--verbosity", "-v").default_value(1).scan<'i', int>();

    // Initial equilibration
    ap.add_argument("--nburn").default_value(static_cast<size_t>(128)).scan<'i', size_t>();

    // Wang-Landau parameters
    ap.add_argument("--E_min").help("Lower energy bound for binning").scan<'g', double>();
    ap.add_argument("--E_max").help("Upper energy bound for binning").scan<'g', double>();
    ap.add_argument("--n_bins").default_value(200).scan<'i', int>();
    ap.add_argument("--f_final")
        .help("WL convergence: stop when ln(f) < this value")
        .default_value(1e-8).scan<'g', double>();
    ap.add_argument("--flatness").default_value(0.8).scan<'g', double>();
    ap.add_argument("--wl_check").help("Sweeps between flatness checks")
        .default_value(static_cast<size_t>(1000)).scan<'i', size_t>();
//    ap.add_argument("--explore_steps")
//        .help("Initial flat-histogram sweeps to estimate E range (0 = use E_min/E_max)")
//        .default_value(static_cast<size_t>(0)).scan<'i', size_t>();

    // Production parameters
    ap.add_argument("--n_prod").default_value(static_cast<size_t>(100000)).scan<'i', size_t>();
    ap.add_argument("--resync").help("Sweeps between full E_current recalculations")
        .default_value(static_cast<size_t>(1000)).scan<'i', size_t>();
    ap.add_argument("--T_reweight")
        .help("Comma-separated temperatures for canonical reweighting, e.g. 0.1,0.5,2.0")
        .default_value(std::string("0.05,0.1,0.2,0.5,1.0,2.0,5.0"));

    ap.parse_args(argc, argv);

    // Load parameters
    ModelParams::get().Jzz = ap.get<double>("--Jzz");
    ModelParams::get().Jxx = ap.get<double>("--Jxx");
    ModelParams::get().Jyy = ap.get<double>("--Jyy");

    int    L      = ap.get<int>("L");
    double p      = ap.get<float>("p");
    size_t dseed  = ap.get<size_t>("--dseed");
    size_t mseed  = ap.get<size_t>("--mseed");
    size_t nburn  = ap.get<size_t>("--nburn");
    int    n_bins = ap.get<int>("--n_bins");
    double f_final_threshold = ap.get<double>("--f_final");
    double flatness = ap.get<double>("--flatness");
    size_t wl_check = ap.get<size_t>("--wl_check");
    size_t n_prod   = ap.get<size_t>("--n_prod");
    size_t resync   = ap.get<size_t>("--resync");
//    size_t explore_steps = ap.get<size_t>("--explore_steps");
    bool exact_boundary = !ap.get<bool>("--mf_boundary");
    int  verbosity = ap.get<int>("--verbosity");

    // Parse T_reweight list
    std::vector<double> T_list_rw;
    {
        std::string trw = ap.get<std::string>("--T_reweight");
        std::istringstream ss(trw);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty()) T_list_rw.push_back(std::stod(tok));
    }

    MCSettings params;
    params.moves = parse_moves(ap.get<std::string>("--moves"));

    // Build lattice
    QClattice sc = initialise_lattice(L);

    auto hashf = std::hash<size_t>();
    std::mt19937 rng_disorder(hashf(dseed));
    params.rng.seed(hashf(mseed ^ (dseed * 105)));

    std::unordered_set<Tetra*> seed_tetras;
    MCStateMF state;

    delete_spins(rng_disorder, sc, p, seed_tetras);
    identify_1o_clusters(seed_tetras, state.clusters);
    identify_flippable_hexas(sc, state.intact_plaqs);

    for (auto& qc : state.clusters) qc.initialise();
    state.partition_spins(sc.get_objects<Spin>());

    if (verbosity >= 1) {
        output_cluster_dist(std::cout, state.clusters, 1);
        int exp_Jzz_bonds = calc_GS_energy(sc.get_objects<Tetra>());
        std::cout << "Expected ground state energy: " << exp_Jzz_bonds << " Jzz\n";
        std::cout << state.classical_spins.size() << " classical  "
                  << state.boundary_spins.size()  << " boundary  "
                  << state.clusters.size()         << " clusters\n";
    }

    // ---- Initial burn-in at infinite T ----
    params.beta = 0.0;
    if (verbosity >= 1) std::cout << "Burning in (" << nburn << " sweeps)...\n";
    for (size_t n = 0; n < nburn; n++) {
        if (exact_boundary) state.sweep<true>(params);
        else                state.sweep<false>(params);
    }
    double E_now = state.energy();
    if (verbosity >= 1) std::cout << "E after burn-in = " << E_now << "\n";

    // ---- Energy range determination ----
    double E_min, E_max;
    {
        double Jxx = ModelParams::get().Jxx;
        double Jyy = ModelParams::get().Jyy;
        E_min = calc_GS_energy(sc.get_objects<Tetra>());

        for (auto q : state.clusters){
            auto n = q.n_spins();
            E_min -= n*(n-1) * sqrt(Jxx*Jxx + Jyy*Jyy)/2;
        }


        E_max = 0;
    }

//    if (explore_steps > 0) {
//        // Short flat random walk (beta=0, no MUCA weights) to find the accessible range.
//        double E_lo = E_now, E_hi = E_now;
//        for (size_t sw = 0; sw < explore_steps; sw++) {
//            if (exact_boundary) state.sweep<true>(params);
//            else                state.sweep<false>(params);
//            double e = state.energy();
//            E_lo = std::min(E_lo, e);
//            E_hi = std::max(E_hi, e);
//        }
//        // Add 10% margin on each side
//        double margin = 0.1 * (E_hi - E_lo);
//        E_min = E_lo - margin;
//        E_max = E_hi + margin;
//        if (verbosity >= 1)
//            std::cout << "Exploration: E in [" << E_lo << ", " << E_hi << "]  "
//                      << "→ bins [" << E_min << ", " << E_max << "]\n";
//    } else {
    if (ap.is_used("--E_min")) 
        E_min = ap.get<double>("--E_min");
    else 
        std::cout << "Using estimated E_min "<<E_min<<std::endl;
    
    if (ap.is_used("--E_max"))
        E_max = ap.get<double>("--E_max");
    else
        std::cout << "Using estimated E_max "<<E_max<<std::endl;

    // ---- Initialise MUCAContext ----
    MUCAContext muca;
    muca.n_bins    = n_bins;
    muca.E_min     = E_min;
    muca.bin_width = (E_max - E_min) / n_bins;
    muca.lnG.assign(n_bins, 0.0);
    muca.wl_H.assign(n_bins, 0);
    muca.E_current = state.energy();

    params.muca = &muca;

    // ---- Wang-Landau phase ----
    if (verbosity >= 1) std::cout << "\n=== Wang-Landau phase ===\n";

    double ln_f = 1.0;   // modification factor in log space
    int wl_iter = 0;

    while (ln_f > f_final_threshold) {
        size_t inner = 0;

        while (true) {
            // Run wl_check sweeps
            for (size_t sw = 0; sw < wl_check; sw++) {
                if (exact_boundary) state.sweep<true>(params);
                else                state.sweep<false>(params);

                if (sw % resync == 0) {
                    muca.E_current = state.energy();
                }
            }
            inner++;

            // Update lnG and wl_H for current bin
            int k = muca.energy_bin(muca.E_current);
            if (k >= 0) {
                muca.lnG[k]  += ln_f;
                muca.wl_H[k] += 1;
            }

            if (histogram_is_flat(muca, flatness)) {
                ln_f /= 2.0;
                muca.wl_H.assign(n_bins, 0);
                wl_iter++;
                if (verbosity >= 1)
                    std::cout << "WL iter " << wl_iter
                              << "  new ln_f = " << ln_f << "\n" << std::flush;
                break;
            }

            if (verbosity >= 2 && inner % 100 == 0) {
                int64_t H_min  = *std::min_element(muca.wl_H.begin(), muca.wl_H.end());
                double  H_mean = static_cast<double>(
                    std::accumulate(muca.wl_H.begin(), muca.wl_H.end(), int64_t(0)))
                    / n_bins;
                std::cout << "  ln_f=" << ln_f << "  H_min=" << H_min
                          << "  H_mean=" << H_mean << "\n";
            }
        }
    }

    if (verbosity >= 1)
        std::cout << "WL converged after " << wl_iter << " iterations.\n";

    // ---- Production phase ----
    if (verbosity >= 1)
        std::cout << "\n=== Production phase (" << n_prod << " sweeps) ===\n";

    std::vector<int64_t> H_prod(n_bins, 0);
    std::vector<double>  E_sum(n_bins, 0.0);
    std::vector<double>  E2_sum(n_bins, 0.0);

    for (size_t sw = 0; sw < n_prod; sw++) {
        if (exact_boundary) state.sweep<true>(params);
        else                state.sweep<false>(params);

        if (sw % resync == 0) {
            muca.E_current = state.energy();
        }

        int k = muca.energy_bin(muca.E_current);
        if (k >= 0) {
            H_prod[k]++;
            E_sum[k]  += muca.E_current;
            E2_sum[k] += muca.E_current * muca.E_current;
        }
    }

    if (verbosity >= 1) {
        int64_t H_min  = *std::min_element(H_prod.begin(), H_prod.end());
        int64_t H_max  = *std::max_element(H_prod.begin(), H_prod.end());
        std::cout << "Production histogram: H_min=" << H_min << "  H_max=" << H_max << "\n";
    }

    // ---- Canonical reweighting ----
    // Bin centres
    std::vector<double> E_bins_centre(n_bins);
    for (int k = 0; k < n_bins; k++)
        E_bins_centre[k] = E_min + (k + 0.5) * muca.bin_width;

    std::vector<double> rw_mean_E(T_list_rw.size());
    std::vector<double> rw_C(T_list_rw.size());

    for (size_t ti = 0; ti < T_list_rw.size(); ti++) {
        double beta_T = 1.0 / T_list_rw[ti];

        // log-sum-exp for Z
        double log_Z = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < n_bins; k++) {
            if (H_prod[k] == 0) continue;
            double log_w = std::log(static_cast<double>(H_prod[k]))
                         - muca.lnG[k]
                         - beta_T * E_bins_centre[k];
            if (log_w > log_Z) {
                log_Z = log_w + std::log1p(std::exp(log_Z - log_w));
            } else {
                log_Z += std::log1p(std::exp(log_w - log_Z));
            }
        }

        double mean_E = 0.0, mean_E2 = 0.0;
        for (int k = 0; k < n_bins; k++) {
            if (H_prod[k] == 0) continue;
            double log_w = std::log(static_cast<double>(H_prod[k]))
                         - muca.lnG[k]
                         - beta_T * E_bins_centre[k];
            double w   = std::exp(log_w - log_Z);
            double e_k = E_sum[k] / static_cast<double>(H_prod[k]);
            double e2_k = E2_sum[k] / static_cast<double>(H_prod[k]);
            mean_E  += w * e_k;
            mean_E2 += w * e2_k;
        }

        double N = static_cast<double>(sc.get_objects<Spin>().size());  // geometry-normalised
        rw_mean_E[ti] = mean_E;
        rw_C[ti]      = beta_T * beta_T * (mean_E2 - mean_E * mean_E) / N;
    }

    // ---- Write HDF5 ----
    std::filesystem::path out_dir = ap.get<std::string>("--output_dir");
    const std::string prefix = ap.get<std::string>("--prefix");
    auto file_path = out_dir / make_filename_muca(L, p, dseed, mseed, n_bins, prefix);

    hid_t fid = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());

    // geometry/
    write_geometry_group(fid, sc);

    // wl/
    {
        hid_t g = make_group(fid, "wl");
        write_1d(g, "lnG",         H5T_NATIVE_DOUBLE, muca.lnG.data(), n_bins);
        write_1d(g, "E_bins_centre", H5T_NATIVE_DOUBLE, E_bins_centre.data(), n_bins);
        // bin edges
        std::vector<double> edges(n_bins + 1);
        for (int k = 0; k <= n_bins; k++) edges[k] = E_min + k * muca.bin_width;
        write_1d(g, "E_bins", H5T_NATIVE_DOUBLE, edges.data(), n_bins + 1);
        write_scalar(g, "ln_f_final",  H5T_NATIVE_DOUBLE, &ln_f);
        write_scalar(g, "n_wl_iters",  H5T_NATIVE_INT,    &wl_iter);
        H5Gclose(g);
    }

    // production/
    {
        hid_t g = make_group(fid, "production");
        write_1d(g, "H",     H5T_NATIVE_INT64,  H_prod.data(), n_bins);
        write_1d(g, "E_sum", H5T_NATIVE_DOUBLE, E_sum.data(),  n_bins);
        write_1d(g, "E2_sum",H5T_NATIVE_DOUBLE, E2_sum.data(), n_bins);
        H5Gclose(g);
    }

    // reweight/
    if (!T_list_rw.empty()) {
        hid_t g = make_group(fid, "reweight");
        hsize_t n_T = T_list_rw.size();
        write_1d(g, "T_list",  H5T_NATIVE_DOUBLE, T_list_rw.data(),  n_T);
        write_1d(g, "mean_E",  H5T_NATIVE_DOUBLE, rw_mean_E.data(),  n_T);
        write_1d(g, "C_per_N", H5T_NATIVE_DOUBLE, rw_C.data(),       n_T);
        H5Gclose(g);
    }

    H5Fclose(fid);
    std::cout << file_path << "\n";

    return 0;
}
