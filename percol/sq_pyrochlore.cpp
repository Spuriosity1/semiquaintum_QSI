#include "lattice_lib/supercell.hpp"
#include "lattice_lib/unitcell_types.hpp"
#include <queue>
//#include "unionfind.hpp"
#include "energy_manager.hpp"
#include "observable_manager.hpp"
#include <argparse/argparse.hpp>
#include <chrono>
#include <unordered_set>

#include "quantum_cluster.hpp"
#include "monte_carlo.hpp"
#include "sim_bits.hpp"

using namespace std;


// Semi-quantum classical MC for disorderd pyrochlores.
// 
//
// 1. Removes spins from the lattice at random.
// 2. Identifies 'quantum clusters' of spins



std::string make_filename(
    int L,
    float p,
    size_t dseed,
    size_t mseed,
    const std::string& prefix = "run",
    const std::string& ext = "dat")
{
    std::ostringstream oss;

    const auto& param = ModelParams::get();

    oss << prefix
        << "_L" << L
        << "_p" << std::fixed << std::setprecision(3) << p
        << "_Jzz" << param.Jzz
        << "_Jxx" << param.Jxx
        << "_Jyy" << param.Jyy
        << "_ds" << dseed
        << "_ms" << mseed
        << "." << ext;

    return oss.str();
}


// ---------------------------------------------------------------------------
// Parallel-tempering helpers
// ---------------------------------------------------------------------------

struct ReplicaState {
    std::vector<int> spin_vals;   // ising_val: classical_spins, then boundary_spins
    std::vector<int> eigenstates; // eigenstate_idx per cluster
};

static ReplicaState save_state(const MCStateMF& state) {
    ReplicaState rs;
    rs.spin_vals.reserve(state.classical_spins.size() + state.boundary_spins.size());
    for (auto s : state.classical_spins) rs.spin_vals.push_back(s->ising_val);
    for (auto s : state.boundary_spins)  rs.spin_vals.push_back(s->ising_val);
    rs.eigenstates.reserve(state.clusters.size());
    for (const auto& qc : state.clusters) rs.eigenstates.push_back(qc.eigenstate_idx);
    return rs;
}

static void load_state(MCStateMF& state, const ReplicaState& rs) {
    size_t j = 0;
    for (auto s : state.classical_spins) s->ising_val = rs.spin_vals[j++];
    for (auto s : state.boundary_spins)  s->ising_val = rs.spin_vals[j++];
    for (size_t c = 0; c < state.clusters.size(); c++) {
        state.clusters[c].sync();   // reads boundary ising_vals, re-diagonalises
        state.clusters[c].eigenstate_idx = rs.eigenstates[c];
    }
}

// ---------------------------------------------------------------------------

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

    ap.add_argument("--Jzz")
        .help("ZZ (classical) coupling strength")
        .default_value((double) 1.0)
        .scan<'g', double>();

    ap.add_argument("--Jxx")
        .help("XX coupling strength on quantum clusters")
        .default_value((double) 0.1)
        .scan<'g', double>();

    ap.add_argument("--Jyy")
        .help("YY coupling strength on quantum clusters")
        .default_value((double) 0.1)
        .scan<'g', double>();

    ap.add_argument("--include_second_order", "-a")
        .help("Includes also second order processes")
        .default_value(false)
        .implicit_value(true);
/// BOOK-KEEPING

    ap.add_argument("--dseed", "-s")
        .help("Disorder seed (controls which spins are deleted)")
        .default_value(static_cast<size_t>(0))
        .scan<'i', size_t>();
    ap.add_argument("--mseed", "-m")
        .help("MC RNG seed (controls thermal Monte Carlo trajectory)")
        .default_value(static_cast<size_t>(0))
        .scan<'i', size_t>();
    ap.add_argument("--prefix")
        .default_value("run");
    ap.add_argument("--output_dir", "-o")
        .help("Output directory (filenames automatically generated)")
        .required();

    /// MC PROTOCOL
    ///

    ap.add_argument("--nburn")
        .help("Burn-In Iterations in the RNG sweep")
        .default_value(static_cast<size_t>(128))
        .scan<'i', size_t>();
    ap.add_argument("--nsweep", "-w")
        .help("Iterations in the sweep")
        .default_value(static_cast<size_t>(16))
        .scan<'i', size_t>();

    ap.add_argument("--nsamp")
        .help("Sampling steps in the sweep")
        .default_value(static_cast<size_t>(16))
        .scan<'i', size_t>();
    ap.add_argument("--nstep")
        .help("Iterations in the RNG sweep")
        .default_value(static_cast<size_t>(100))
        .scan<'i', size_t>();

    ap.add_argument("--Thot")
        .help("Beginning Temperature")
        .default_value(1.0)
        .scan<'g', double>();
    ap.add_argument("--Tcold")
        .help("Final Temperature")
        .default_value(0.001)
        .scan<'g', double>();


    ap.add_argument("--mf_boundary")
        .help("Use approximate eigenvalue shift for boundary spin flips (faster but less accurate than default)")
        .default_value(false)
        .implicit_value(true);

    ap.add_argument("--classical")
        .help("Skip quantum cluster identification — treat all spins as classical (benchmarking)")
        .default_value(false)
        .implicit_value(true);

    ap.add_argument("--moves")
        .help("MC moves to attempt, any combination of R(ing) C(lassical) W(orm) M(onopole) B(oundary) Q(uantum). Default: all enabled.")
        .default_value(std::string("CRWBQ"));

    ap.add_argument("--n_replicas")
        .help("Number of parallel-tempering replicas (1 = standard annealing)")
        .default_value(1)
        .scan<'i', int>();

    ap.add_argument("--n_replica_swaps")
        .help("Number of attempted replica exchanges per temperature")
        .default_value(1)
        .scan<'i', int>();

    ap.add_argument("--verbosity", "-v")
        .help("Output verbosity: 0=silent, 1=normal, 2=+Q² spinon texture, 5=+cluster spectra")
        .default_value(1)
        .scan<'i', int>();

    ap.parse_args(argc, argv);

    // load in the parameters

    double Thot = ap.get<double>("--Thot");
    double Tcold = ap.get<double>("--Tcold");
    size_t n_step = ap.get<size_t>("--nstep");

    ModelParams::get().Jzz = ap.get<double>("--Jzz");
    ModelParams::get().Jxx = ap.get<double>("--Jxx");
    ModelParams::get().Jyy = ap.get<double>("--Jyy");


    int L = ap.get<int>("L");
    double p = ap.get<float>("p"); // site deletion probability
    size_t dseed = ap.get<size_t>("--dseed");
    size_t mseed = ap.get<size_t>("--mseed");
    size_t nsweep = ap.get<size_t>("--nsweep");
    size_t nsamp = ap.get<size_t>("--nsamp");
    size_t nburn = ap.get<size_t>("--nburn");
    bool exact_boundary  = !ap.get<bool>("--mf_boundary");
    bool classical_only  = ap.get<bool>("--classical");
    int  verbosity       = ap.get<int>("--verbosity");
    int  n_replicas      = ap.get<int>("--n_replicas");
    int  n_replica_swaps = ap.get<int>("--n_replica_swaps");

    MCSettings params;
    params.moves = parse_moves(ap.get<std::string>("--moves"));


    QClattice sc = initialise_lattice(L);

    // Delete about p*100% of the spins
    // (Bernoulli sample)
    auto hashf = std::hash<size_t>();
    std::mt19937 rng(hashf(dseed));
    params.rng.seed(hashf(mseed ^ (dseed * 105 ) ) );
    std::unordered_set<Tetra*> seed_tetras;
    // Identify the quantum-cluster distribution
    MCStateMF state;

    delete_spins(rng, sc, p, seed_tetras);
    if (!classical_only)
        identify_1o_clusters(seed_tetras, state.clusters);
    identify_flippable_hexas(sc, state.intact_plaqs);

    for (auto& qc : state.clusters) qc.initialise();

    // Partitions spins into boundary, cluster and neighbour
    state.partition_spins(sc.get_objects<Spin>());

    if (verbosity >= 1) {
        output_cluster_dist(std::cout, state.clusters, 1);

        int exp_Jzz_bonds = calc_GS_energy(sc.get_objects<Tetra>());
        std::cout << "Expected ground state energy: " << exp_Jzz_bonds << "Jzz = "
                  << exp_Jzz_bonds * ModelParams::get().Jzz << "\n";

        std::cout << "\n==========================\n"
                  << state.classical_spins.size() << " Full classical spins\n"
                  << state.boundary_spins.size()  << " Boundary spins\n"
                  << state.clusters.size()         << " Quantum clusters\n";
    }

    // Burn-In
    params.beta = 1./Thot;
    if (verbosity >= 1) std::cout << "Burning in...\n";
    for (size_t n=1; n<=nburn; n++){
        if (verbosity >= 1) std::cout << "\r" << n << "/" << nburn << std::flush;
        if (exact_boundary) state.sweep<true>(params);
        else                state.sweep<false>(params);
    }
    if (verbosity >= 1) std::cout << std::endl;

    static constexpr const char* Q2_labels[4] = {"complete", "triangle", "line", "dangling"};

    // Helper: run one sweep respecting exact_boundary template param.
    auto do_sweep = [&]() {
        if (exact_boundary) state.sweep<true>(params);
        else                state.sweep<false>(params);
    };

    std::filesystem::path out_dir = ap.get<std::string>("--output_dir");
    const std::string prefix  = ap.get<std::string>("--prefix") + (classical_only ? "_class" : "");
    auto file_path = out_dir/make_filename(L, p, dseed, mseed, prefix, "h5");

    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    }

    if (n_replicas <= 1) {
        ///////////////////////////////////////////////////////////////////////
        /// STANDARD ANNEALING LOOP ///////////////////////////////////////////

        energy_manager em;
        Q_manager sm;
        double factor = exp( log(Tcold/Thot) / n_step );

        for (size_t i=0; i<n_step; i++){
            params.beta /= factor;
            const double T = 1./params.beta;
            em.new_T(T);
            sm.new_T(T);

            for (size_t n=0; n<nburn; n++) do_sweep();

            for (size_t n=0; n<nsamp; n++){
                for (size_t n=0; n<nsweep; n++) do_sweep();
                em.sample(state.energy());
                sm.sample(sc);
            }

            if (verbosity >= 1) {
                if (!state.intact_plaqs.empty())    params.accepted_plaq      /= state.intact_plaqs.size();
                if (!state.classical_spins.empty()) params.accepted_classical  /= state.classical_spins.size();
                if (!state.clusters.empty())        params.accepted_quantum    /= state.clusters.size();
                if (!state.boundary_spins.empty())  params.accepted_boundary   /= state.boundary_spins.size();

                std::cout << "Step "<<i+1<<std::setprecision(6) << " T = " << T
                          << "\tE = " << em.curr_E()
                          << "\t" << params.acceptance() <<"\n";
            }

            if (verbosity >= 2) {
                auto q2 = sm.curr_Q2();
                std::cout << "  Q2:";
                for (int k = 0; k < 4; k++)
                    std::cout << "  " << Q2_labels[k] << "=" << std::setprecision(4) << q2[k];
                std::cout << "\n";
            }

            if (verbosity >= 5) {
                std::cout << "  Cluster spectra:\n";
                for (size_t c = 0; c < state.clusters.size(); c++) {
                    const auto& qc = state.clusters[c];
                    std::cout << "    [" << c << "] n=" << qc.n_spins() << " evals:";
                    for (int k = 0; k < (int)qc.eigenvalues.size(); k++)
                        std::cout << " " << std::setprecision(5) << qc.eigenvalues[k];
                    std::cout << "  (state=" << qc.eigenstate_idx << ")\n";
                }
            }

            if (verbosity >= 1) std::cout << std::flush;
            params.reset_acceptance();
        }

        std::cout << "Done! Writing to file... " << std::endl;

        em.write_group(file_id, "energy");
        sm.write_group(file_id, "Q2");

    } else {
        ///////////////////////////////////////////////////////////////////////
        /// SIMULTANEOUS ANNEALING WITH REPLICA EXCHANGE /////////////////////
        //
        // n_replicas replicas tile the log-T range in equal slices.
        // Replica r starts at T_r(0) = Thot * (Tcold/Thot)^{r/n_replicas} and
        // anneals for n_step/n_replicas steps, ending exactly where replica r+1
        // began.  Together they cover [Thot, Tcold] with n_step total T-points.

        // Initial betas: staggered by 1/n_replicas of the full log range.
        std::vector<double> betas(n_replicas);
        for (int r = 0; r < n_replicas; r++) {
            double T_r = Thot * std::pow(Tcold / Thot, (double)r / n_replicas);
            betas[r] = 1.0 / T_r;
        }

        // Cooling factor — same per-step rate as the standard annealing branch.
        // After n_step/n_replicas steps each replica cools by (Tcold/Thot)^{1/n_replicas}.
        const double factor = std::exp(std::log(Tcold / Thot) / n_step);
        const size_t steps_per_replica = n_step / n_replicas;

        // Initialise one snapshot per replica from the current (post-burn-in) state.
        std::vector<ReplicaState> replicas(n_replicas, save_state(state));

        // Burn-in each replica at its initial temperature.
        if (verbosity >= 1) std::cout << "PT burn-in per replica...\n";
        for (int r = 0; r < n_replicas; r++) {
            load_state(state, replicas[r]);
            params.beta = betas[r];
            for (size_t n = 0; n < nburn; n++) do_sweep();
            replicas[r] = save_state(state);
            if (verbosity >= 1) std::cout << "\r  replica " << r+1 << "/" << n_replicas << std::flush;
        }
        if (verbosity >= 1) std::cout << "\n";

        // One energy_manager per replica; each accumulates steps_per_replica T-entries.
        // Total T-points across all replicas = n_step, same as the single-replica run.
        std::vector<energy_manager> ems(n_replicas);

        std::vector<int64_t> swap_accepted(n_replicas - 1, 0);
        std::vector<int64_t> swap_attempted(n_replicas - 1, 0);

        std::uniform_real_distribution<double> uni(0.0, 1.0);

        for (size_t t = 0; t < steps_per_replica; t++) {

            // Advance all temperatures by one annealing step.
            for (int r = 0; r < n_replicas; r++) betas[r] /= factor;

            for (int j=0; j<n_replica_swaps; j++){

                // Thermalise + sample each replica at its current temperature.
                for (int r = 0; r < n_replicas; r++) {
                    if (verbosity >= 1){
                        std::cout << "replica " << r+1 
                            << " run "<<j+1<<"/"<<n_replica_swaps
                            <<" @ T="<<1.0/betas[r]<< std::endl;
                    }

                    load_state(state, replicas[r]);
                    params.beta = betas[r];
                    ems[r].new_T(1.0 / betas[r]);
                    for (size_t n = 0; n < nburn; n++) do_sweep();
                    for (size_t n = 0; n < nsamp; n++) {
                        for (size_t n = 0; n < nsweep; n++) do_sweep();
                        ems[r].sample(state.energy());
                    }
                    replicas[r] = save_state(state);
                }

                // Attempt swaps between all adjacent replica pairs.
                for (int r = 0; r < n_replicas - 1; r++) {
                    load_state(state, replicas[r]);
                    double E_r = state.energy();
                    load_state(state, replicas[r+1]);
                    double E_r1 = state.energy();

                    double log_acc = (betas[r+1] - betas[r]) * (E_r - E_r1);
                    swap_attempted[r]++;
                    if (log_acc >= 0.0 || uni(params.rng) < std::exp(log_acc)) {
                        std::swap(replicas[r], replicas[r+1]);
                        swap_accepted[r]++;
                    }
                }
            }

            if (verbosity >= 1) {
                for (int r = 0; r < n_replicas; r++)
                    std::cout << "T=" << std::setprecision(4) << (1.0/betas[r])
                              << " E=" << std::setprecision(5) << ems[r].curr_E() << "  ";
                std::cout << "\n  swaps:";
                for (int r = 0; r < n_replicas - 1; r++)
                    std::cout << " " << swap_accepted[r] << "/" << swap_attempted[r];
                std::cout << "\n" << std::flush;
            }
        }

        std::cout << "Done! Writing to file... " << std::endl;

        // Append each replica's T-entries into the single "energy" group (r=0
        // first so T runs hot→cold, matching the standard annealing format).
        for (int r = 0; r < n_replicas; r++)
            ems[r].write_group(file_id, "energy");

        // Write PT swap statistics.
        {
            hid_t pt_group = H5Gcreate2(file_id, "pt", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            const hsize_t dims[1] = { (hsize_t)(n_replicas - 1) };
            hid_t space = H5Screate_simple(1, dims, nullptr);
            auto write_ds = [&](const char* name, const void* data) {
                hid_t ds = H5Dcreate2(pt_group, name, H5T_NATIVE_INT64, space,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
                H5Dclose(ds);
            };
            write_ds("swap_accepted",  swap_accepted.data());
            write_ds("swap_attempted", swap_attempted.data());
            H5Sclose(space);
            H5Gclose(pt_group);
        }
    }

    write_geometry_group(file_id, sc);
    H5Fclose(file_id);


    std::cout<<file_path<<std::endl;

    return 0;
}


