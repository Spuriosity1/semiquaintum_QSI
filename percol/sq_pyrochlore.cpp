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
    size_t seed,
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
        << "_s" << seed
        << "." << ext;

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

    ap.add_argument("--seed", "-s")
        .help("RNG seed")
        .default_value(static_cast<size_t>(0))
        .scan<'i', size_t>();
    ap.add_argument("--output_dir", "-o")
        .help("Output directory (filenames automatically generated)")
        .required();

    /// MC PROTOCOL
    ///

    ap.add_argument("--nburn")
        .help("Burn-In Iterations in the RNG sweep")
        .default_value(static_cast<size_t>(100))
        .scan<'i', size_t>();
    ap.add_argument("--nsweep", "-w")
        .help("Sweeps to establish equilibrium")
        .scan<'i', size_t>();
    ap.add_argument("--nsamp")
        .help("Sampling steps post sweep")
        .default_value(static_cast<size_t>(128))
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
    size_t seed = ap.get<size_t>("--seed");
    size_t nsweep = ap.get<size_t>("--nsweep");
    size_t nsamp = ap.get<size_t>("--nsamp");
    size_t nburn = ap.get<size_t>("--nburn");
    bool exact_boundary = !ap.get<bool>("--mf_boundary");
    int  verbosity      = ap.get<int>("--verbosity");

    MCSettings params;


    QClattice sc = initialise_lattice(L);

    // Delete about p*100% of the spins
    // (Bernoulli sample)
    std::mt19937 rng(seed);
    std::unordered_set<Tetra*> seed_tetras;
    // Identify the quantum-cluster distribution
    MCStateMF state;

    delete_spins(rng, sc, p, seed_tetras);
    identify_1o_clusters(seed_tetras, state.clusters);
    identify_flippable_hexas(sc, state.intact_plaqs);

    // initialise the clusters
    for (auto& qc : state.clusters){
        qc.initialise();
    }

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

    // Measure
    energy_manager em;
    Q_manager sm;

    double factor = exp( log(Tcold/Thot) / n_step );

    static constexpr const char* Q2_labels[4] = {"complete", "triangle", "line", "dangling"};

    for (size_t i=0; i<n_step; i++){
        const double T = 1./params.beta;
        em.new_T(T);
        sm.new_T(T);
        params.beta /= factor;

        auto t0 = std::chrono::steady_clock::now();
        for (size_t n=0; n<nsweep; n++){
            if (exact_boundary) state.sweep<true>(params);
            else                state.sweep<false>(params);
        }
        double ms_per_sweep = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count() / nsweep;

        for (size_t n=0; n<nsamp; n++){
            if (exact_boundary) state.sweep<true>(params);
            else                state.sweep<false>(params);
            em.sample(state.energy());
            sm.sample(sc);
        }

        if (verbosity >= 1) {
            params.accepted_plaq     /= state.intact_plaqs.size();
            params.accepted_classical /= state.classical_spins.size();
            params.accepted_quantum  /= state.clusters.size();
            params.accepted_boundary /= state.boundary_spins.size();

            std::cout << std::setprecision(6) << "T = " << T
                      << "\tE = " << em.curr_E()
                      << "\t" << params.acceptance()
                      << "\t" << std::fixed << std::setprecision(2) << ms_per_sweep << " ms/sweep\n";
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
                for (int k = 0; k < qc.eigenvalues.size(); k++)
                    std::cout << " " << std::setprecision(5) << qc.eigenvalues[k];
                std::cout << "  (state=" << qc.eigenstate_idx << ")\n";
            }
        }

        if (verbosity >= 1) std::cout << std::flush;
        params.reset_acceptance();
    }


    std::cout<<"Done! Writing to file... "<<std::endl;

    std::filesystem::path out_dir = ap.get<std::string>("--output_dir");
    
    auto file_path = out_dir/make_filename(L, p, seed, "run", "h5");

    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    }
    em.write_group(file_id, "energy");
    sm.write_group(file_id, "Q2");

    write_geometry_group(file_id, sc);

    H5Fclose(file_id);


    std::cout<<file_path<<std::endl;

    return 0;
}


