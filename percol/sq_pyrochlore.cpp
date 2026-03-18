#include "lattice_lib/supercell.hpp"
#include "lattice_lib/unitcell_types.hpp"
#include <queue>
//#include "unionfind.hpp"
#include "energy_manager.hpp"
#include <argparse/argparse.hpp>
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
        .help("Iterations in the RNG sweep")
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


    ap.add_argument("--include_second_order", "-a")
        .help("Includes also second order processes")
        .default_value(false)
        .implicit_value(true);

    ap.parse_args(argc, argv);



    int L = ap.get<int>("L");
    double p = ap.get<float>("p"); // site deletion probability
    size_t seed = ap.get<size_t>("--seed");
    size_t nsweep = ap.get<size_t>("--nsweep");
    size_t nburn = ap.get<size_t>("--nburn");

    MCSettings params;

    SuperLat sc = initialise_lattice(L);

    // Delete about p*100% of the spins
    // (Bernoulli sample)
    std::mt19937 rng(seed);
    std::unordered_set<Tetra*> seed_tetras;
    // Identify the quantum-cluster distribution
    MCStateMF state;

    delete_spins(rng, sc, p, seed_tetras, &state.intact_plaqs);

    identify_1o_clusters(seed_tetras, state.clusters);

    // initialise the clusters
    for (auto& qc : state.clusters){
        qc.initialise();
    }

    // Partitions spins into boundary, cluster and neighbour
    state.partition_spins(sc.get_objects<Spin>());

    // output the cluster distribution and other stats
    output_cluster_dist(std::cout, state.clusters, 1);

    std::cout<<"\n\n==========================\n"<<
        state.classical_spins.size() << " Full classical spins\n" <<
        state.boundary_spins.size() << " Boundary spins\n"<<
        state.clusters.size() << " Quantum clusters\n";

    // load in the parameters

    double Thot = ap.get<double>("--Thot");
    double Tcold = ap.get<double>("--Tcold");
    size_t n_step = ap.get<size_t>("--nstep");

    params.beta = 1./Thot;
    ModelParams::get().Jzz = ap.get<double>("--Jzz");
    ModelParams::get().Jxx = ap.get<double>("--Jxx");
    ModelParams::get().Jyy = ap.get<double>("--Jyy");

    // Burn-In
    std::cout<<"Burning in... \n";
    for (size_t n=1; n<=nburn; n++){
        std::cout<<"\r"<<n<<"/"<<nburn<<std::flush;
        state.sweep(params);
    }
    std::cout<<std::endl;

    // Measure
    energy_manager em;

    double factor = exp( log(Tcold/Thot) / n_step );

    for (size_t i=0; i<n_step; i++){
        std::cout<<"T = "<<1./params.beta<<"\t";

        em.new_T(1./params.beta);
        params.beta /= factor;

        for (size_t n=0; n<nsweep; n++){
            state.sweep(params);
            em.sample(state.energy());
        }

        params.accepted_plaq /= state.intact_plaqs.size();
        params.accepted_classical /= state.classical_spins.size();
        params.accepted_quantum /= state.clusters.size();
        params.accepted_boundary /= state.boundary_spins.size();

        std::cout<<"E = "<<em.curr_E()<<"\t"<<params.acceptance()<<std::endl;
        params.reset_acceptance();
    }

    std::filesystem::path out_dir = ap.get<std::string>("--output_dir");
    
    em.save(out_dir/make_filename(L, p, seed, "run", "h5"));


    return 0;
}


