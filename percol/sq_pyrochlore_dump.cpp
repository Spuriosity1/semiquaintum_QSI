#include "lattice_lib/supercell.hpp"
#include "lattice_lib/unitcell_types.hpp"
#include <queue>
//#include "unionfind.hpp"
#include "geometry.hpp"
#include "energy_manager.hpp"
#include <argparse/argparse.hpp>
#include <unordered_set>
#include <fstream>

#include "quantum_cluster.hpp"
#include "monte_carlo.hpp"
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
    bool use_nn4,
    const std::string& prefix = "run"
    )
{
    std::ostringstream oss;

    oss << prefix
        << "_L" << L
        << "_p" << std::fixed << std::setprecision(3) << p
        << "_s" << seed
        << "_w" << nsweep
        << (use_nn4 ? "_nn4" : "_nn2");

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
        .default_value(static_cast<size_t>(100))
        .scan<'i', size_t>();


    ap.add_argument("--include_second_order", "-a")
        .help("Includes also second order processes")
        .default_value(false)
        .implicit_value(true);

    ap.add_argument("--save_spins")
        .implicit_value(true)
        .default_value(false);

    ap.parse_args(argc, argv);



    int L = ap.get<int>("L");
    double p = ap.get<float>("p"); // site deletion probability
    size_t seed = ap.get<size_t>("--seed");

    SuperLat sc = initialise_lattice(L);

    size_t nsweep = ap.get<size_t>("--nsweep");

    std::mt19937 rng(seed);
    std::unordered_set<Tetra*> seed_tetras;


    std::map<size_t, size_t> cluster_hist;

    bool use_nn4 = ap.get<bool>("--include_second_order");


    for (size_t n=0; n<nsweep; n++){
        // Set about p*100% of the spins to "deleted" state
        // (Bernoulli sample)
        delete_spins(rng, sc, p, seed_tetras);
        // Identify the quantum-cluster distribution
        std::vector<QCluster> clusters;
        // populates state.clusters
        if (use_nn4){
            identify_quantum_clusters<QuantumRule::eq24nn>(seed_tetras, clusters);
        } else {
            identify_quantum_clusters<QuantumRule::eq2nn>(seed_tetras, clusters);
        }

        for (const auto& Q : clusters){
            size_t size = Q.spins.size();
            cluster_hist[size]++;
        }
    }


    std::filesystem::path out_dir = ap.get<std::string>("--output_dir");

    std::string file_stem = make_filestem(L, p, seed, nsweep, use_nn4, "hist");
    auto hist_fname = out_dir/(file_stem+".csv");
    std::ofstream ofs(hist_fname);
    size_t denom = L*L*L*16*nsweep;
    output_cluster_hist(ofs, cluster_hist, 1);
    output_cluster_hist(std::cout, cluster_hist, denom);

    double n_quantum=0;
    for (auto& [size, n ] : cluster_hist){
        n_quantum += n*size;
    }
    std::cout<<100*n_quantum / denom <<"% of spins are in clusters\n";

    std::cout<<"Saved hist to to "<<hist_fname.string()<<std::endl;


    if (ap.get<bool>("--save_spins")){

        std::string file_stem = make_filestem(L, p, seed, nsweep, use_nn4, "geometry");
        auto spins_fname = out_dir/(file_stem+".spins.csv");
        auto bonds_fname = out_dir/(file_stem+".bonds.csv");

        FILE* tmp_f = std::fopen(spins_fname.string().c_str(),"w");
        for (auto& s : sc.get_objects<Spin>()){
//            if (s.deleted) continue;
            ipos_t X = s.ipos;
            void* cluster_root=nullptr;
            if (s.is_quantum()){
                cluster_root = find_q_root(s.q_cluster_root);
            }

            fprintf(tmp_f, "%p\t%lld\t%lld\t%lld\t%d\t%p\n", static_cast<void*>(&s), X[0], X[1], X[2], s.deleted, cluster_root);
        }
        fclose(tmp_f);

        tmp_f = std::fopen(bonds_fname.string().c_str(),"w");
        for (auto& s : sc.get_objects<Spin>()) {   
            for (auto& nb : s.neighbours){
                if (nb > &s){
                    fprintf(tmp_f, "%p\t%p\n",
                            static_cast<void*>(&s),
                            static_cast<void*>(nb));
                }
            }
        }

        std::cout<<"Saved spins to "<<spins_fname.string()<<std::endl;
        std::cout<<"Saved bonds to "<<bonds_fname.string()<<std::endl;
    }
    

    return 0;
}


