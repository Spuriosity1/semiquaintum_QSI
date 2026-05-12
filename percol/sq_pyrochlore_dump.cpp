#include "lattice_lib/supercell.hpp"
#include "lattice_lib/unitcell_types.hpp"
#include <queue>
//#include "unionfind.hpp"
#include "geometry.hpp"
#include "energy_manager.hpp"
#include <argparse/argparse.hpp>
#include <unordered_set>
#include <hdf5.h>

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
    std::string clust_type,
    const std::string& prefix = "run"
    )
{
    std::ostringstream oss;

    oss << prefix
        << "_L" << L
        << "_p" << std::fixed << std::setprecision(3) << p
        << "_s" << seed
        << "_w" << nsweep
        << "_" << clust_type;

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


    ap.add_argument("--cluster_def", "-a")
        .help("Includes also second order processes")
        .choices("nn2", "nn24", "nn2_mf");

    ap.add_argument("--save_spins")
        .implicit_value(true)
        .default_value(false);

    ap.parse_args(argc, argv);



    int L = ap.get<int>("L");
    double p = ap.get<float>("p"); // site deletion probability
    size_t seed = ap.get<size_t>("--seed");

    QClattice sc = initialise_lattice(L);

    size_t nsweep = ap.get<size_t>("--nsweep");

    std::mt19937 rng(seed);
    std::unordered_set<Tetra*> seed_tetras;


    std::map<size_t, size_t>    cluster_hist;    // Σ_n x_n[s]
    std::map<size_t, uint64_t>  cluster_hist_sq; // Σ_n x_n[s]^2

    auto cdef = ap.get<std::string>("--cluster_def");


    for (size_t n=0; n<nsweep; n++){
        // Set about p*100% of the spins to "deleted" state
        // (Bernoulli sample)
        delete_spins(rng, sc, p, seed_tetras);
        // Identify the quantum-cluster distribution

        std::map<size_t, size_t> sweep_hist;

        if (cdef == "nn2_mf"){
            std::vector<QClusterMF> clusters;
            identify_1o_clusters(seed_tetras, clusters);
            for (const auto& Q : clusters) sweep_hist[ Q.spins.size() ]++;
        } else {
            std::vector<QCluster> clusters;
            if (cdef == "nn2"){
                identify_quantum_clusters<QuantumRule::eq2nn>(seed_tetras, clusters);
            } else if (cdef == "nn24") {
                identify_quantum_clusters<QuantumRule::eq24nn>(seed_tetras, clusters);
            }
            for (const auto& Q : clusters) sweep_hist[ Q.spins.size() ]++;
        }

        for (auto& [sz, cnt] : sweep_hist) {
            cluster_hist[sz]    += cnt;
            cluster_hist_sq[sz] += static_cast<uint64_t>(cnt) * cnt;
        }
    }


    std::filesystem::path out_dir = ap.get<std::string>("--output_dir");

    // flatten maps into parallel arrays
    std::vector<int64_t> h_sizes, h_counts, h_counts_sq;
    std::vector<double>  h_var;
    h_sizes.reserve(cluster_hist.size());
    h_counts.reserve(cluster_hist.size());
    h_counts_sq.reserve(cluster_hist.size());
    h_var.reserve(cluster_hist.size());
    for (auto& [sz, cnt] : cluster_hist) {
        const uint64_t sq  = cluster_hist_sq.count(sz) ? cluster_hist_sq.at(sz) : 0;
        const double   mu  = static_cast<double>(cnt) / nsweep;
        // Bessel-corrected variance: (Σx² - (Σx)²/N) / (N-1)
        const double   var = (static_cast<double>(sq) - static_cast<double>(cnt)*cnt / nsweep)
                             / (nsweep - 1);
        h_sizes.push_back(static_cast<int64_t>(sz));
        h_counts.push_back(static_cast<int64_t>(cnt));
        h_counts_sq.push_back(static_cast<int64_t>(sq));
        h_var.push_back(var);
        (void)mu;
    }
    const hsize_t nbins = h_sizes.size();

    const int64_t N      = static_cast<int64_t>(L)*L*L*16;
    const int64_t nsweep_i = static_cast<int64_t>(nsweep);

    std::string file_stem = make_filestem(L, p, seed, nsweep, cdef, "hist");
    auto hist_fname = out_dir / (file_stem + ".h5");

    hid_t file_id = H5Fcreate(hist_fname.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0)
        throw std::runtime_error("Failed to create HDF5 file: " + hist_fname.string());

    // write 1-D arrays
    {
        hid_t space = H5Screate_simple(1, &nbins, nullptr);
        auto write_i64 = [&](const char* name, const int64_t* data) {
            hid_t ds = H5Dcreate2(file_id, name, H5T_NATIVE_INT64, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            H5Dclose(ds);
        };
        auto write_f64 = [&](const char* name, const double* data) {
            hid_t ds = H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            H5Dclose(ds);
        };
        write_i64("sizes",      h_sizes.data());
        write_i64("counts",     h_counts.data());
        write_i64("counts_sq",  h_counts_sq.data());
        write_f64("var",        h_var.data());
        H5Sclose(space);
    }

    // write scalars
    {
        hid_t space = H5Screate(H5S_SCALAR);
        auto write_scalar = [&](const char* name, int64_t val) {
            hid_t ds = H5Dcreate2(file_id, name, H5T_NATIVE_INT64, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);
            H5Dclose(ds);
        };
        write_scalar("nsweep", nsweep_i);
        write_scalar("N",      N);
        H5Sclose(space);
    }

    H5Fclose(file_id);

    double n_quantum = 0;
    for (auto& [sz, cnt] : cluster_hist) n_quantum += cnt * sz;
    const size_t denom = static_cast<size_t>(N) * nsweep;
    std::cout << 100.0 * n_quantum / denom << "% of spins are in clusters\n";
    std::cout << "Saved hist to " << hist_fname.string() << std::endl;


    if (ap.get<bool>("--save_spins")){

        std::string file_stem = make_filestem(L, p, seed, nsweep, cdef, "geometry");
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


