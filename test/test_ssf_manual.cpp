#include "energy_manager.hpp"
#include "observable_manager.hpp"
#include <argparse/argparse.hpp>
#include <unordered_set>

#include "quantum_cluster.hpp"
#include "monte_carlo.hpp"
#include "sim_bits.hpp"
#include "ssf_manager.hpp"

using namespace std;


// Test code writes a plane wave of the wave-vector
// 
// q = B Q
// where cols of B are the r.l.v. of the INDEX cell



std::string make_filename(
    int L,
    ipos_t Q,
    const std::string& prefix = "test",
    const std::string& ext = "h5")
{
    std::ostringstream oss;

    oss << prefix
        << "_L" << L
        << "_Q" << Q[0] <<"_"<<Q[1]<<"_"<<Q[2]
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


    ap.add_argument("qx")
        .help("Momentum vector x")
        .scan<'i', int>()
        .required();
    ap.add_argument("qy")
        .help("Momentum vector y")
        .scan<'i', int>()
        .required();
    ap.add_argument("qz")
        .help("Momentum vector z")
        .scan<'i', int>()
        .required();

    ap.add_argument("--unit_cell")
        .choices("primitive", "cubic")
        .default_value("cubic");

    ap.add_argument("--output_dir", "-o")
        .help("Output directory (filenames automatically generated)")
        .required();


    ap.add_argument("--verbosity", "-v")
        .help("Output verbosity: 0=silent, 1=normal, 2=+Q² spinon texture, 5=+cluster spectra")
        .default_value(1)
        .scan<'i', int>();

    ap.parse_args(argc, argv);

    // load in the parameters

    int L = ap.get<int>("L");

    auto Z = imat33_t::from_cols({L,0,0}, {0, L, 0}, {0, 0, L});

    auto cell =  (ap.get<std::string>("unit_cell") == "cubic") ? 
        make_cubic_unit_cell() : make_primitive_unit_cell();

    int verbosity = ap.get<int>("verbosity");

    Supercell sc = build_supercell<Spin,Tetra,Plaq>(cell, Z);


    std::filesystem::path out_dir = ap.get<std::string>("--output_dir");
    ivec3_t Q;
    Q[0] = ap.get<int>("qx");
    Q[1] = ap.get<int>("qy");
    Q[2] = ap.get<int>("qz");
    auto file_path = out_dir/make_filename(L, Q, "ssftest", "h5");

    hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + file_path.string());
    }

    const auto q = sc.lattice.wavevector_from_idx3(Q);
    if (verbosity > 2) {
        std::cout << "q = B \\hat{Q} = " << q << "\n";
    }

    ssf_manager ssf(sc, pyrochlore::pyrochlore_local_axes());
    ssf.set_T(1);
    for (const auto& [I, c] : sc.enumerate_cells()){
        for (auto [mu, s] : c.enumerate_objects<Spin>()){
            // sublattice-independent value
            double y = std::cos( dot<double>(q, s->ipos) );
            // finite-precision hack to avoid converting to float
            s->ising_val = (mu ==0) ?  (1<<16) * y : 0 ;
        }
    }

    ssf.sample();
    ssf.write_group(file_id, "/ssf");

    write_geometry_group(file_id, sc);
    H5Fclose(file_id);


    std::cout<<file_path<<std::endl;

    return 0;
}


