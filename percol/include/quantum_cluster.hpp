#pragma once



#include "lattice_lib/supercell.hpp"
#include "lattice_lib/unitcell_types.hpp"
#include "lattice_lib/unitcellspec.hpp"
#include <Eigen/Dense>

struct ModelParams {
    double Jzz = 1.0;
    double Jxx = 0.5;
    double Jyy = 0.5;
//    double h = 0;

    static ModelParams& get() {
        static ModelParams instance;
        return instance;
    }
private:
    ModelParams() = default;
};


struct QCluster;
struct Tetra;

struct Spin {
    ipos_t ipos;
    std::vector<Spin*> neighbours;
    std::array<Tetra*, 2> owning_tetras; // sl index
    bool deleted=false;
    Spin* q_cluster_root=nullptr; // null = not quantum
    QCluster* owning_cluster=nullptr;


    bool is_quantum() const {
        return q_cluster_root != nullptr;
    }

    void reset(){
        q_cluster_root=nullptr;
        owning_cluster=nullptr;
    }

    // state for Monte Carlo
    int ising_val = 1; // all up, why not

    // constructors
    Spin() : ipos(0,0,0){ }
    Spin(const ipos_t& x) : ipos(x) {}
};

struct Tetra {
    // static data 
    ipos_t ipos;
    std::vector<Spin*> member_spins;

    struct Neighbour {
        Tetra* tet;
        Spin* via;
    };

    // dynamically updated at runtime
    std::vector<Neighbour> neighbours;
    bool is_complete;


    // constructors
    Tetra() : ipos(0,0,0) { }
    Tetra(const ipos_t& x) : ipos(x) {}
};





struct QCluster{
    using BoundaryConfig = uint32_t; // bitmask over boundary spins

    std::vector<Spin*> spins;
    std::vector<Spin*> boundary_spins;

    // --- Base Hamiltonian (does not include boundary spins) ---
    Eigen::MatrixXd H_base;

    BoundaryConfig boundary_config = 0; // bit i = spin of boundary_spins[i] (+1 if set)
    
    // --- current spectrum (recomputed whenever boundary_config changes) ---
    Eigen::VectorXd eigenvalues;        // sorted ascending
    Eigen::MatrixXd eigenvectors;       // columns are eigenvectors

    // -- current state for MC --
    int eigenstate_idx;

    int n_spins()    const { return (int)spins.size(); }
    int hilbert_dim() const { return 1 << n_spins(); }

    // map spin pointer to its index in spins[]
    int spin_index(const Spin* s) const {
        for (int i = 0; i < (int)spins.size(); i++)
            if (spins[i] == s) return i;
        return -1;
    }

    // +1 or -1 for boundary spin i given current config
    int boundary_val(int i) const {
        return (boundary_config >> i) & 1 ? +1 : -1;
    }

    // --- build H_base: ZZ bonds + transverse field, no boundary ---
    void build_matrix_rep();
    void diagonalise(BoundaryConfig config);

    double energy() const {
        return eigenvalues[eigenstate_idx];
    }

    double partition_function(double beta) const {
        double Z=0;
        for (auto& E : eigenvalues){
            Z += exp(-beta * E);
        }
        return Z;
    }
};

const static int MAX_CLUSTER_SPINS = 16;

using MyCell = UnitCellSpecifier<Spin, Tetra>;
using SuperLat = Supercell<Spin, Tetra>;
